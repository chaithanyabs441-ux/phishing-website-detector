"""Hash-verified model loading and inference for trusted local artifacts."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import load
from scipy.special import logit

from src.evaluation.phase4_data import CATEGORICAL_FEATURES, PHASE4_FEATURE_COLUMNS
from src.features.html_features_v2 import (
    HTML_FEATURE_COLUMNS_V2,
    MAX_HTML_CHARS,
    _empty_features,
    extract_html_features_v2,
)
from src.features.url_features import URL_FEATURE_COLUMNS_V2, extract_url_features_v2


NUMERIC_HTML_COLUMNS = [
    column for column in HTML_FEATURE_COLUMNS_V2 if column != "cleaned_text"
]


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _apply_calibrator(calibrator, raw_probability):
    values = np.asarray(raw_probability, dtype=np.float64).reshape(-1, 1)
    estimator = calibrator["model"]
    method = calibrator["method"]
    if method == "platt":
        features = logit(np.clip(values, 1e-8, 1.0 - 1e-8))
        return estimator.predict_proba(features)[:, 1]
    if method == "isotonic":
        return estimator.predict(values.ravel())
    raise ValueError("Unsupported calibration method in trusted artifact")


class ModelRegistry:
    """Loads one calibrated primary model and a verified URL-SVM fallback."""

    def __init__(self, project_root, config_path=None, thresholds_path=None):
        self.root = Path(project_root).resolve()
        config_path = Path(config_path or self.root / "configs" / "model_config.yaml")
        thresholds_path = Path(thresholds_path or self.root / "configs" / "thresholds.yaml")
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.threshold_config = yaml.safe_load(thresholds_path.read_text(encoding="utf-8"))
        manifest_path = self.root / "models" / "MANIFEST.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.entries = {item["file"].replace("\\", "/"): item for item in self.manifest["artifacts"]}
        self.fallback_vectorizer = self._load_verified(self.config["serving"]["fallback_vectorizer"])
        self.fallback_model = self._load_verified(self.config["serving"]["fallback_classifier"])
        self.fallback_threshold = float(self.config["thresholds"]["url_tfidf"])
        self.primary_models = {}
        self.calibrators = {}
        self.primary_error = None
        for key in ("url_only", "url_html"):
            try:
                self.primary_models[key] = self._load_verified(
                    self.config["serving"][f"{key}_model"],
                )
                self.calibrators[key] = self._load_verified(
                    self.config["serving"][f"{key}_calibrator"],
                )
            except Exception as error:
                self.primary_models[key] = None
                self.calibrators[key] = None
                if self.primary_error is None:
                    self.primary_error = "{}: {}".format(type(error).__name__, error)

        # Keep these aliases for callers that inspect the HTML-capable model.
        self.primary_model = self.primary_models.get("url_html")
        self.calibrator = self.calibrators.get("url_html")

    def _load_verified(self, relative_path):
        normalized = str(relative_path).replace("\\", "/")
        path = (self.root / normalized).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError("Artifact path escapes the project root") from error
        manifest_item = self.entries.get(normalized)
        if manifest_item is None and normalized.startswith("models/"):
            manifest_item = self.entries.get(normalized[len("models/"):])
        if manifest_item is None:
            raise ValueError("Artifact is not listed in models/MANIFEST.json")
        if not path.is_file():
            raise FileNotFoundError("Required model artifact is missing: {}".format(normalized))
        actual_hash = _sha256(path)
        if actual_hash.lower() != manifest_item["sha256"].lower():
            raise ValueError("SHA-256 verification failed for {}".format(normalized))
        return load(path)

    @property
    def active_backend(self):
        active = [name for name, model in self.primary_models.items() if model is not None]
        return "catboost_{}".format("+".join(active)) if active else "url_tfidf_fallback"

    def _catboost_features(self, url, html, model_key):
        url_features = extract_url_features_v2(url)
        html_text = html if isinstance(html, str) else ""
        if model_key == "url_only":
            row = pd.DataFrame([url_features], columns=URL_FEATURE_COLUMNS_V2)
            for column in CATEGORICAL_FEATURES:
                row[column] = row[column].fillna("__MISSING__").astype(str)
            numeric = [column for column in URL_FEATURE_COLUMNS_V2 if column not in CATEGORICAL_FEATURES]
            row[numeric] = row[numeric].apply(pd.to_numeric, errors="coerce").fillna(0)
            return row, url_features, None

        try:
            html_features = extract_html_features_v2(html_text, url)
        except Exception:
            html_features = _empty_features(
                html_length=len(html_text),
                parsed_length=min(len(html_text), MAX_HTML_CHARS),
                html_missing=int(not html_text.strip()),
                truncated=int(len(html_text) > MAX_HTML_CHARS),
                parse_error=1,
            )
        row = pd.DataFrame(
            [{**url_features, **html_features}],
            columns=[*URL_FEATURE_COLUMNS_V2, *NUMERIC_HTML_COLUMNS],
        )
        for column in CATEGORICAL_FEATURES:
            row[column] = row[column].fillna("__MISSING__").astype(str)
        numeric = [column for column in PHASE4_FEATURE_COLUMNS if column not in CATEGORICAL_FEATURES]
        row[numeric] = row[numeric].apply(pd.to_numeric, errors="coerce").fillna(0)
        return row[PHASE4_FEATURE_COLUMNS], url_features, html_features

    def predict(self, url, html=""):
        model_key = "url_html" if isinstance(html, str) and html.strip() else "url_only"
        model = self.primary_models.get(model_key)
        calibrator = self.calibrators.get(model_key)
        fallback_reason = self.primary_error
        if model is not None and calibrator is not None:
            try:
                features, url_features, html_features = self._catboost_features(url, html, model_key)
                raw_probability = float(model.predict_proba(features)[0, 1])
                probability = float(_apply_calibrator(calibrator, [raw_probability])[0])
                probability = min(1.0, max(0.0, probability))
                return {
                    "backend": "catboost_{}".format(model_key),
                    "model_key": model_key,
                    "calibrated": True,
                    "probability": probability,
                    "raw_probability": raw_probability,
                    "url_features": url_features,
                    "html_features": html_features,
                    "fallback_reason": None,
                }
            except Exception as error:
                # Use this model's fallback for this request only. Keep the
                # loaded primary available for the next request, and do not
                # retain exception text that could contain the submitted URL.
                fallback_reason = "primary_model_inference_error: {}".format(
                    type(error).__name__,
                )

        url_features = extract_url_features_v2(url)
        vector = self.fallback_vectorizer.transform([url])
        score = float(self.fallback_model.decision_function(vector)[0])
        return {
            "backend": "url_tfidf_fallback",
            "calibrated": False,
            "probability": None,
            "decision_score": score,
            "url_features": url_features,
            "html_features": None,
            "fallback_reason": fallback_reason,
        }
