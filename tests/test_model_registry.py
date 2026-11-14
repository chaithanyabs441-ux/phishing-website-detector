import hashlib

import joblib
import numpy as np
import pytest

from src.serving.model_registry import ModelRegistry, _apply_calibrator
from src.evaluation.calibration_validation import apply_calibrator


def test_registry_checks_manifest_hash_before_loading(tmp_path):
    artifact = tmp_path / "safe.joblib"
    joblib.dump({"trusted": True}, artifact)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    registry = ModelRegistry.__new__(ModelRegistry)
    registry.root = tmp_path.resolve()
    registry.entries = {"safe.joblib": {"sha256": digest}}
    assert registry._load_verified("safe.joblib") == {"trusted": True}
    registry.entries["safe.joblib"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 verification failed"):
        registry._load_verified("safe.joblib")


def test_registry_rejects_paths_outside_project_root(tmp_path):
    registry = ModelRegistry.__new__(ModelRegistry)
    registry.root = tmp_path.resolve()
    registry.entries = {}
    with pytest.raises(ValueError, match="escapes the project root"):
        registry._load_verified("../outside.joblib")


def test_serving_platt_transform_matches_phase7_calibration():
    from webapp.app import REGISTRY

    raw_probabilities = [0.01, 0.5, 0.83662053, 0.99958644]
    expected = apply_calibrator(REGISTRY.calibrator, raw_probabilities)
    actual = _apply_calibrator(REGISTRY.calibrator, raw_probabilities)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_registry_routes_by_html_presence():
    from webapp.app import REGISTRY

    url_only = REGISTRY.predict("https://example.org/account")
    url_html = REGISTRY.predict(
        "https://example.org/account",
        "<html><body><form><input type='password'></form></body></html>",
    )

    assert url_only["model_key"] == "url_only"
    assert url_only["backend"] == "catboost_url_only"
    assert url_html["model_key"] == "url_html"
    assert url_html["backend"] == "catboost_url_html"
    assert 0.0 <= url_only["probability"] <= 1.0
    assert 0.0 <= url_html["probability"] <= 1.0


def test_primary_inference_failure_falls_back_for_one_request_without_url_text():
    class IntermittentModel:
        def __init__(self):
            self.calls = 0

        def predict_proba(self, _features):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("failed for https://private.example/path")
            return np.array([[0.2, 0.8]])

    class FallbackVectorizer:
        def transform(self, urls):
            return urls

    class FallbackModel:
        def decision_function(self, _features):
            return np.array([-0.25])

    registry = ModelRegistry.__new__(ModelRegistry)
    model = IntermittentModel()
    registry.primary_models = {"url_only": model, "url_html": None}
    registry.calibrators = {
        "url_only": {"method": "isotonic", "model": type("Identity", (), {"predict": lambda _self, values: values})()},
        "url_html": None,
    }
    registry.primary_error = None
    registry.fallback_vectorizer = FallbackVectorizer()
    registry.fallback_model = FallbackModel()
    registry.fallback_threshold = 1.5
    registry._catboost_features = lambda _url, _html, _key: (
        object(), {"url_length": 10}, None,
    )

    first = registry.predict("https://private.example/path")
    second = registry.predict("https://private.example/path")

    assert first["backend"] == "url_tfidf_fallback"
    assert "https://private.example/path" not in first["fallback_reason"]
    assert registry.primary_models["url_only"] is model
    assert second["backend"] == "catboost_url_only"
    assert model.calls == 2
