"""Classical URL/HTML text representations and registered-domain OOF scores."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from scipy.special import logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

from src.data.engineered_feature_dataset import HTML_CACHE_ROOT, HTML_FEATURE_HASH, load_phase4_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEXT_ARTIFACT_DIR = PROJECT_ROOT / "models" / "phase5"

MODEL_SPECS = {
    "char_svc_2_5": ("url", "tfidf_char_2_5", "linear_svc"),
    "char_svc_3_6": ("url", "tfidf_char_3_6", "linear_svc"),
    "char_logistic_liblinear": ("url", "tfidf_char_2_5", "logistic_liblinear"),
    "char_logistic_saga": ("url", "tfidf_char_2_5", "logistic_saga"),
    "hash_sgd_modified_huber": ("url", "hashing_char", "sgd_modified_huber"),
    "hash_sgd_log_loss": ("url", "hashing_char", "sgd_log_loss"),
    "char_complement_nb": ("url", "tfidf_char_2_5", "complement_nb"),
    "char_multinomial_nb": ("url", "tfidf_char_2_5", "multinomial_nb"),
    "char_nbsvm": ("url", "tfidf_char_2_5", "nbsvm"),
    "html_word_svc": ("cleaned_html", "tfidf_word_1_2", "linear_svc"),
    "char_svc_2_5_min_df1": ("url", "tfidf_char_2_5_min_df1", "linear_svc"),
    "char_svc_2_5_min_df3": ("url", "tfidf_char_2_5_min_df3", "linear_svc"),
}


class NBSVMClassifier(BaseEstimator, ClassifierMixin):
    """Linear SVM weighted by the log-count ratio of positive/negative terms."""

    def __init__(self, c=1.0, class_weight="balanced", max_iter=5000, random_state=42):
        self.c = c
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, features, targets):
        targets = np.asarray(targets, dtype=np.int8)
        positive = targets == 1
        negative = ~positive
        positive_counts = np.asarray(features[positive].sum(axis=0)).ravel()
        negative_counts = np.asarray(features[negative].sum(axis=0)).ravel()
        positive_ratio = (positive_counts + 1.0) / (positive.sum() + 1.0)
        negative_ratio = (negative_counts + 1.0) / (negative.sum() + 1.0)
        self.r_ = np.log(positive_ratio / negative_ratio)
        self.model_ = LinearSVC(
            C=self.c, class_weight=self.class_weight, max_iter=self.max_iter,
            random_state=self.random_state, dual=True,
        )
        self.model_.fit(features.multiply(self.r_), targets)
        self.classes_ = self.model_.classes_
        return self

    def decision_function(self, features):
        return self.model_.decision_function(features.multiply(self.r_))

    def predict(self, features):
        return self.model_.predict(features.multiply(self.r_))


def make_vectorizer(vectorizer_name, overrides=None):
    params = dict(overrides or {})
    if vectorizer_name.startswith("tfidf_char"):
        ngram_range = (3, 6) if vectorizer_name == "tfidf_char_3_6" else (2, 5)
        min_df = 1 if vectorizer_name.endswith("min_df1") else 3 if vectorizer_name.endswith("min_df3") else 2
        return TfidfVectorizer(
            analyzer="char", ngram_range=ngram_range, min_df=min_df,
            max_features=200_000, sublinear_tf=True, **params,
        )
    if vectorizer_name == "hashing_char":
        return HashingVectorizer(
            analyzer="char", ngram_range=(2, 5), n_features=2**20,
            alternate_sign=False, norm="l2", binary=True, **params,
        )
    if vectorizer_name == "tfidf_word_1_2":
        return TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2,
            max_features=150_000, sublinear_tf=True, **params,
        )
    raise ValueError(f"Unsupported vectorizer: {vectorizer_name}")


def make_classifier(classifier_name, seed=42, overrides=None):
    params = dict(overrides or {})
    if classifier_name == "linear_svc":
        return LinearSVC(
            C=1.0, class_weight="balanced", max_iter=5000, random_state=seed,
            dual=True, **params,
        )
    if classifier_name == "logistic_liblinear":
        return LogisticRegression(
            C=1.0, class_weight="balanced", solver="liblinear", max_iter=1000,
            random_state=seed, **params,
        )
    if classifier_name == "logistic_saga":
        return LogisticRegression(
            C=1.0, class_weight="balanced", solver="saga", max_iter=1000,
            tol=1e-3, random_state=seed, n_jobs=1, **params,
        )
    if classifier_name == "sgd_modified_huber":
        return SGDClassifier(
            loss="modified_huber", class_weight="balanced", max_iter=2000,
            tol=1e-3, random_state=seed, **params,
        )
    if classifier_name == "sgd_log_loss":
        return SGDClassifier(
            loss="log_loss", class_weight="balanced", max_iter=2000,
            tol=1e-3, random_state=seed, **params,
        )
    if classifier_name == "complement_nb":
        return ComplementNB(alpha=0.1, **params)
    if classifier_name == "multinomial_nb":
        return MultinomialNB(alpha=0.1, **params)
    if classifier_name == "nbsvm":
        return NBSVMClassifier(random_state=seed, **params)
    raise ValueError(f"Unsupported classifier: {classifier_name}")


def score_estimator(model, features):
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features), dtype=np.float64)
    probabilities = model.predict_proba(features)
    if probabilities.shape[1] == 2:
        return np.log(np.clip(probabilities[:, 1], 1e-12, 1.0)) - np.log(
            np.clip(probabilities[:, 0], 1e-12, 1.0)
        )
    return logit(np.clip(probabilities[:, 1], 1e-12, 1.0 - 1e-12))


def _view_values(frame, view):
    column = "url" if view == "url" else "cleaned_text"
    if column not in frame:
        raise ValueError(f"frame has no {column} field for {view} model")
    return frame[column].fillna("").astype(str).to_numpy()


def fit_text_model(model_name, train_frame, validation_frame=None, seed=42, save_path=None):
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown text model: {model_name}")
    view, vectorizer_name, classifier_name = MODEL_SPECS[model_name]
    train_values = _view_values(train_frame, view)
    targets = train_frame["target"].to_numpy(dtype=np.int8)
    vectorizer = make_vectorizer(vectorizer_name)
    fit_started = time.monotonic()
    train_features = vectorizer.fit_transform(train_values)
    vectorizer_seconds = time.monotonic() - fit_started
    model = make_classifier(classifier_name, seed)
    fit_started = time.monotonic()
    model.fit(train_features, targets)
    classifier_seconds = time.monotonic() - fit_started
    validation_scores = None
    validation_transform_seconds = None
    inference_seconds_per_1000 = None
    if validation_frame is not None:
        transform_started = time.monotonic()
        validation_features = vectorizer.transform(_view_values(validation_frame, view))
        validation_transform_seconds = time.monotonic() - transform_started
        inference_rows = min(1000, len(validation_frame))
        inference_started = time.perf_counter()
        score_estimator(model, validation_features[:inference_rows])
        inference_seconds_per_1000 = (
            time.perf_counter() - inference_started
        ) / max(1, inference_rows) * 1000
        validation_scores = score_estimator(model, validation_features)
    matrix_bytes = int(
        train_features.data.nbytes + train_features.indices.nbytes + train_features.indptr.nbytes
    )
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dump({"model_name": model_name, "view": view, "vectorizer": vectorizer, "model": model}, path)
    return vectorizer, model, validation_scores, {
        "view": view,
        "vectorizer": vectorizer_name,
        "classifier": classifier_name,
        "fit_rows": int(len(targets)),
        "feature_count": int(train_features.shape[1]),
        "nonzero_features": int(train_features.nnz),
        "sparse_matrix_bytes": matrix_bytes,
        "vectorizer_seconds": vectorizer_seconds,
        "classifier_seconds": classifier_seconds,
        "validation_transform_seconds": validation_transform_seconds,
        "inference_seconds_per_1000": inference_seconds_per_1000,
    }


def fit_text_models_shared(model_names, train_frame, validation_frame, seed=42,
                           artifact_dir=TEXT_ARTIFACT_DIR):
    """Reuse each train-fitted sparse representation across compatible classifiers."""
    grouped = {}
    for model_name in model_names:
        if model_name not in MODEL_SPECS:
            raise ValueError(f"Unknown text model: {model_name}")
        view, vectorizer_name, classifier_name = MODEL_SPECS[model_name]
        grouped.setdefault((view, vectorizer_name), []).append((model_name, classifier_name))

    targets = train_frame.target.to_numpy(dtype=np.int8)
    results = {}
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for (view, vectorizer_name), candidates in grouped.items():
        vectorizer = make_vectorizer(vectorizer_name)
        fit_started = time.monotonic()
        train_features = vectorizer.fit_transform(_view_values(train_frame, view))
        vectorizer_seconds = time.monotonic() - fit_started
        transform_started = time.monotonic()
        validation_features = vectorizer.transform(_view_values(validation_frame, view))
        validation_transform_seconds = time.monotonic() - transform_started
        matrix_bytes = int(
            train_features.data.nbytes
            + train_features.indices.nbytes
            + train_features.indptr.nbytes
        )

        for model_name, classifier_name in candidates:
            model = make_classifier(classifier_name, seed)
            fit_started = time.monotonic()
            model.fit(train_features, targets)
            classifier_seconds = time.monotonic() - fit_started
            scores = score_estimator(model, validation_features)
            inference_rows = min(1000, len(validation_frame))
            inference_started = time.perf_counter()
            score_estimator(model, validation_features[:inference_rows])
            inference_seconds_per_1000 = (
                time.perf_counter() - inference_started
            ) / max(1, inference_rows) * 1000
            artifact = artifact_dir / f"{model_name}.joblib"
            dump({
                "model_name": model_name,
                "view": view,
                "vectorizer": vectorizer,
                "model": model,
            }, artifact)
            results[model_name] = {
                "scores": scores,
                "metadata": {
                    "view": view,
                    "vectorizer": vectorizer_name,
                    "classifier": classifier_name,
                    "fit_rows": int(len(targets)),
                    "feature_count": int(train_features.shape[1]),
                    "nonzero_features": int(train_features.nnz),
                    "sparse_matrix_bytes": matrix_bytes,
                    "vectorizer_seconds": vectorizer_seconds,
                    "vectorizer_shared_with": [name for name, _ in candidates if name != model_name],
                    "classifier_seconds": classifier_seconds,
                    "validation_transform_seconds": validation_transform_seconds,
                    "inference_seconds_per_1000": inference_seconds_per_1000,
                },
                "artifact": str(artifact),
                "artifact_bytes": artifact.stat().st_size,
            }
        del train_features, validation_features, vectorizer
    return results


def grouped_oof_text_scores(
    model_name,
    train_frame,
    validation_frame,
    seed=42,
    folds=5,
):
    """Fit vectorizers/models inside grouped folds and assign one score per row."""
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown text model: {model_name}")
    view, vectorizer_name, classifier_name = MODEL_SPECS[model_name]
    values = _view_values(train_frame, view)
    targets = train_frame["target"].to_numpy(dtype=np.int8)
    groups = train_frame["group_domain"].fillna("").astype(str).to_numpy()
    if len(np.unique(groups)) < folds:
        raise ValueError(f"at least {folds} domains are required for grouped OOF")
    scores = np.full(len(train_frame), np.nan, dtype=np.float64)
    assignments = np.zeros(len(train_frame), dtype=np.int8)
    splitter = GroupKFold(n_splits=folds)
    for fit_indices, score_indices in splitter.split(values, targets, groups):
        if set(groups[fit_indices]) & set(groups[score_indices]):
            raise AssertionError("registered domains overlap in an OOF fold")
        vectorizer = make_vectorizer(vectorizer_name)
        fit_features = vectorizer.fit_transform(values[fit_indices])
        score_features = vectorizer.transform(values[score_indices])
        model = make_classifier(classifier_name, seed)
        model.fit(fit_features, targets[fit_indices])
        scores[score_indices] = score_estimator(model, score_features)
        assignments[score_indices] += 1
    if not np.all(assignments == 1) or np.isnan(scores).any():
        raise AssertionError("every row must receive exactly one grouped OOF text score")
    _, full_model, validation_scores, metadata = fit_text_model(
        model_name, train_frame, validation_frame, seed,
    )
    metadata.update({
        "oof_folds": folds,
        "oof_rows": int(len(scores)),
        "oof_assignment_min": int(assignments.min()),
        "oof_assignment_max": int(assignments.max()),
    })
    return scores, validation_scores, full_model, metadata


def load_phase5_data():
    """Attach cached cleaned title/text to the SHA-aligned Phase 4 training frame."""
    frame = load_phase4_data()
    frame = frame[frame["split"].isin(["train", "validation"])].reset_index(drop=True)
    manifest_path = HTML_CACHE_ROOT / HTML_FEATURE_HASH / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wanted = set(frame.loc[frame["split"].isin(["train", "validation"]), "sha256"].astype(str))
    text_parts = []
    cache_dir = manifest_path.parent
    for shard in manifest["shards"].values():
        part = pd.read_parquet(cache_dir / shard["output_file"], columns=["sha256", "cleaned_text"])
        part["sha256"] = part["sha256"].astype(str)
        text_parts.append(part[part["sha256"].isin(wanted)])
    cleaned = pd.concat(text_parts, ignore_index=True)
    if cleaned["sha256"].duplicated().any():
        raise ValueError("cleaned HTML cache has repeated SHA-256 values")
    merged = frame.merge(cleaned, on="sha256", how="left", validate="one_to_one", indicator=True)
    if (merged["_merge"] != "both").any():
        raise ValueError("cleaned HTML cache is missing temporal train/validation rows")
    return merged.drop(columns="_merge")