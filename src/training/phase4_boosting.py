"""Domain-blocked tuning utilities for classical Phase 4 tree models."""

import argparse
import hashlib
import json
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from joblib import dump
from scipy.special import logit
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from src.data.engineered_feature_dataset import (
    CATEGORICAL_FEATURES,
    HTML_FEATURE_HASH,
    PHASE4_FEATURE_COLUMNS,
    load_phase4_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_DIR = PROJECT_ROOT / "experiments" / "phase4_studies"
TRIAL_DIR = PROJECT_ROOT / "experiments"
RESULT_PATH = TRIAL_DIR / "phase4_gbm_results.json"
PILOT_RESULT_PATH = TRIAL_DIR / "phase4_tuning_pilot.json"
ARTIFACT_DIR = PROJECT_ROOT / "models" / "phase4"


def _stratified_cap(indices, targets, maximum, seed):
    if len(indices) <= maximum:
        return np.asarray(indices, dtype=np.int64)
    selected, _ = train_test_split(
        indices,
        train_size=maximum,
        random_state=seed,
        stratify=targets[indices],
    )
    return np.sort(selected)


def grouped_tuning_split(
    frame,
    seed=42,
    domain_holdout_fraction=0.2,
    fit_rows=100_000,
    validation_rows=25_000,
):
    """Return bounded positional indices with no registered-domain overlap."""
    if "target" not in frame or "group_domain" not in frame:
        raise ValueError("frame must contain target and group_domain columns")
    targets = frame["target"].to_numpy(dtype=np.int8)
    groups = frame["group_domain"].fillna("").astype(str).to_numpy()
    if len(np.unique(targets)) != 2:
        raise ValueError("grouped tuning data must contain both classes")
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=domain_holdout_fraction, random_state=seed,
    )
    fit_indices, validation_indices = next(
        splitter.split(np.zeros(len(frame)), targets, groups)
    )
    fit_indices = _stratified_cap(fit_indices, targets, fit_rows, seed)
    validation_indices = _stratified_cap(
        validation_indices, targets, validation_rows, seed + 1,
    )
    fit_domains = set(groups[fit_indices])
    validation_domains = set(groups[validation_indices])
    if fit_domains & validation_domains:
        raise AssertionError("registered domains overlap in tuning split")
    if len(np.unique(targets[fit_indices])) != 2 or len(np.unique(targets[validation_indices])) != 2:
        raise ValueError("domain holdout must contain both labels on each side")
    return fit_indices, validation_indices


def prepare_feature_frames(train_frame, *evaluation_frames, additional_numeric=()):
    """Fit missing-value and categorical vocabularies on train rows only."""
    columns = [*PHASE4_FEATURE_COLUMNS, *additional_numeric]
    train_features = train_frame[columns].copy()
    evaluations = [frame[columns].copy() for frame in evaluation_frames]
    for column in CATEGORICAL_FEATURES:
        train_values = train_features[column].fillna("__MISSING__").astype(str)
        categories = sorted(set(train_values) | {"__UNKNOWN__"})
        train_features[column] = pd.Categorical(train_values, categories=categories)
        for features in evaluations:
            values = features[column].fillna("__MISSING__").astype(str)
            values = values.where(values.isin(categories), "__UNKNOWN__")
            features[column] = pd.Categorical(values, categories=categories)

    numeric_columns = [column for column in columns if column not in CATEGORICAL_FEATURES]
    medians = train_features[numeric_columns].median()
    train_features[numeric_columns] = train_features[numeric_columns].fillna(medians).fillna(0)
    for features in evaluations:
        features[numeric_columns] = features[numeric_columns].fillna(medians).fillna(0)
    return train_features, evaluations


def generate_oof_text_scores(
    train_frame,
    validation_frame,
    seed=42,
    folds=5,
    vectorizer_kwargs=None,
    classifier_kwargs=None,
):
    """Generate grouped OOF URL-text margins and full-train holdout margins."""
    groups = train_frame["group_domain"].fillna("").astype(str).to_numpy()
    if len(np.unique(groups)) < folds:
        raise ValueError(f"at least {folds} training domains are required for text OOF")
    urls = train_frame["url"].fillna("").astype(str).to_numpy()
    targets = train_frame["target"].to_numpy(dtype=np.int8)
    scores = np.full(len(train_frame), np.nan, dtype=np.float64)
    assignments = np.zeros(len(train_frame), dtype=np.int8)
    vectorizer_params = {
        "analyzer": "char", "ngram_range": (2, 5), "min_df": 2,
        "max_features": 200_000, "sublinear_tf": True,
    }
    if vectorizer_kwargs:
        vectorizer_params.update(vectorizer_kwargs)
    text_model_params = {"C": 1.0, "class_weight": "balanced", "max_iter": 5000}
    if classifier_kwargs:
        text_model_params.update(classifier_kwargs)

    splitter = GroupKFold(n_splits=folds)
    for fit_indices, oof_indices in splitter.split(urls, targets, groups):
        vectorizer = TfidfVectorizer(**vectorizer_params)
        x_fit = vectorizer.fit_transform(urls[fit_indices])
        x_oof = vectorizer.transform(urls[oof_indices])
        model = LinearSVC(dual=True, random_state=seed, **text_model_params)
        model.fit(x_fit, targets[fit_indices])
        scores[oof_indices] = model.decision_function(x_oof)
        assignments[oof_indices] += 1

    if not np.all(assignments == 1) or np.isnan(scores).any():
        raise AssertionError("every training row must receive exactly one OOF text score")

    full_vectorizer = TfidfVectorizer(**vectorizer_params)
    full_x_train = full_vectorizer.fit_transform(urls)
    full_model = LinearSVC(dual=True, random_state=seed, **text_model_params)
    full_model.fit(full_x_train, targets)
    validation_urls = validation_frame["url"].fillna("").astype(str)
    validation_scores = full_model.decision_function(full_vectorizer.transform(validation_urls))
    return scores, validation_scores, {
        "folds": folds,
        "rows": int(len(scores)),
        "oof_assignment_min": int(assignments.min()),
        "oof_assignment_max": int(assignments.max()),
        "vectorizer_params": {
            **vectorizer_params,
            "ngram_range": list(vectorizer_params["ngram_range"]),
        },
        "classifier_params": text_model_params,
    }


def suggest_parameters(trial, model_name):
    if model_name == "lightgbm":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 120),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
    if model_name == "xgboost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        }
    if model_name == "catboost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "depth": trial.suggest_int("depth", 5, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        }
    raise ValueError(f"Unsupported model: {model_name}")


def make_estimator(model_name, params, seed=42, n_jobs=4, positive_weight=1.0):
    if model_name == "lightgbm":
        return LGBMClassifier(
            **params, n_estimators=3000, objective="binary", class_weight="balanced",
            random_state=seed, n_jobs=n_jobs, verbosity=-1, subsample_freq=1,
            deterministic=True, force_col_wise=True,
        )
    if model_name == "xgboost":
        return XGBClassifier(
            **params, n_estimators=3000, objective="binary:logistic", eval_metric="aucpr",
            tree_method="hist", enable_categorical=True, scale_pos_weight=positive_weight,
            random_state=seed, n_jobs=n_jobs, early_stopping_rounds=100,
        )
    if model_name == "catboost":
        return CatBoostClassifier(
            **params, iterations=3000, loss_function="Logloss", eval_metric="PRAUC",
            auto_class_weights="Balanced", random_seed=seed, thread_count=n_jobs,
            allow_writing_files=False, verbose=False,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def fit_with_early_stopping(model_name, model, x_fit, y_fit, x_valid, y_valid):
    if model_name == "lightgbm":
        model.fit(
            x_fit, y_fit, eval_set=[(x_valid, y_valid)],
            callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
        )
    elif model_name == "xgboost":
        model.fit(x_fit, y_fit, eval_set=[(x_valid, y_valid)], verbose=False)
    else:
        model.fit(
            x_fit, y_fit, eval_set=(x_valid, y_valid),
            cat_features=CATEGORICAL_FEATURES, early_stopping_rounds=100,
            use_best_model=True, verbose=False,
        )
    return model


def _study_identity(model_name, split_hash, config_hash):
    identity = hashlib.sha256(f"{model_name}:{split_hash}:{config_hash}".encode()).hexdigest()[:12]
    return f"phase4_{model_name}_{identity}"


def tune_model(
    model_name,
    fit_frame,
    seed=42,
    n_jobs=4,
    trials=60,
    timeout_seconds=900,
    fit_rows=100_000,
    validation_rows=25_000,
    domain_holdout_fraction=0.2,
    split_hash="unrecorded",
    config_hash="unrecorded",
):
    """Tune one CPU booster on a bounded, domain-disjoint training-only subset."""
    fit_indices, validation_indices = grouped_tuning_split(
        fit_frame, seed, domain_holdout_fraction, fit_rows, validation_rows,
    )
    tuning_fit = fit_frame.iloc[fit_indices]
    tuning_valid = fit_frame.iloc[validation_indices]
    x_fit, (x_valid,) = prepare_feature_frames(tuning_fit, tuning_valid)
    y_fit = tuning_fit["target"].to_numpy(dtype=np.int8)
    y_valid = tuning_valid["target"].to_numpy(dtype=np.int8)
    positive_weight = float((y_fit == 0).sum() / max(1, (y_fit == 1).sum()))

    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    study_name = _study_identity(model_name, split_hash, config_hash)
    storage_path = (STUDY_DIR / f"{study_name}.sqlite3").resolve().as_posix()
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        load_if_exists=True,
    )

    def objective(trial):
        params = suggest_parameters(trial, model_name)
        model = make_estimator(model_name, params, seed, n_jobs, positive_weight)
        fit_with_early_stopping(model_name, model, x_fit, y_fit, x_valid, y_valid)
        score = model.predict_proba(x_valid)[:, 1]
        ap = float(average_precision_score(y_valid, score))
        trial.report(ap, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return ap

    initial_trial_count = len(study.trials)
    remaining = max(0, trials - initial_trial_count)
    started = time.monotonic()
    if remaining:
        study.optimize(
            objective, n_trials=remaining, timeout=timeout_seconds,
            n_jobs=1, gc_after_trial=True, show_progress_bar=False,
        )
    elapsed = time.monotonic() - started
    successful = [trial for trial in study.trials if trial.value is not None]
    if not successful:
        raise RuntimeError(f"No successful trials for {model_name}")
    completed_this_run = len(study.trials) - initial_trial_count
    trial_path = TRIAL_DIR / f"{study_name}_trials.csv"
    study.trials_dataframe().to_csv(trial_path, index=False)
    try:
        if len(successful) < 5:
            raise ValueError("at least five completed trials are required for importances")
        parameter_importance = optuna.importance.get_param_importances(study)
    except (ValueError, RuntimeError):
        parameter_importance = {}
    return {
        "model": model_name,
        "study_name": study_name,
        "best_value_ap": study.best_value,
        "best_params": study.best_params,
        "total_trials": len(study.trials),
        "trials_this_run": completed_this_run,
        "elapsed_seconds_this_run": elapsed,
        "study_db": str((STUDY_DIR / f"{study_name}.sqlite3").relative_to(PROJECT_ROOT)),
        "trials_csv": str(trial_path.relative_to(PROJECT_ROOT)),
        "fit_rows": int(len(tuning_fit)),
        "validation_rows": int(len(tuning_valid)),
        "fit_domains": int(tuning_fit["group_domain"].nunique()),
        "validation_domains": int(tuning_valid["group_domain"].nunique()),
        "parameter_importance": parameter_importance,
    }


def _fit_and_score(name, model_name, params, fit_frame, validation_frame, seed, n_jobs,
                   additional_numeric=(), positive_weight=1.0):
    x_fit, (x_valid,) = prepare_feature_frames(
        fit_frame, validation_frame, additional_numeric=additional_numeric,
    )
    y_fit = fit_frame["target"].to_numpy(dtype=np.int8)
    y_valid = validation_frame["target"].to_numpy(dtype=np.int8)
    model = make_estimator(model_name, params, seed, n_jobs, positive_weight)
    started = time.monotonic()
    fit_with_early_stopping(model_name, model, x_fit, y_fit, x_valid, y_valid)
    training_seconds = time.monotonic() - started
    probabilities = model.predict_proba(x_valid)[:, 1]
    scores = logit(np.clip(probabilities, 1e-12, 1.0 - 1e-12))

    artifact_dir = Path(ARTIFACT_DIR).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{name}.joblib"
    dump(model, artifact_path)
    validation_sample_size = min(5000, len(x_valid))
    sample_indices = np.random.default_rng(seed).choice(
        len(x_valid), validation_sample_size, replace=False,
    )
    permutation = permutation_importance(
        model, x_valid.iloc[sample_indices], y_valid[sample_indices],
        scoring="average_precision", n_repeats=3, random_state=seed, n_jobs=1,
    )
    if model_name == "lightgbm":
        native_importance = model.booster_.feature_importance(importance_type="gain")
        native_importance_type = "gain"
    elif model_name == "xgboost":
        importance_by_name = model.get_booster().get_score(importance_type="gain")
        native_importance = [importance_by_name.get(feature, 0.0) for feature in x_fit.columns]
        native_importance_type = "gain"
    else:
        native_importance = model.get_feature_importance(type="PredictionValuesChange")
        native_importance_type = "PredictionValuesChange"
    gain = [
        {"feature": feature, "importance": float(value)}
        for feature, value in zip(x_fit.columns, native_importance)
    ]
    gain.sort(key=lambda item: item["importance"], reverse=True)
    permutation_rows = [
        {"feature": feature, "importance_mean": float(mean), "importance_std": float(std)}
        for feature, mean, std in zip(
            x_fit.columns, permutation.importances_mean, permutation.importances_std,
        )
    ]
    permutation_rows.sort(key=lambda item: item["importance_mean"], reverse=True)

    inference_rows = min(1000, len(x_valid))
    inference_times = []
    for _ in range(3):
        started = time.perf_counter()
        model.predict_proba(x_valid.iloc[:inference_rows])
        inference_times.append(time.perf_counter() - started)
    inference_per_1000 = float(np.median(inference_times) / inference_rows * 1000)
    return {
        "model": model,
        "metrics_scores": scores,
        "metrics_probabilities": probabilities,
        "training_seconds": training_seconds,
        "inference_seconds_per_1000": inference_per_1000,
        "model_size_bytes": artifact_path.stat().st_size,
        "artifact": str(artifact_path.relative_to(PROJECT_ROOT)),
        "best_iteration": getattr(model, "best_iteration_", getattr(model, "best_iteration", None)),
        "native_importance_type": native_importance_type,
        "native_importance_top_30": gain[:30],
        "permutation_importance_top_30": permutation_rows[:30],
        "validation_rows": int(len(y_valid)),
        "training_rows": int(len(y_fit)),
    }


def _logistic_pipeline(seed, c_value):
    numeric = [
        column for column in PHASE4_FEATURE_COLUMNS
        if column not in CATEGORICAL_FEATURES
    ]
    categorical = ["public_suffix", "tld_category"]
    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=2), categorical),
    ])
    return Pipeline([
        ("preprocess", preprocessing),
        ("classifier", LogisticRegression(
            C=c_value, class_weight="balanced", max_iter=1000, solver="liblinear",
            random_state=seed, n_jobs=1,
        )),
    ])


def tune_logistic(fit_frame, seed, fit_rows, validation_rows, domain_holdout_fraction):
    fit_indices, valid_indices = grouped_tuning_split(
        fit_frame, seed, domain_holdout_fraction, fit_rows, validation_rows,
    )
    tuning_fit = fit_frame.iloc[fit_indices]
    tuning_valid = fit_frame.iloc[valid_indices]
    y_fit = tuning_fit.target.to_numpy(dtype=np.int8)
    y_valid = tuning_valid.target.to_numpy(dtype=np.int8)
    feature_columns = [
        column for column in PHASE4_FEATURE_COLUMNS
        if column != "registered_domain"
    ]
    results = []
    for c_value in (0.01, 0.1, 1.0, 10.0):
        model = _logistic_pipeline(seed, c_value)
        model.fit(tuning_fit[feature_columns], y_fit)
        score = model.predict_proba(tuning_valid[feature_columns])[:, 1]
        results.append({"C": c_value, "average_precision": float(average_precision_score(y_valid, score))})
    best = max(results, key=lambda row: row["average_precision"])
    return {
        "best_C": best["C"],
        "best_grouped_ap": best["average_precision"],
        "grid_results": results,
        "fit_rows": int(len(tuning_fit)),
        "validation_rows": int(len(tuning_valid)),
        "fit_domains": int(tuning_fit.group_domain.nunique()),
        "validation_domains": int(tuning_valid.group_domain.nunique()),
    }


def _fit_logistic(fit_frame, validation_frame, seed, c_value):
    y_fit = fit_frame["target"].to_numpy(dtype=np.int8)
    y_valid = validation_frame["target"].to_numpy(dtype=np.int8)
    feature_columns = [
        column for column in PHASE4_FEATURE_COLUMNS
        if column != "registered_domain"
    ]
    model = _logistic_pipeline(seed, c_value)
    started = time.monotonic()
    model.fit(fit_frame[feature_columns], y_fit)
    training_seconds = time.monotonic() - started
    probabilities = model.predict_proba(validation_frame[feature_columns])[:, 1]
    scores = logit(np.clip(probabilities, 1e-12, 1.0 - 1e-12))
    artifact_dir = Path(ARTIFACT_DIR).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "logistic_engineered.joblib"
    dump(model, artifact_path)
    sample_size = min(1000, len(validation_frame))
    inference_start = time.perf_counter()
    model.predict_proba(validation_frame[feature_columns].iloc[:sample_size])
    inference_seconds_per_1000 = (time.perf_counter() - inference_start) / sample_size * 1000
    return {
        "metrics_scores": scores,
        "metrics_probabilities": probabilities,
        "training_seconds": training_seconds,
        "inference_seconds_per_1000": inference_seconds_per_1000,
        "model_size_bytes": artifact_path.stat().st_size,
        "artifact": str(artifact_path.relative_to(PROJECT_ROOT)),
        "validation_rows": int(len(y_valid)),
        "training_rows": int(len(y_fit)),
    }


def run_tuning_pilot(config_path=None):
    """Measure one grouped tuning trial per booster; does not refit finalists."""
    started = time.monotonic()
    config_path = Path(config_path or PROJECT_ROOT / "configs" / "phase4_gbm.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tuning = config["tuning"]
    seed = int(config["seed"])
    data = load_phase4_data()
    fit_frame = data[data["split"] == "train"].reset_index(drop=True)
    split_manifest = PROJECT_ROOT / "data" / "splits" / "MANIFEST.json"
    split_hash = hashlib.sha256(split_manifest.read_bytes()).hexdigest()
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    results = {}
    for model_name in ("lightgbm", "xgboost", "catboost"):
        results[model_name] = tune_model(
            model_name, fit_frame, seed=seed, n_jobs=int(tuning["n_jobs"]),
            trials=1, timeout_seconds=min(300, int(tuning["timeout_seconds_per_model"])),
            fit_rows=int(tuning["fit_rows"]),
            validation_rows=int(tuning["validation_rows"]),
            domain_holdout_fraction=float(tuning["domain_holdout_fraction"]),
            split_hash=split_hash, config_hash=config_hash,
        )
        print(
            f"{model_name} pilot: AP={results[model_name]['best_value_ap']:.6f}; "
            f"trial_seconds={results[model_name]['elapsed_seconds_this_run']:.1f}"
        )
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
    ).strip()
    output = {
        "experiment": "phase4_grouped_tuning_runtime_pilot",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "seed": seed,
        "split_manifest_sha256": split_hash,
        "config_sha256": config_hash,
        "html_feature_spec_hash": HTML_FEATURE_HASH,
        "tuning": results,
        "wall_seconds": time.monotonic() - started,
        "final_refits_performed": False,
        "official_test_accessed": False,
    }
    PILOT_RESULT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote tuning pilot record: {PILOT_RESULT_PATH}")
    print(f"Pilot wall seconds: {output['wall_seconds']:.2f}")
    return output


def run_phase4(config_path=None, trials=None, timeout_seconds=None, bootstrap_replicates=None,
               include_oof_text_score=True, models=None):
    """Tune, refit, and evaluate classical boosters on training-only blocked data."""
    phase_started = time.monotonic()
    config_path = Path(config_path or PROJECT_ROOT / "configs" / "phase4_gbm.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    tuning = config["tuning"]
    n_jobs = int(tuning["n_jobs"])
    trials = int(trials or tuning["trials_per_model"])
    timeout_seconds = int(timeout_seconds or tuning["timeout_seconds_per_model"])
    bootstrap_replicates = int(bootstrap_replicates or config["bootstrap"]["replicates"])
    model_names = models or ["lightgbm", "xgboost", "catboost"]

    data = load_phase4_data()
    fit_frame = data[data["split"] == "train"].reset_index(drop=True)
    validation_frame = data[data["split"] == "validation"].reset_index(drop=True)
    if fit_frame.empty or validation_frame.empty:
        raise ValueError("frozen temporal train and validation partitions must both be nonempty")
    split_manifest_path = PROJECT_ROOT / "data" / "splits" / "MANIFEST.json"
    split_hash = hashlib.sha256(split_manifest_path.read_bytes()).hexdigest()
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    tuning_results = {}
    for model_name in model_names:
        tuning_results[model_name] = tune_model(
            model_name, fit_frame, seed=seed, n_jobs=n_jobs, trials=trials,
            timeout_seconds=timeout_seconds, fit_rows=int(tuning["fit_rows"]),
            validation_rows=int(tuning["validation_rows"]),
            domain_holdout_fraction=float(tuning["domain_holdout_fraction"]),
            split_hash=split_hash, config_hash=config_hash,
        )
        print(
            f"{model_name}: best grouped AP={tuning_results[model_name]['best_value_ap']:.6f}; "
            f"elapsed={tuning_results[model_name]['elapsed_seconds_this_run']:.1f}s"
        )
    logistic_tuning = tune_logistic(
        fit_frame, seed, int(tuning["fit_rows"]), int(tuning["validation_rows"]),
        float(tuning["domain_holdout_fraction"]),
    )

    text_scores = None
    text_oof_metadata = None
    if include_oof_text_score:
        text_started = time.monotonic()
        text_scores = generate_oof_text_scores(
            fit_frame, validation_frame, seed=seed,
            folds=int(config["text_score"]["grouped_oof_folds"]),
            vectorizer_kwargs={
                **config["text_score"]["vectorizer"],
                "ngram_range": tuple(config["text_score"]["vectorizer"]["ngram_range"]),
            },
            classifier_kwargs=config["text_score"]["classifier"],
        )
        text_oof_metadata = {**text_scores[2], "elapsed_seconds": time.monotonic() - text_started}
        fit_frame = fit_frame.copy()
        validation_frame = validation_frame.copy()
        fit_frame["oof_text_score"] = text_scores[0]
        validation_frame["oof_text_score"] = text_scores[1]

    from src.evaluation.metrics import _domain_bootstrap, _metric_bundle

    y_valid = validation_frame["target"].to_numpy(dtype=np.int8)
    domains = validation_frame["group_domain"].fillna("").astype(str)
    unique_domains, domain_codes = np.unique(domains, return_inverse=True)
    results = {}
    for model_name, tuning_result in tuning_results.items():
        for variant in (["engineered", "+oof_text_score"] if include_oof_text_score else ["engineered"]):
            name = f"{model_name}{variant}"
            scored = _fit_and_score(
                name, model_name, tuning_result["best_params"], fit_frame,
                validation_frame, seed, n_jobs,
                additional_numeric=("oof_text_score",) if variant == "+oof_text_score" else (),
                positive_weight=float((fit_frame.target == 0).sum() / max(1, (fit_frame.target == 1).sum())),
            )
            point = _metric_bundle(y_valid, scored["metrics_scores"])
            ci, completed = _domain_bootstrap(
                y_valid, scored["metrics_scores"], domain_codes, len(unique_domains),
                bootstrap_replicates, seed,
            )
            results[name] = {
                key: value for key, value in scored.items()
                if key not in {"model", "metrics_scores", "metrics_probabilities"}
            }
            results[name].update({
                "metrics": point,
                "domain_bootstrap_95ci": ci,
                "bootstrap_replicates_completed": completed,
            })
            print(
                f"{name}: AP={point['average_precision']:.6f}, "
                f"ROC-AUC={point['roc_auc']:.6f}, F1={point['f1']:.6f}, "
                f"FPR={point['false_positive_rate']:.6f}"
            )

    logistic_result = _fit_logistic(
        fit_frame, validation_frame, seed, logistic_tuning["best_C"],
    )
    logistic_point = _metric_bundle(y_valid, logistic_result["metrics_scores"])
    logistic_ci, logistic_completed = _domain_bootstrap(
        y_valid, logistic_result["metrics_scores"], domain_codes, len(unique_domains),
        bootstrap_replicates, seed + 100,
    )
    results["logistic_engineered"] = {
        key: value for key, value in logistic_result.items()
        if key not in {"metrics_scores", "metrics_probabilities"}
    }
    results["logistic_engineered"].update({
        "metrics": logistic_point,
        "domain_bootstrap_95ci": logistic_ci,
        "bootstrap_replicates_completed": logistic_completed,
    })

    svm_reference_path = PROJECT_ROOT / "experiments" / "phase2_svm_baseline.json"
    svm_reference = json.loads(svm_reference_path.read_text(encoding="utf-8"))
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
    ).strip()
    output = {
        "experiment": "phase4_classical_boosted_models",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "seed": seed,
        "split_manifest_sha256": split_hash,
        "config_sha256": config_hash,
        "html_feature_spec_hash": HTML_FEATURE_HASH,
        "training_rows": int(len(fit_frame)),
        "validation_rows": int(len(validation_frame)),
        "validation_domains": int(len(unique_domains)),
        "n_jobs": n_jobs,
        "tuning": tuning_results,
        "tuning_settings": {
            "trials_per_model": trials,
            "timeout_seconds_per_model": timeout_seconds,
            "sampler": tuning["sampler"],
            "pruner": tuning["pruner"],
            "objective": tuning["objective"],
            "fit_rows_cap": int(tuning["fit_rows"]),
            "validation_rows_cap": int(tuning["validation_rows"]),
        },
        "logistic_tuning": logistic_tuning,
        "text_oof": text_oof_metadata,
        "existing_svm_reference": svm_reference["models"],
        "bootstrap": {
            "resampling_unit": "registered_domain",
            "requested_replicates": bootstrap_replicates,
            "confidence_level": 0.95,
        },
        "models": results,
        "wall_seconds": time.monotonic() - phase_started,
        "official_test_accessed": False,
    }
    TRIAL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote experiment record: {RESULT_PATH}")
    print(f"Total wall seconds: {output['wall_seconds']:.2f}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "phase4_gbm.yaml")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    parser.add_argument("--models", nargs="+", choices=["lightgbm", "xgboost", "catboost"])
    parser.add_argument("--no-oof-text-score", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="run one grouped trial per booster for timing")
    args = parser.parse_args()
    if args.pilot:
        run_tuning_pilot(args.config)
        return
    run_phase4(
        args.config, args.trials, args.timeout_seconds, args.bootstrap_replicates,
        include_oof_text_score=not args.no_oof_text_score, models=args.models,
    )


if __name__ == "__main__":
    main()