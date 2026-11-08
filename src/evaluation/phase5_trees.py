"""Lightly tuned classical tree references for Phase 5."""

import hashlib
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from joblib import dump
from scipy.special import logit
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from src.evaluation.evaluate_svm_blocked import _domain_bootstrap, _metric_bundle
from src.evaluation.phase4_boosting import grouped_tuning_split
from src.evaluation.phase4_data import PROJECT_ROOT, load_phase4_data


CONFIG_PATH = PROJECT_ROOT / "configs" / "phase5_text.yaml"
RESULT_PATH = PROJECT_ROOT / "experiments" / "phase5_tree_results.json"
PILOT_PATH = PROJECT_ROOT / "experiments" / "phase5_tree_pilot.json"
STUDY_DIR = PROJECT_ROOT / "experiments" / "phase5_studies"
MODEL_DIR = PROJECT_ROOT / "models" / "phase5"
MODEL_NAMES = ("extra_trees", "hist_gradient_boosting")
EXCLUDED_COLUMNS = {"registered_domain", "public_suffix", "tld_category"}


def tree_feature_columns(frame):
    """Return numeric engineered columns, excluding high-cardinality/string fields."""
    columns = [
        column for column in frame.columns
        if column not in EXCLUDED_COLUMNS | {"sha256", "url", "label", "target", "split", "fold", "group_domain", "cleaned_text"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not columns:
        raise ValueError("no numeric engineered features available for tree models")
    return columns


def make_tree_model(model_name, params, seed=42, n_jobs=4):
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            **params, class_weight="balanced", n_jobs=n_jobs,
            random_state=seed, bootstrap=False,
        )
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            **params, random_state=seed, early_stopping=False,
        )
    raise ValueError(f"Unsupported tree model: {model_name}")


def suggest_tree_params(trial, model_name):
    if model_name == "extra_trees":
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300]),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.5, 1.0]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_depth": trial.suggest_categorical("max_depth", [None, 24, 40]),
        }
    if model_name == "hist_gradient_boosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 300),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 100),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        }
    raise ValueError(f"Unsupported tree model: {model_name}")


def _tune_one(model_name, fit_frame, columns, seed, config, split_hash, config_hash):
    tuning = config["tree_reference"]
    fit_indices, valid_indices = grouped_tuning_split(
        fit_frame, seed=seed, fit_rows=int(tuning["fit_rows"]),
        validation_rows=int(tuning["validation_rows"]),
    )
    tuning_fit = fit_frame.iloc[fit_indices]
    tuning_valid = fit_frame.iloc[valid_indices]
    x_fit = tuning_fit[columns].to_numpy(dtype=np.float32)
    x_valid = tuning_valid[columns].to_numpy(dtype=np.float32)
    y_fit = tuning_fit.target.to_numpy(dtype=np.int8)
    y_valid = tuning_valid.target.to_numpy(dtype=np.int8)
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
    ).strip()
    identity = hashlib.sha256(
        f"{model_name}:{split_hash}:{config_hash}:{source_commit}".encode()
    ).hexdigest()[:12]
    study_name = f"phase5_{model_name}_{identity}"
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    db_path = (STUDY_DIR / f"{study_name}.sqlite3").resolve().as_posix()
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0),
        load_if_exists=True,
    )

    def objective(trial):
        model = make_tree_model(
            model_name, suggest_tree_params(trial, model_name), seed,
            int(tuning["n_jobs"]),
        )
        model.fit(x_fit, y_fit)
        ap = float(average_precision_score(y_valid, model.predict_proba(x_valid)[:, 1]))
        trial.report(ap, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return ap

    initial_trials = len(study.trials)
    started = time.monotonic()
    study.optimize(
        objective,
        n_trials=max(0, int(tuning["trials_per_model"]) - initial_trials),
        timeout=int(tuning["timeout_seconds_per_model"]),
        n_jobs=1, gc_after_trial=True, show_progress_bar=False,
    )
    elapsed = time.monotonic() - started
    successful = [trial for trial in study.trials if trial.value is not None]
    if not successful:
        raise RuntimeError(f"No completed trials for {model_name}")
    csv_path = PROJECT_ROOT / "experiments" / f"{study_name}_trials.csv"
    study.trials_dataframe().to_csv(csv_path, index=False)
    return {
        "study_name": study_name,
        "best_params": study.best_params,
        "best_grouped_ap": float(study.best_value),
        "total_trials": len(study.trials),
        "trials_this_run": len(study.trials) - initial_trials,
        "elapsed_seconds": elapsed,
        "sqlite": str((STUDY_DIR / f"{study_name}.sqlite3").relative_to(PROJECT_ROOT)),
        "trial_csv": str(csv_path.relative_to(PROJECT_ROOT)),
        "fit_rows": int(len(tuning_fit)),
        "validation_rows": int(len(tuning_valid)),
    }


def run_phase5_trees(config_path=None, models=None, bootstrap_replicates=None, pilot=False):
    started = time.monotonic()
    config_path = Path(config_path or CONFIG_PATH)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    tuning = config["tree_reference"]
    bootstrap_replicates = int(bootstrap_replicates or config["full"]["bootstrap_replicates"])
    if pilot:
        tuning["trials_per_model"] = 1
        tuning["timeout_seconds_per_model"] = min(300, int(tuning["timeout_seconds_per_model"]))
        tuning["fit_rows"] = int(config["pilot"]["fit_rows"])
        tuning["validation_rows"] = int(config["pilot"]["validation_rows"])
    data = load_phase4_data()
    train = data[data.split == "train"].reset_index(drop=True)
    validation = data[data.split == "validation"].reset_index(drop=True)
    columns = tree_feature_columns(train)
    split_manifest = PROJECT_ROOT / "data" / "splits" / "MANIFEST.json"
    split_hash = hashlib.sha256(split_manifest.read_bytes()).hexdigest()
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    names = models or MODEL_NAMES
    tuning_results = {}
    for model_name in names:
        tuning_results[model_name] = _tune_one(
            model_name, train, columns, seed, config, split_hash, config_hash,
        )
        print(
            f"{model_name}: best grouped AP={tuning_results[model_name]['best_grouped_ap']:.6f}; "
            f"elapsed={tuning_results[model_name]['elapsed_seconds']:.1f}s"
        )

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
    ).strip()
    if pilot:
        pilot_result = {
            "experiment": "phase5_tree_runtime_pilot",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "seed": seed,
            "split_manifest_sha256": split_hash,
            "config_sha256": config_hash,
            "feature_columns": columns,
            "tuning": tuning_results,
            "final_refits_performed": False,
            "official_test_accessed": False,
            "elapsed_seconds": time.monotonic() - started,
        }
        PILOT_PATH.write_text(json.dumps(pilot_result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote tree pilot record: {PILOT_PATH}")
        return pilot_result

    results = {}
    x_train = train[columns].to_numpy(dtype=np.float32)
    x_valid = validation[columns].to_numpy(dtype=np.float32)
    y_train = train.target.to_numpy(dtype=np.int8)
    y_valid = validation.target.to_numpy(dtype=np.int8)
    validation_domains = validation.group_domain.fillna("").astype(str)
    unique_domains, domain_codes = np.unique(validation_domains, return_inverse=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for index, model_name in enumerate(names):
        model = make_tree_model(
            model_name, tuning_results[model_name]["best_params"], seed,
            int(tuning["n_jobs"]),
        )
        started = time.monotonic()
        model.fit(x_train, y_train)
        training_seconds = time.monotonic() - started
        probabilities = model.predict_proba(x_valid)[:, 1]
        scores = logit(np.clip(probabilities, 1e-12, 1.0 - 1e-12))
        artifact = MODEL_DIR / f"{model_name}.joblib"
        dump(model, artifact)
        inference_rows = min(1000, len(x_valid))
        inference_times = []
        for _ in range(3):
            inference_started = time.perf_counter()
            model.predict_proba(x_valid[:inference_rows])
            inference_times.append(time.perf_counter() - inference_started)
        inference_seconds_per_1000 = float(np.median(inference_times) / inference_rows * 1000)
        point = _metric_bundle(y_valid, scores)
        intervals, completed = _domain_bootstrap(
            y_valid, scores, domain_codes, len(unique_domains), bootstrap_replicates, seed + index,
        )
        results[model_name] = {
            "feature_columns": columns,
            "training_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "training_seconds": training_seconds,
            "inference_seconds_per_1000": inference_seconds_per_1000,
            "model_size_bytes": artifact.stat().st_size,
            "artifact": str(artifact.relative_to(PROJECT_ROOT)),
            "metrics": point,
            "domain_bootstrap_95ci": intervals,
            "bootstrap_replicates_completed": completed,
        }
        print(f"{model_name}: temporal AP={point['average_precision']:.6f}")

    output = {
        "experiment": "phase5_engineered_tree_references",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "seed": seed,
        "split_manifest_sha256": split_hash,
        "config_sha256": config_hash,
        "feature_columns": columns,
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "tuning": tuning_results,
        "bootstrap": {
            "resampling_unit": "registered_domain",
            "requested_replicates": bootstrap_replicates,
            "confidence_level": 0.95,
        },
        "models": results,
        "elapsed_seconds": time.monotonic() - started,
        "official_test_accessed": False,
    }
    RESULT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES)
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    run_phase5_trees(args.config, args.models, args.bootstrap_replicates, args.pilot)


if __name__ == "__main__":
    main()