"""Compare classical URL and cleaned-HTML text models on blocked splits."""

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from joblib import dump, load
from scipy.special import logit

from src.evaluation.metrics import _domain_bootstrap, _metric_bundle
from src.training.phase4_boosting import (
    fit_with_early_stopping,
    grouped_tuning_split,
    make_estimator,
    prepare_feature_frames,
)
from src.data.engineered_feature_dataset import PHASE4_FEATURE_COLUMNS, PROJECT_ROOT
from src.training.text_models import (
    MODEL_SPECS,
    fit_text_model,
    fit_text_models_shared,
    grouped_oof_text_scores,
    load_phase5_data,
)


CONFIG_PATH = PROJECT_ROOT / "configs" / "model_comparison_phase5.yaml"
PILOT_PATH = PROJECT_ROOT / "experiments" / "phase5_text_pilot.json"
RESULT_PATH = PROJECT_ROOT / "experiments" / "phase5_text_results.json"
ARTIFACT_DIR = PROJECT_ROOT / "models" / "phase5"


def _experiment_identity(config_path, seed):
    split_path = PROJECT_ROOT / "data" / "splits" / "MANIFEST.json"
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
        ).strip(),
        "split_manifest_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "seed": seed,
    }


def _metric_details(frame, scores, bootstrap_replicates, seed):
    targets = frame["target"].to_numpy(dtype=np.int8)
    domains = frame["group_domain"].fillna("").astype(str)
    unique_domains, domain_codes = np.unique(domains, return_inverse=True)
    point = _metric_bundle(targets, scores)
    intervals, completed = _domain_bootstrap(
        targets, scores, domain_codes, len(unique_domains), bootstrap_replicates, seed,
    )
    return point, intervals, completed, int(len(unique_domains))


def _paired_ap_bootstrap(y_true, baseline_scores, hybrid_scores, domain_codes, n_domains,
                         replicates, seed):
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(replicates):
        sampled = rng.integers(0, n_domains, size=n_domains)
        weights = np.bincount(sampled, minlength=n_domains)[domain_codes]
        if len(np.unique(y_true[weights > 0])) < 2:
            continue
        baseline_ap = _metric_bundle(y_true, baseline_scores, weights)["average_precision"]
        hybrid_ap = _metric_bundle(y_true, hybrid_scores, weights)["average_precision"]
        differences.append(hybrid_ap - baseline_ap)
    if not differences:
        raise RuntimeError("no valid paired domain-bootstrap replicates")
    return {
        "point_difference": float(
            _metric_bundle(y_true, hybrid_scores)["average_precision"]
            - _metric_bundle(y_true, baseline_scores)["average_precision"]
        ),
        "ci_95": [float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))],
        "replicates_completed": len(differences),
    }


def _fit_catboost_text_hybrid(best_text_model, train, validation, text_train_scores,
                              text_validation_scores, seed, bootstrap_replicates):
    phase4_path = PROJECT_ROOT / "experiments" / "phase4_gbm_results.json"
    phase4 = json.loads(phase4_path.read_text(encoding="utf-8"))
    params = phase4["tuning"]["catboost"]["best_params"]
    baseline = phase4["models"]["catboostengineered"]
    hybrid_train = train.copy()
    hybrid_validation = validation.copy()
    hybrid_train["oof_text_score"] = text_train_scores
    hybrid_validation["oof_text_score"] = text_validation_scores
    x_train, (x_validation,) = prepare_feature_frames(
        hybrid_train, hybrid_validation, additional_numeric=("oof_text_score",),
    )
    y_train = train.target.to_numpy(dtype=np.int8)
    y_validation = validation.target.to_numpy(dtype=np.int8)
    model = make_estimator(
        "catboost", params, seed=seed, n_jobs=4,
        positive_weight=float((y_train == 0).sum() / max(1, (y_train == 1).sum())),
    )
    training_started = time.monotonic()
    fit_with_early_stopping(
        "catboost", model, x_train, y_train, x_validation, y_validation,
    )
    training_seconds = time.monotonic() - training_started
    probabilities = model.predict_proba(x_validation)[:, 1]
    scores = logit(np.clip(probabilities, 1e-12, 1.0 - 1e-12))
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = ARTIFACT_DIR / "catboost_best_text_oof_hybrid.joblib"
    dump(model, artifact)
    point, intervals, completed, domain_count = _metric_details(
        validation, scores, bootstrap_replicates, seed,
    )
    baseline_model_path = PROJECT_ROOT / baseline["artifact"]
    if not baseline_model_path.exists():
        raise FileNotFoundError(f"Phase 4 CatBoost artifact is missing: {baseline_model_path}")
    baseline_model = load(baseline_model_path)
    baseline_probabilities = baseline_model.predict_proba(
        x_validation[PHASE4_FEATURE_COLUMNS]
    )[:, 1]
    baseline_scores = logit(np.clip(baseline_probabilities, 1e-12, 1.0 - 1e-12))
    domains = validation.group_domain.fillna("").astype(str)
    _, domain_codes = np.unique(domains, return_inverse=True)
    paired = _paired_ap_bootstrap(
        y_validation, baseline_scores, scores, domain_codes, domain_count,
        bootstrap_replicates, seed + 100,
    )
    inference_rows = min(1000, len(x_validation))
    inference_started = time.perf_counter()
    model.predict_proba(x_validation.iloc[:inference_rows])
    inference_seconds_per_1000 = (
        time.perf_counter() - inference_started
    ) / inference_rows * 1000
    return {
        "model": "catboost_best_text_oof_hybrid",
        "text_model": best_text_model,
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "training_seconds": training_seconds,
        "inference_seconds_per_1000": inference_seconds_per_1000,
        "model_size_bytes": artifact.stat().st_size,
        "artifact": str(artifact.relative_to(PROJECT_ROOT)),
        "metrics": point,
        "domain_bootstrap_95ci": intervals,
        "bootstrap_replicates_completed": completed,
        "paired_ap_difference_vs_catboost_engineered": paired,
    }


def run_text_pilot(config_path=None, model_names=None):
    config_path = Path(config_path or CONFIG_PATH)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    data = load_phase5_data()
    train = data[data["split"] == "train"].reset_index(drop=True)
    fit_indices, valid_indices = grouped_tuning_split(
        train, seed=seed, fit_rows=int(config["pilot"]["fit_rows"]),
        validation_rows=int(config["pilot"]["validation_rows"]),
    )
    fit_frame = train.iloc[fit_indices].reset_index(drop=True)
    validation_frame = train.iloc[valid_indices].reset_index(drop=True)
    results = {}
    started = time.monotonic()
    selected_models = model_names or config["pilot"]["models"]
    for model_name in selected_models:
        if model_name not in MODEL_SPECS:
            raise ValueError(f"Unknown pilot model {model_name}")
        _, _, scores, metadata = fit_text_model(
            model_name, fit_frame, validation_frame, seed=seed,
        )
        metrics = _metric_bundle(validation_frame.target.to_numpy(dtype=np.int8), scores)
        results[model_name] = {"metadata": metadata, "metrics": metrics}
        print(
            f"{model_name}: AP={metrics['average_precision']:.6f}; "
            f"fit_s={metadata['vectorizer_seconds'] + metadata['classifier_seconds']:.2f}; "
            f"inference_s_per_1k={metadata['inference_seconds_per_1000']:.4f}; "
            f"matrix_MB={metadata['sparse_matrix_bytes'] / 1e6:.1f}"
        )
    identity = _experiment_identity(config_path, seed)
    existing = {}
    pilot_runs = []
    if PILOT_PATH.exists():
        previous = json.loads(PILOT_PATH.read_text(encoding="utf-8"))
        if (
            previous.get("config_sha256") == identity["config_sha256"]
            and previous.get("split_manifest_sha256") == identity["split_manifest_sha256"]
            and previous.get("seed") == seed
        ):
            existing = previous.get("models", {})
            pilot_runs = previous.get("pilot_runs", [])
    existing.update(results)
    pilot_runs.append({
        "models": list(selected_models),
        "elapsed_seconds": time.monotonic() - started,
    })
    output = {
        "experiment": "phase5_text_runtime_pilot",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **identity,
        "html_feature_spec_hash": "f5434603820e65699462bc276cc34b2c6ba4ad7ed72f9f061a1cc0d13d79e6d3",
        "fit_rows": int(len(fit_frame)),
        "validation_rows": int(len(validation_frame)),
        "fit_domains": int(fit_frame.group_domain.nunique()),
        "validation_domains": int(validation_frame.group_domain.nunique()),
        "models": existing,
        "pilot_runs": pilot_runs,
        "elapsed_seconds": sum(run["elapsed_seconds"] for run in pilot_runs),
        "official_test_accessed": False,
    }
    PILOT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Pilot elapsed seconds: {output['elapsed_seconds']:.2f}")
    print(f"Wrote experiment record: {PILOT_PATH}")
    return output


def run_text_comparison(config_path=None, model_names=None, bootstrap_replicates=None):
    started = time.monotonic()
    config_path = Path(config_path or CONFIG_PATH)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    bootstrap_replicates = int(bootstrap_replicates or config["full"]["bootstrap_replicates"])
    model_names = model_names or list(config["text_models"])
    data = load_phase5_data()
    train = data[data["split"] == "train"].reset_index(drop=True)
    validation = data[data["split"] == "validation"].reset_index(drop=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    shared_fits = fit_text_models_shared(
        model_names, train, validation, seed=seed, artifact_dir=ARTIFACT_DIR,
    )
    for index, model_name in enumerate(model_names):
        fit_result = shared_fits[model_name]
        scores = fit_result["scores"]
        metadata = fit_result["metadata"]
        artifact = Path(fit_result["artifact"])
        point, intervals, completed, domain_count = _metric_details(
            validation, scores, bootstrap_replicates, seed + index,
        )
        results[model_name] = {
            "metadata": metadata,
            "artifact": str(artifact.relative_to(PROJECT_ROOT)),
            "artifact_bytes": fit_result["artifact_bytes"],
            "validation_rows": int(len(validation)),
            "validation_domains": domain_count,
            "metrics": point,
            "domain_bootstrap_95ci": intervals,
            "bootstrap_replicates_completed": completed,
        }
        print(
            f"{model_name}: AP={point['average_precision']:.6f}; "
            f"ROC-AUC={point['roc_auc']:.6f}; fit_s="
            f"{metadata['vectorizer_seconds'] + metadata['classifier_seconds']:.1f}; "
            f"infer_s_per_1k={metadata['inference_seconds_per_1000']:.4f}"
        )
    best_text_model = max(
        results, key=lambda name: results[name]["metrics"]["average_precision"],
    )
    oof_train_scores, oof_validation_scores, _, oof_metadata = grouped_oof_text_scores(
        best_text_model, train, validation, seed=seed,
        folds=int(config["full"]["oof_folds"]),
    )
    hybrid_result = _fit_catboost_text_hybrid(
        best_text_model, train, validation, oof_train_scores,
        oof_validation_scores, seed, bootstrap_replicates,
    )
    results["catboost_best_text_oof_hybrid"] = hybrid_result
    output = {
        "experiment": "phase5_classical_text_models",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **_experiment_identity(config_path, seed),
        "html_feature_spec_hash": "f5434603820e65699462bc276cc34b2c6ba4ad7ed72f9f061a1cc0d13d79e6d3",
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "best_text_model_by_temporal_ap": best_text_model,
        "grouped_oof_text": oof_metadata,
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
    print(f"Total elapsed seconds: {output['elapsed_seconds']:.2f}")
    print(f"Wrote experiment record: {RESULT_PATH}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_SPECS))
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    args = parser.parse_args()
    if args.pilot:
        run_text_pilot(args.config, args.models)
    else:
        run_text_comparison(args.config, args.models, args.bootstrap_replicates)


if __name__ == "__main__":
    main()