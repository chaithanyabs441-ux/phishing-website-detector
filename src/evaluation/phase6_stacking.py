"""Leakage-guarded grouped OOF stacking for Phase 6."""

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import dump, load
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from lightgbm import LGBMClassifier

from src.evaluation.evaluate_svm_blocked import _domain_bootstrap, _metric_bundle
from src.evaluation.phase4_boosting import make_estimator, prepare_feature_frames
from src.evaluation.phase4_data import CATEGORICAL_FEATURES, PROJECT_ROOT
from src.evaluation.phase5_text import MODEL_SPECS, load_phase5_data, make_classifier, make_vectorizer, score_estimator


CONFIG_PATH = PROJECT_ROOT / "configs" / "phase6_stacking.yaml"
PHASE4_RESULTS = PROJECT_ROOT / "experiments" / "phase4_gbm_results.json"
PHASE5_RESULTS = PROJECT_ROOT / "experiments" / "phase5_text_results.json"
RESULT_PATH = PROJECT_ROOT / "experiments" / "phase6_stacking_results.json"
SHUFFLE_RESULT_PATH = PROJECT_ROOT / "experiments" / "phase6_label_shuffle.json"
OOF_PATH = PROJECT_ROOT / "data" / "processed" / "phase6_base_oof.parquet"
VALIDATION_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "phase6_validation_predictions.parquet"
ARTIFACT_DIR = PROJECT_ROOT / "models" / "phase6"


def build_grouped_fold_ledger(frame, n_splits=5):
    targets = frame["target"].to_numpy(dtype=np.int8)
    groups = frame["group_domain"].fillna("").astype(str).to_numpy()
    splitter = GroupKFold(n_splits=n_splits)
    ledger = []
    assignments = np.zeros(len(frame), dtype=np.int8)
    for fold, (fit_indices, score_indices) in enumerate(splitter.split(np.zeros(len(frame)), targets, groups)):
        fit_domains = set(groups[fit_indices])
        score_domains = set(groups[score_indices])
        if fit_domains & score_domains:
            raise AssertionError(f"registered domains overlap in fold {fold}")
        assignments[score_indices] += 1
        ledger.append((fit_indices, score_indices))
    if not np.all(assignments == 1):
        raise AssertionError("every training row must appear in exactly one OOF fold")
    return ledger, assignments


def build_meta_features(probabilities):
    """Convert ordered base probabilities to finite logit features."""
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("base probabilities must be a finite 2D matrix")
    return logit(np.clip(matrix, 1e-6, 1.0 - 1e-6))


def label_shuffle_ap(features, targets, groups, seed=42, repeats=5):
    """Measure chance-level meta performance after shuffling fit labels."""
    targets = np.asarray(targets, dtype=np.int8)
    groups = np.asarray(groups).astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    fit_indices, valid_indices = next(splitter.split(features, targets, groups))
    if set(groups[fit_indices]) & set(groups[valid_indices]):
        raise AssertionError("registered domains overlap in label-shuffle split")
    if len(np.unique(targets[fit_indices])) != 2 or len(np.unique(targets[valid_indices])) != 2:
        raise ValueError("label-shuffle split must contain both classes on both sides")
    model = LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", random_state=seed)
    model.fit(features[fit_indices], targets[fit_indices])
    unshuffled_scores = model.decision_function(features[valid_indices])
    unshuffled_ap = float(average_precision_score(targets[valid_indices], unshuffled_scores))
    shuffled_aps = []
    for repeat in range(repeats):
        shuffled_targets = np.random.default_rng(seed + repeat + 1).permutation(targets[fit_indices])
        shuffled_model = LogisticRegression(
            C=1.0, class_weight="balanced", solver="liblinear", random_state=seed + repeat,
        ).fit(features[fit_indices], shuffled_targets)
        scores = shuffled_model.decision_function(features[valid_indices])
        shuffled_aps.append(float(average_precision_score(targets[valid_indices], scores)))
    shuffled_ap = float(np.mean(shuffled_aps))
    return {
        "average_precision": shuffled_ap,
        "standard_deviation": float(np.std(shuffled_aps, ddof=1)) if len(shuffled_aps) > 1 else 0.0,
        "repeat_scores": shuffled_aps,
        "repeats": repeats,
        "unshuffled_average_precision": unshuffled_ap,
        "validation_prevalence": float(targets[valid_indices].mean()),
        "difference_from_chance": float(
            shuffled_ap - targets[valid_indices].mean()
        ),
        "fit_rows": int(len(fit_indices)),
        "validation_rows": int(len(valid_indices)),
        "fit_domains": int(len(np.unique(groups[fit_indices]))),
        "validation_domains": int(len(np.unique(groups[valid_indices]))),
        "domain_overlap": int(len(set(groups[fit_indices]) & set(groups[valid_indices]))),
    }


def paired_ap_bootstrap(y_true, baseline_scores, comparison_scores, domain_codes,
                        n_domains, replicates=200, seed=42):
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(replicates):
        sampled = rng.integers(0, n_domains, size=n_domains)
        weights = np.bincount(sampled, minlength=n_domains)[domain_codes]
        if len(np.unique(y_true[weights > 0])) < 2:
            continue
        baseline_ap = average_precision_score(y_true, baseline_scores, sample_weight=weights)
        comparison_ap = average_precision_score(y_true, comparison_scores, sample_weight=weights)
        differences.append(comparison_ap - baseline_ap)
    if not differences:
        raise RuntimeError("no valid paired domain-bootstrap samples")
    return {
        "point_difference": float(
            average_precision_score(y_true, comparison_scores)
            - average_precision_score(y_true, baseline_scores)
        ),
        "ci_95": [float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))],
        "replicates_completed": len(differences),
    }


def _set_fold_iterations(model, name, best_iteration):
    iterations = max(1, int(best_iteration or 300))
    if name in {"lightgbm", "xgboost"}:
        model.set_params(n_estimators=iterations)
    if name == "xgboost":
        model.set_params(early_stopping_rounds=None)
    elif name == "catboost":
        model.set_params(iterations=iterations)
    return model


def _fit_gbm_oof(name, train, validation, ledger, phase4, seed, n_jobs):
    settings = phase4["tuning"][name]
    reference = phase4["models"][f"{name}engineered"]
    best_iteration = reference.get("best_iteration")
    y = train.target.to_numpy(dtype=np.int8)
    oof = np.full(len(train), np.nan, dtype=np.float64)
    assignments = np.zeros(len(train), dtype=np.int8)
    for fold, (fit_indices, score_indices) in enumerate(ledger):
        fit_frame = train.iloc[fit_indices]
        score_frame = train.iloc[score_indices]
        x_fit, (x_score,) = prepare_feature_frames(fit_frame, score_frame)
        positive_weight = float((y[fit_indices] == 0).sum() / max(1, (y[fit_indices] == 1).sum()))
        model = make_estimator(name, settings["best_params"], seed + fold, n_jobs, positive_weight)
        _set_fold_iterations(model, name, best_iteration)
        if name == "catboost":
            model.fit(x_fit, y[fit_indices], cat_features=CATEGORICAL_FEATURES, verbose=False)
        else:
            model.fit(x_fit, y[fit_indices])
        oof[score_indices] = model.predict_proba(x_score)[:, 1]
        assignments[score_indices] += 1
        print(f"{name} OOF fold {fold + 1}/{len(ledger)} complete; rows={len(score_indices)}", flush=True)

    if not np.all(assignments == 1) or np.isnan(oof).any():
        raise AssertionError(f"{name} must score every training row exactly once")

    saved_model = load(PROJECT_ROOT / reference["artifact"])
    train_features, (validation_features,) = prepare_feature_frames(train, validation)
    del train_features
    validation_scores = saved_model.predict_proba(validation_features)[:, 1]
    return oof, validation_scores, {
        "fold_assignment_min": int(assignments.min()),
        "fold_assignment_max": int(assignments.max()),
        "fold_rows": [int(len(score_indices)) for _, score_indices in ledger],
        "fixed_iterations_from_phase4_temporal_fit": best_iteration,
    }


def _fit_text_oof(train, validation, ledger, model_name, seed):
    view, vectorizer_name, classifier_name = MODEL_SPECS[model_name]
    column = "url" if view == "url" else "cleaned_text"
    values = train[column].fillna("").astype(str).to_numpy()
    targets = train.target.to_numpy(dtype=np.int8)
    groups = train.group_domain.fillna("").astype(str).to_numpy()
    oof = np.full(len(train), np.nan, dtype=np.float64)
    assignments = np.zeros(len(train), dtype=np.int8)
    for fold, (fit_indices, score_indices) in enumerate(ledger):
        if set(groups[fit_indices]) & set(groups[score_indices]):
            raise AssertionError(f"text OOF domains overlap in fold {fold}")
        vectorizer = make_vectorizer(vectorizer_name)
        x_fit = vectorizer.fit_transform(values[fit_indices])
        x_score = vectorizer.transform(values[score_indices])
        model = make_classifier(classifier_name, seed)
        model.fit(x_fit, targets[fit_indices])
        oof[score_indices] = expit(score_estimator(model, x_score))
        assignments[score_indices] += 1
        print(f"{model_name} OOF fold {fold + 1}/{len(ledger)} complete; rows={len(score_indices)}", flush=True)
    if not np.all(assignments == 1) or np.isnan(oof).any():
        raise AssertionError("text base model must score every training row exactly once")

    artifact_path = PROJECT_ROOT / "models" / "phase5" / f"{model_name}.joblib"
    text_artifact = load(artifact_path)
    validation_values = validation[column].fillna("").astype(str)
    validation_scores = expit(score_estimator(
        text_artifact["model"], text_artifact["vectorizer"].transform(validation_values),
    ))
    return oof, validation_scores, {
        "fold_assignment_min": int(assignments.min()),
        "fold_assignment_max": int(assignments.max()),
        "fold_rows": [int(len(score_indices)) for _, score_indices in ledger],
        "artifact": str(artifact_path.relative_to(PROJECT_ROOT)),
    }


def _model_metrics(validation, scores, bootstrap_replicates, seed):
    targets = validation.target.to_numpy(dtype=np.int8)
    domains = validation.group_domain.fillna("").astype(str)
    unique_domains, domain_codes = np.unique(domains, return_inverse=True)
    point = _metric_bundle(targets, logit(np.clip(scores, 1e-6, 1 - 1e-6)))
    ci, completed = _domain_bootstrap(
        targets, logit(np.clip(scores, 1e-6, 1 - 1e-6)), domain_codes,
        len(unique_domains), bootstrap_replicates, seed,
    )
    return point, ci, completed


def run_saved_oof_label_shuffle(config_path=None, repeats=5):
    config_path = Path(config_path or CONFIG_PATH)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stack_record = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    oof = pd.read_parquet(OOF_PATH)
    base_names = stack_record["base_models"]
    features = build_meta_features(oof[base_names].to_numpy(dtype=np.float64))
    targets = oof.target.to_numpy(dtype=np.int8)
    groups = oof.group_domain.fillna("").astype(str).to_numpy()
    started = time.monotonic()
    diagnostic = label_shuffle_ap(
        features, targets, groups, seed=int(config["seed"]), repeats=repeats,
    )
    split_manifest = PROJECT_ROOT / "data" / "splits" / "MANIFEST.json"
    output = {
        "experiment": "phase6_grouped_oof_meta_label_shuffle",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
        ).strip(),
        "seed": int(config["seed"]),
        "repeats": repeats,
        "base_models": base_names,
        "split_manifest_sha256": hashlib.sha256(split_manifest.read_bytes()).hexdigest(),
        "stack_results_sha256": hashlib.sha256(RESULT_PATH.read_bytes()).hexdigest(),
        "oof_predictions_sha256": hashlib.sha256(OOF_PATH.read_bytes()).hexdigest(),
        "diagnostic": diagnostic,
        "elapsed_seconds": time.monotonic() - started,
        "official_test_accessed": False,
    }
    SHUFFLE_RESULT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        f"unshuffled AP={diagnostic['unshuffled_average_precision']:.6f}; "
        f"shuffled AP={diagnostic['average_precision']:.6f} +/- "
        f"{diagnostic['standard_deviation']:.6f}; "
        f"prevalence={diagnostic['validation_prevalence']:.6f}; "
        f"domain_overlap={diagnostic['domain_overlap']}"
    )
    print(f"Wrote diagnostic: {SHUFFLE_RESULT_PATH}")
    return output


def run_stacking(config_path=None):
    started = time.monotonic()
    config_path = Path(config_path or CONFIG_PATH)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    n_jobs = int(config["n_jobs"])
    bootstrap_replicates = int(config["bootstrap"]["replicates"])
    data = load_phase5_data()
    train = data[data.split == "train"].reset_index(drop=True)
    validation = data[data.split == "validation"].reset_index(drop=True)
    ledger, shared_assignments = build_grouped_fold_ledger(train, int(config["grouped_folds"]))
    phase4 = json.loads(PHASE4_RESULTS.read_text(encoding="utf-8"))
    phase5 = json.loads(PHASE5_RESULTS.read_text(encoding="utf-8"))
    base_names = ["lightgbm", "xgboost", "catboost", phase5["best_text_model_by_temporal_ap"]]
    oof_columns = {}
    validation_columns = {}
    oof_metadata = {}
    for name in base_names:
        if name == phase5["best_text_model_by_temporal_ap"]:
            train_scores, validation_scores, metadata = _fit_text_oof(
                train, validation, ledger, name, seed,
            )
        else:
            train_scores, validation_scores, metadata = _fit_gbm_oof(
                name, train, validation, ledger, phase4, seed, n_jobs,
            )
        oof_columns[name] = train_scores
        validation_columns[name] = validation_scores
        oof_metadata[name] = metadata

    oof_frame = pd.DataFrame({
        "sha256": train.sha256.astype(str),
        "target": train.target.astype(np.int8),
        "group_domain": train.group_domain.astype(str),
        "fold": np.full(len(train), -1, dtype=np.int8),
        **oof_columns,
    })
    for fold, (_, score_indices) in enumerate(ledger):
        oof_frame.loc[score_indices, "fold"] = fold
    validation_frame = pd.DataFrame({
        "sha256": validation.sha256.astype(str),
        "target": validation.target.astype(np.int8),
        "group_domain": validation.group_domain.astype(str),
        **validation_columns,
    })
    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    oof_frame.to_parquet(OOF_PATH, index=False, compression="zstd")
    VALIDATION_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    validation_frame.to_parquet(VALIDATION_PREDICTIONS_PATH, index=False, compression="zstd")

    oof_matrix = oof_frame[base_names].to_numpy(dtype=np.float64)
    validation_matrix = validation_frame[base_names].to_numpy(dtype=np.float64)
    train_meta = build_meta_features(oof_matrix)
    validation_meta = build_meta_features(validation_matrix)
    y_train = train.target.to_numpy(dtype=np.int8)
    logistic = LogisticRegression(
        C=float(config["meta"]["logistic"]["C"]),
        class_weight=config["meta"]["logistic"]["class_weight"],
        solver="liblinear", random_state=seed,
    ).fit(train_meta, y_train)
    lightgbm_meta = LGBMClassifier(
        **config["meta"]["lightgbm"], objective="binary", class_weight="balanced",
        random_state=seed, n_jobs=n_jobs, verbosity=-1,
    ).fit(train_meta, y_train)

    validation_scores = {name: validation_matrix[:, i] for i, name in enumerate(base_names)}
    validation_scores["mean_probability"] = validation_matrix.mean(axis=1)
    validation_scores["rank_average"] = np.mean(
        np.column_stack([
            rankdata(validation_matrix[:, i], method="average") / len(validation_matrix)
            for i in range(validation_matrix.shape[1])
        ]), axis=1,
    )
    logistic_started = time.perf_counter()
    validation_scores["logistic_meta"] = logistic.predict_proba(validation_meta)[:, 1]
    logistic_seconds_per_1000 = (
        time.perf_counter() - logistic_started
    ) / len(validation_meta) * 1000
    lightgbm_started = time.perf_counter()
    validation_scores["lightgbm_meta"] = lightgbm_meta.predict_proba(validation_meta)[:, 1]
    lightgbm_seconds_per_1000 = (
        time.perf_counter() - lightgbm_started
    ) / len(validation_meta) * 1000

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    dump(logistic, ARTIFACT_DIR / "logistic_meta.joblib")
    dump(lightgbm_meta, ARTIFACT_DIR / "lightgbm_meta.joblib")
    results = {}
    for index, (name, scores) in enumerate(validation_scores.items()):
        point, ci, completed = _model_metrics(validation, scores, bootstrap_replicates, seed + index)
        results[name] = {
            "metrics": point,
            "domain_bootstrap_95ci": ci,
            "bootstrap_replicates_completed": completed,
        }

    baseline = max(base_names, key=lambda name: results[name]["metrics"]["average_precision"])
    baseline_ap = results[baseline]["metrics"]["average_precision"]
    best_ensemble = max(
        ("mean_probability", "rank_average", "logistic_meta", "lightgbm_meta"),
        key=lambda name: results[name]["metrics"]["average_precision"],
    )
    y_validation = validation.target.to_numpy(dtype=np.int8)
    unique_domains, domain_codes = np.unique(
        validation.group_domain.fillna("").astype(str), return_inverse=True,
    )
    paired_gain = paired_ap_bootstrap(
        y_validation, validation_scores[baseline], validation_scores[best_ensemble],
        domain_codes, len(unique_domains), bootstrap_replicates, seed,
    )
    correlation = pd.DataFrame(validation_matrix, columns=base_names).corr().to_dict()
    shuffled = label_shuffle_ap(
        train_meta, y_train, train.group_domain.fillna("").astype(str).to_numpy(), seed,
    )
    phase4_costs = phase4["models"]
    base_costs = {}
    for name in base_names:
        if name in {"lightgbm", "xgboost", "catboost"}:
            item = phase4_costs[f"{name}engineered"]
            base_costs[name] = {
                "model_size_bytes": item["model_size_bytes"],
                "inference_seconds_per_1000": item["inference_seconds_per_1000"],
            }
        else:
            item = phase5["models"][name]
            transform_seconds_per_1000 = (
                item["metadata"]["validation_transform_seconds"] / len(validation) * 1000
            )
            base_costs[name] = {
                "model_size_bytes": item["artifact_bytes"],
                "sparse_transform_seconds_per_1000": transform_seconds_per_1000,
                "classifier_seconds_per_1000": item["metadata"]["inference_seconds_per_1000"],
                "inference_seconds_per_1000": (
                    transform_seconds_per_1000 + item["metadata"]["inference_seconds_per_1000"]
                ),
            }
    base_inference_seconds = sum(item["inference_seconds_per_1000"] for item in base_costs.values())
    cost_summary = {
        "base_models": base_costs,
        "mean_probability_total_seconds_per_1000": base_inference_seconds,
        "rank_average_total_seconds_per_1000": base_inference_seconds,
        "logistic_meta_seconds_per_1000": logistic_seconds_per_1000,
        "logistic_stack_total_seconds_per_1000": base_inference_seconds + logistic_seconds_per_1000,
        "lightgbm_meta_seconds_per_1000": lightgbm_seconds_per_1000,
        "lightgbm_stack_total_seconds_per_1000": base_inference_seconds + lightgbm_seconds_per_1000,
        "base_model_artifact_bytes_total": sum(item["model_size_bytes"] for item in base_costs.values()),
        "meta_model_artifact_bytes": {
            "logistic": (ARTIFACT_DIR / "logistic_meta.joblib").stat().st_size,
            "lightgbm": (ARTIFACT_DIR / "lightgbm_meta.joblib").stat().st_size,
        },
    }
    split_manifest = PROJECT_ROOT / "data" / "splits" / "MANIFEST.json"
    output = {
        "experiment": "phase6_grouped_oof_stacking",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
        ).strip(),
        "seed": seed,
        "split_manifest_sha256": hashlib.sha256(split_manifest.read_bytes()).hexdigest(),
        "phase4_results_sha256": hashlib.sha256(PHASE4_RESULTS.read_bytes()).hexdigest(),
        "phase5_results_sha256": hashlib.sha256(PHASE5_RESULTS.read_bytes()).hexdigest(),
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "base_models": base_names,
        "grouped_folds": int(config["grouped_folds"]),
        "oof_assignment_min": int(shared_assignments.min()),
        "oof_assignment_max": int(shared_assignments.max()),
        "oof_metadata": oof_metadata,
        "oof_parquet": str(OOF_PATH.relative_to(PROJECT_ROOT)),
        "oof_sha256": hashlib.sha256(OOF_PATH.read_bytes()).hexdigest(),
        "validation_predictions_parquet": str(VALIDATION_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)),
        "validation_predictions_sha256": hashlib.sha256(VALIDATION_PREDICTIONS_PATH.read_bytes()).hexdigest(),
        "base_prediction_correlations": correlation,
        "best_single_model": baseline,
        "best_single_average_precision": baseline_ap,
        "best_ensemble": best_ensemble,
        "ensemble_ap_gain_over_best_single": (
            results[best_ensemble]["metrics"]["average_precision"] - baseline_ap
        ),
        "paired_ap_gain_vs_best_single": paired_gain,
        "costs": cost_summary,
        "label_shuffle_meta_test": shuffled,
        "bootstrap": {
            "resampling_unit": "registered_domain",
            "requested_replicates": bootstrap_replicates,
            "confidence_level": 0.95,
        },
        "models": results,
        "official_test_accessed": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    RESULT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"best single={baseline}; best ensemble={best_ensemble}; AP gain={output['ensemble_ap_gain_over_best_single']:.6f}")
    print(f"Wrote experiment record: {RESULT_PATH}; elapsed={output['elapsed_seconds']:.2f}s")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--label-shuffle-only", action="store_true")
    parser.add_argument("--shuffle-repeats", type=int, default=5)
    args = parser.parse_args()
    if args.label_shuffle_only:
        run_saved_oof_label_shuffle(args.config, args.shuffle_repeats)
        return
    run_stacking(args.config)


if __name__ == "__main__":
    main()