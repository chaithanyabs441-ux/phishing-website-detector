"""Train URL-only and URL+HTML CatBoost serving models on temporal data."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier, __version__ as catboost_version
from joblib import dump
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
import sklearn

from src.data.engineered_feature_dataset import (
    CATEGORICAL_FEATURES,
    PHASE4_FEATURE_COLUMNS,
    PROJECT_ROOT,
    load_phase4_data,
)
from src.evaluation.calibration_validation import (
    apply_calibrator,
    cross_validated_calibration,
)
from src.features.url_features import URL_FEATURE_COLUMNS_V2


PHASE4_RESULTS_PATH = PROJECT_ROOT / "experiments" / "phase4_gbm_results.json"
PHASE7_CONFIG_PATH = PROJECT_ROOT / "configs" / "phase7_calibration.yaml"
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "serving.yaml"
THRESHOLD_CONFIG_PATH = PROJECT_ROOT / "configs" / "thresholds.yaml"
MANIFEST_PATH = PROJECT_ROOT / "models" / "MANIFEST.json"
ARTIFACT_DIR = PROJECT_ROOT / "models" / "phase10"
REPORT_PATH = PROJECT_ROOT / "reports" / "phase_10c_serving_models.md"
RESULT_PATH = PROJECT_ROOT / "experiments" / "phase10c_serving_models.json"

MODEL_FEATURES = {
    "url_only": list(URL_FEATURE_COLUMNS_V2),
    "url_html": list(PHASE4_FEATURE_COLUMNS),
}
ARTIFACT_FILES = {
    "url_only": {
        "model": "catboost_url_only.joblib",
        "calibrator": "catboost_url_only_calibrator.joblib",
        "feature_version": "url_features_v2",
    },
    "url_html": {
        "model": "catboost_url_html.joblib",
        "calibrator": "catboost_url_html_calibrator.joblib",
        "feature_version": "url_features_v2+html_features_v2",
    },
}


def _prepare_frames(train_frame, validation_frame, feature_columns):
    """Prepare model columns with train-only numeric medians and categories."""
    categorical = [column for column in CATEGORICAL_FEATURES if column in feature_columns]
    x_train = train_frame[feature_columns].copy()
    x_validation = validation_frame[feature_columns].copy()
    for column in categorical:
        train_values = x_train[column].fillna("__MISSING__").astype(str)
        categories = sorted(set(train_values) | {"__UNKNOWN__"})
        validation_values = x_validation[column].fillna("__MISSING__").astype(str)
        validation_values = validation_values.where(
            validation_values.isin(categories), "__UNKNOWN__",
        )
        x_train[column] = pd.Categorical(train_values, categories=categories)
        x_validation[column] = pd.Categorical(validation_values, categories=categories)
    numeric = [column for column in feature_columns if column not in categorical]
    medians = x_train[numeric].median()
    x_train[numeric] = x_train[numeric].fillna(medians).fillna(0)
    x_validation[numeric] = x_validation[numeric].fillna(medians).fillna(0)
    return x_train, x_validation, categorical


def _select_fpr_threshold(targets, probabilities, max_fpr):
    """Choose the lowest finite threshold whose observed validation FPR meets cap."""
    fpr, tpr, thresholds = roc_curve(
        np.asarray(targets, dtype=np.int8),
        np.asarray(probabilities, dtype=np.float64),
        drop_intermediate=False,
    )
    candidates = [
        index for index, threshold in enumerate(thresholds)
        if np.isfinite(threshold) and fpr[index] <= max_fpr
    ]
    if not candidates:
        raise ValueError(f"No finite validation threshold meets FPR <= {max_fpr}")
    index = min(candidates, key=lambda candidate: thresholds[candidate])
    return {
        "probability": float(thresholds[index]),
        "validation_fpr": float(fpr[index]),
        "validation_recall": float(tpr[index]),
        "max_fpr": float(max_fpr),
    }


def _fit_one(name, train, validation, params, phase7_config):
    columns = MODEL_FEATURES[name]
    x_train, x_validation, categorical = _prepare_frames(train, validation, columns)
    y_train = train["target"].to_numpy(dtype=np.int8)
    y_validation = validation["target"].to_numpy(dtype=np.int8)
    groups = validation["group_domain"].fillna("").astype(str).to_numpy()
    model = CatBoostClassifier(
        **params,
        iterations=3000,
        loss_function="Logloss",
        eval_metric="PRAUC",
        auto_class_weights="Balanced",
        random_seed=int(phase7_config["seed"]),
        thread_count=4,
        allow_writing_files=False,
        verbose=False,
    )
    started = time.monotonic()
    model.fit(
        x_train,
        y_train,
        eval_set=(x_validation, y_validation),
        cat_features=categorical,
        early_stopping_rounds=100,
        use_best_model=True,
        verbose=False,
    )
    training_seconds = time.monotonic() - started
    raw_probabilities = model.predict_proba(x_validation)[:, 1]
    selected_method, calibrator, calibration = cross_validated_calibration(
        raw_probabilities,
        y_validation,
        groups,
        methods=tuple(phase7_config["calibration"]["methods"]),
        folds=int(phase7_config["calibration"]["grouped_folds"]),
        ece_bins=int(phase7_config["calibration"]["ece_bins"]),
        seed=int(phase7_config["seed"]),
    )
    calibrated_oof = calibration[selected_method]["oof_probabilities"]
    block = _select_fpr_threshold(y_validation, calibrated_oof, 0.001)
    warn = _select_fpr_threshold(y_validation, calibrated_oof, 0.02)
    calibrated_final = apply_calibrator(calibrator, raw_probabilities)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    paths = ARTIFACT_FILES[name]
    model_path = ARTIFACT_DIR / paths["model"]
    calibrator_path = ARTIFACT_DIR / paths["calibrator"]
    dump(model, model_path)
    dump(calibrator, calibrator_path)

    return {
        "feature_columns": columns,
        "categorical_features": categorical,
        "best_iteration": int(model.best_iteration_),
        "training_seconds": float(training_seconds),
        "raw_validation": {
            "average_precision": float(average_precision_score(y_validation, raw_probabilities)),
            "roc_auc": float(roc_auc_score(y_validation, raw_probabilities)),
        },
        "calibrated_oof_validation": {
            "average_precision": float(average_precision_score(y_validation, calibrated_oof)),
            "roc_auc": float(roc_auc_score(y_validation, calibrated_oof)),
            "brier_score": float(calibration[selected_method]["metrics"]["brier_score"]),
            "log_loss": float(calibration[selected_method]["metrics"]["log_loss"]),
            "ece": float(calibration[selected_method]["metrics"]["ece"]),
        },
        "calibrated_final_validation": {
            "average_precision": float(average_precision_score(y_validation, calibrated_final)),
            "roc_auc": float(roc_auc_score(y_validation, calibrated_final)),
        },
        "calibration_method": selected_method,
        "calibration_assignment_min": int(calibration[selected_method]["assignment_min"]),
        "calibration_assignment_max": int(calibration[selected_method]["assignment_max"]),
        "thresholds": {"block": block, "warn": warn},
        "model_path": model_path,
        "calibrator_path": calibrator_path,
        "feature_version": paths["feature_version"],
    }


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _register_artifacts(results):
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = {item["file"]: item for item in manifest["artifacts"]}
    for name, result in results.items():
        paths = ARTIFACT_FILES[name]
        for kind in ("model", "calibrator"):
            path = result[f"{kind}_path"]
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            entries[relative] = {
                "file": relative,
                "sha256": _hash_file(path),
                "trained_with": "src/evaluation/phase10c_train_serving_models.py",
                "trained_on_split": "frozen temporal train split; early stopping and calibration/threshold selection on temporal validation only",
                "feature_version": paths["feature_version"],
                "artifact_role": "catboost_classifier" if kind == "model" else "grouped_oof_probability_calibrator",
                "framework_version": catboost_version if kind == "model" else sklearn.__version__,
            }
    manifest["artifacts"] = sorted(entries.values(), key=lambda item: item["file"])
    manifest.setdefault("notes", []).append(
        "Phase 10 URL-only and URL+HTML CatBoost models and grouped-OOF calibrators are registered by SHA-256."
    )
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _write_threshold_config(results, phase7_config):
    config = {
        "version": 2,
        "feature_version": "url_features_v2 or url_features_v2+html_features_v2; selected by supplied HTML",
        "source": "validation temporal split only; grouped OOF-calibrated scores",
        "selection_method": "lowest probability threshold meeting each validation FPR cap",
        "assumed_prevalence": float(phase7_config["thresholds"]["assumed_prevalence"]),
        "models": {},
    }
    for name, result in results.items():
        relative_calibrator = result["calibrator_path"].relative_to(PROJECT_ROOT).as_posix()
        config["models"][name] = {
            "feature_version": result["feature_version"],
            "calibration_method": result["calibration_method"],
            "calibrator_artifact": relative_calibrator,
            "thresholds": {
                level: float(result["thresholds"][level]["probability"])
                for level in ("block", "warn")
            },
            "selection_constraints": {
                "block_max_fpr": 0.001,
                "warn_max_fpr": 0.02,
            },
            "validation_metrics": {
                "block_fpr": float(result["thresholds"]["block"]["validation_fpr"]),
                "block_recall": float(result["thresholds"]["block"]["validation_recall"]),
                "warn_fpr": float(result["thresholds"]["warn"]["validation_fpr"]),
                "warn_recall": float(result["thresholds"]["warn"]["validation_recall"]),
            },
        }
    THRESHOLD_CONFIG_PATH.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _json_safe(results):
    safe = {}
    for name, result in results.items():
        safe[name] = {
            key: value for key, value in result.items()
            if key not in {"model_path", "calibrator_path"}
        }
        for kind in ("model", "calibrator"):
            safe[name][f"{kind}_artifact"] = result[f"{kind}_path"].relative_to(PROJECT_ROOT).as_posix()
            safe[name][f"{kind}_sha256"] = _hash_file(result[f"{kind}_path"])
        safe[name]["feature_columns"] = list(result["feature_columns"])
    return safe


def _write_report(output):
    lines = [
        "# Phase 10C — URL-only and URL+HTML serving models",
        "",
        "Models were trained on the frozen temporal training split. Early stopping, grouped-OOF calibrator selection/refit, and both operating thresholds used the temporal validation split only. No official test data, predictions, labels, or evaluation artifacts were read.",
        "",
        f"Training rows: {output['training_rows']:,}; validation rows: {output['validation_rows']:,}; validation registered domains: {output['validation_domains']:,}.",
        "",
        "| Model route | Features | Best iteration | Calibration | Validation AP (raw) | Validation ROC-AUC (raw) | BLOCK threshold / FPR / recall | WARN threshold / FPR / recall |",
        "|---|---:|---:|---|---:|---:|---|---|",
    ]
    for name, result in output["models"].items():
        block = result["thresholds"]["block"]
        warn = result["thresholds"]["warn"]
        lines.append(
            f"| {name} | {len(result['feature_columns'])} | {result['best_iteration']} | {result['calibration_method']} | "
            f"{result['raw_validation']['average_precision']:.6f} | {result['raw_validation']['roc_auc']:.6f} | "
            f"{block['probability']:.8f} / {block['validation_fpr']:.6f} / {block['validation_recall']:.6f} | "
            f"{warn['probability']:.8f} / {warn['validation_fpr']:.6f} / {warn['validation_recall']:.6f} |"
        )
        lines.append("")
        lines.append(f"### {name} calibration and artifacts")
        lines.append("")
        lines.append(
            f"Grouped-OOF validation AP={result['calibrated_oof_validation']['average_precision']:.6f}, "
            f"ROC-AUC={result['calibrated_oof_validation']['roc_auc']:.6f}, "
            f"Brier={result['calibrated_oof_validation']['brier_score']:.6f}, "
            f"log-loss={result['calibrated_oof_validation']['log_loss']:.6f}, "
            f"ECE={result['calibrated_oof_validation']['ece']:.6f}. "
            f"Each validation row received one held-out calibration score "
            f"(assignment min/max {result['calibration_assignment_min']}/{result['calibration_assignment_max']})."
        )
        lines.append("")
        lines.append(
            f"Model artifact: `{result['model_artifact']}` (SHA-256 `{result['model_sha256']}`); "
            f"calibrator: `{result['calibrator_artifact']}` (SHA-256 `{result['calibrator_sha256']}`)."
        )
        lines.append("")
    lines.extend([
        "The BLOCK threshold is the lowest calibrated probability whose validation FPR is no greater than 0.1%; WARN is the lowest probability whose validation FPR is no greater than 2%. Both were selected from grouped-OOF calibrated validation scores. The runtime chooses `url_only` when HTML is empty and `url_html` when supplied HTML is nonempty.",
        "",
        "Threshold provenance is recorded as validation-only in `configs/thresholds.yaml`; it contains no manual override. The artifact SHA-256 hashes are registered in `models/MANIFEST.json`.",
        "",
        "This step did not access the official test set and did not alter any recorded Phase 7 benchmark result.",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_training():
    """Fit both serving variants; calibrate and select thresholds on validation."""
    started = time.monotonic()
    phase4 = json.loads(PHASE4_RESULTS_PATH.read_text(encoding="utf-8"))
    phase7_config = yaml.safe_load(PHASE7_CONFIG_PATH.read_text(encoding="utf-8"))
    params = phase4["tuning"]["catboost"]["best_params"]
    data = load_phase4_data()
    train = data.loc[data["split"] == "train"].reset_index(drop=True)
    validation = data.loc[data["split"] == "validation"].reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError("Frozen temporal train and validation partitions must both be nonempty")
    group_count = int(validation["group_domain"].nunique())

    results = {}
    for name in ("url_only", "url_html"):
        results[name] = _fit_one(name, train, validation, params, phase7_config)
        result = results[name]
        print(
            f"{name}: AP={result['raw_validation']['average_precision']:.6f}; "
            f"ROC-AUC={result['raw_validation']['roc_auc']:.6f}; "
            f"calibration={result['calibration_method']}; "
            f"BLOCK_FPR={result['thresholds']['block']['validation_fpr']:.6f}; "
            f"WARN_FPR={result['thresholds']['warn']['validation_fpr']:.6f}; "
            f"best_iteration={result['best_iteration']}",
            flush=True,
        )

    _register_artifacts(results)
    _write_threshold_config(results, phase7_config)
    # The registry reads paths from serving.yaml; keep the route paths explicit and portable.
    model_config = yaml.safe_load(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    for name, kind in (("url_only", "model"), ("url_html", "model"),
                       ("url_only", "calibrator"), ("url_html", "calibrator")):
        field = f"{name}_{kind}"
        model_config["serving"][field] = results[name][f"{kind}_path"].relative_to(PROJECT_ROOT).as_posix()
    MODEL_CONFIG_PATH.write_text(yaml.safe_dump(model_config, sort_keys=False), encoding="utf-8")

    output = {
        "experiment": "phase10c_url_only_url_html_serving_catboost",
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "validation_domains": group_count,
        "params_source": "experiments/phase4_gbm_results.json tuning.catboost.best_params",
        "params": params,
        "early_stopping_rounds": 100,
        "calibration_method_source": "src/evaluation/phase7_calibration.cross_validated_calibration",
        "calibration_folds": int(phase7_config["calibration"]["grouped_folds"]),
        "threshold_source": "temporal validation only; grouped OOF calibrated probabilities",
        "threshold_constraints": {"block_max_fpr": 0.001, "warn_max_fpr": 0.02},
        "models": _json_safe(results),
        "official_test_accessed": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    _write_report(output)
    print(f"Wrote report: {REPORT_PATH}")
    print(f"Wrote experiment: {RESULT_PATH}")
    return output


def main():
    run_training()


if __name__ == "__main__":
    main()
