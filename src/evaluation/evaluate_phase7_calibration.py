"""Calibrate the pre-registered CatBoost finalist and select validation thresholds."""

import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from joblib import dump
from scipy.special import logit

from src.evaluation.evaluate_svm_blocked import _domain_bootstrap, _metric_bundle
from src.evaluation.phase4_boosting import prepare_feature_frames
from src.evaluation.phase4_data import PROJECT_ROOT, load_phase4_data
from src.evaluation.phase7_calibration import (
    _ece,
    apply_calibrator,
    choose_thresholds,
    cross_validated_calibration,
)


CONFIG_PATH = PROJECT_ROOT / "configs" / "phase7_calibration.yaml"
RESULT_PATH = PROJECT_ROOT / "experiments" / "phase7_calibration.json"
CALIBRATOR_PATH = PROJECT_ROOT / "models" / "phase7" / "catboost_engineered_calibrator.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "configs" / "thresholds.yaml"
RELIABILITY_PATH = PROJECT_ROOT / "reports" / "figures" / "phase7_reliability.png"


def _metrics_at_threshold(y_true, probabilities, threshold):
    scores = logit(np.clip(probabilities, 1e-12, 1 - 1e-12)) - logit(
        np.clip(threshold, 1e-12, 1 - 1e-12)
    )
    return scores


def run_calibration(config_path=None):
    started = time.monotonic()
    config_path = Path(config_path or CONFIG_PATH)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    data = load_phase4_data()
    train = data[data["split"] == "train"].reset_index(drop=True)
    validation = data[data["split"] == "validation"].reset_index(drop=True)
    y_validation = validation.target.to_numpy(dtype=np.int8)
    groups = validation.group_domain.fillna("").astype(str).to_numpy()

    model_path = PROJECT_ROOT / config["model_artifact"]
    if not model_path.exists():
        raise FileNotFoundError(f"Pre-registered finalist artifact missing: {model_path}")
    model = __import__("joblib").load(model_path)
    _, (validation_features,) = prepare_feature_frames(train, validation)
    raw_probabilities = model.predict_proba(validation_features)[:, 1]

    methods = tuple(config["calibration"]["methods"])
    selected, final_calibrator, calibration_results = cross_validated_calibration(
        raw_probabilities,
        y_validation,
        groups,
        methods=methods,
        folds=int(config["calibration"]["grouped_folds"]),
        ece_bins=int(config["calibration"]["ece_bins"]),
        seed=seed,
    )
    selected_oof_probabilities = calibration_results[selected].pop("oof_probabilities")
    thresholds = choose_thresholds(
        y_validation,
        selected_oof_probabilities,
        max_block_fpr=float(config["thresholds"]["block_max_fpr"]),
        min_warn_recall=float(config["thresholds"]["warn_min_recall"]),
    )

    unique_domains, domain_codes = np.unique(groups, return_inverse=True)
    threshold_metrics = {}
    bootstrap_replicates = int(config["bootstrap"]["replicates"])
    for index, name in enumerate(("probability_0.5", "warn", "block")):
        threshold = 0.5 if name == "probability_0.5" else thresholds[f"{name}_probability"]
        score = _metrics_at_threshold(y_validation, selected_oof_probabilities, threshold)
        point = _metric_bundle(y_validation, score)
        intervals, completed = _domain_bootstrap(
            y_validation, score, domain_codes, len(unique_domains),
            bootstrap_replicates, seed + index,
        )
        threshold_metrics[name] = {
            "probability_threshold": float(threshold),
            "metrics": point,
            "domain_bootstrap_95ci": intervals,
            "bootstrap_replicates_completed": completed,
        }

    final_calibrated_validation = apply_calibrator(final_calibrator, raw_probabilities)
    final_ece, final_reliability = _ece(
        y_validation, final_calibrated_validation, int(config["calibration"]["ece_bins"]),
    )
    CALIBRATOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(final_calibrator, CALIBRATOR_PATH)

    RELIABILITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6, 6), dpi=140)
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    selected_reliability = calibration_results[selected]["reliability_bins"]
    axis.plot(
        [item["mean_probability"] for item in selected_reliability],
        [item["observed_rate"] for item in selected_reliability],
        marker="o", label=f"{selected} OOF",
    )
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed phishing rate")
    axis.set_title("CatBoost engineered: grouped OOF calibration")
    axis.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(RELIABILITY_PATH)
    plt.close(fig)

    selected_output = {
        "method": selected,
        "oof_metrics": calibration_results[selected]["metrics"],
        "oof_reliability_bins": selected_reliability,
        "final_refit_brier": float(np.mean((final_calibrated_validation - y_validation) ** 2)),
        "final_refit_log_loss": float(-np.mean(
            y_validation * np.log(np.clip(final_calibrated_validation, 1e-12, 1.0))
            + (1 - y_validation) * np.log(np.clip(1 - final_calibrated_validation, 1e-12, 1.0))
        )),
        "final_refit_ece": final_ece,
        "final_refit_reliability_bins": final_reliability,
        "artifact": str(CALIBRATOR_PATH.relative_to(PROJECT_ROOT)),
    }
    threshold_config = {
        "version": 1,
        "selected_model": config["selected_model"],
        "feature_version": "url_features_v2+html_features_v2",
        "calibration_method": selected,
        "calibrator_artifact": selected_output["artifact"],
        "assumed_prevalence": float(config["thresholds"]["assumed_prevalence"]),
        "thresholds": {
            "block": float(thresholds["block_probability"]),
            "warn": float(thresholds["warn_probability"]),
        },
        "selection_constraints": {
            "block_max_fpr": float(config["thresholds"]["block_max_fpr"]),
            "warn_min_recall": float(config["thresholds"]["warn_min_recall"]),
        },
        "source": "temporal validation only; grouped OOF-calibrated scores",
    }
    THRESHOLD_PATH.write_text(yaml.safe_dump(threshold_config, sort_keys=False), encoding="utf-8")

    split_manifest = PROJECT_ROOT / "data" / "splits" / "MANIFEST.json"
    output = {
        "experiment": "phase7_validation_calibration_thresholds",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
        ).strip(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "split_manifest_sha256": hashlib.sha256(split_manifest.read_bytes()).hexdigest(),
        "finalist": config["selected_model"],
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "validation_domains": int(len(unique_domains)),
        "calibration_methods": {
            method: {key: value for key, value in item.items() if key != "oof_probabilities"}
            for method, item in calibration_results.items()
        },
        "selected_calibrator": selected_output,
        "threshold_selection": thresholds,
        "threshold_metrics": threshold_metrics,
        "reliability_plot": str(RELIABILITY_PATH.relative_to(PROJECT_ROOT)),
        "threshold_config": str(THRESHOLD_PATH.relative_to(PROJECT_ROOT)),
        "threshold_config_sha256": hashlib.sha256(THRESHOLD_PATH.read_bytes()).hexdigest(),
        "assumed_prevalence": float(config["thresholds"]["assumed_prevalence"]),
        "bootstrap": {
            "resampling_unit": "registered_domain",
            "requested_replicates": bootstrap_replicates,
            "confidence_level": 0.95,
        },
        "official_test_accessed": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    RESULT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"selected_calibrator={selected}; thresholds={thresholds}")
    print(f"Wrote calibration record: {RESULT_PATH}")
    return output


if __name__ == "__main__":
    run_calibration()
