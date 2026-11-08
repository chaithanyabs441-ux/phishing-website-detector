"""Validation-only calibration, threshold selection, and hash benchmark helpers."""

import numpy as np
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, precision_recall_curve, roc_curve
from sklearn.model_selection import GroupKFold


def _ece(y_true, probabilities, bins=10):
    y_true = np.asarray(y_true, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    details = []
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if not mask.any():
            continue
        confidence = float(probabilities[mask].mean())
        accuracy = float(y_true[mask].mean())
        count = int(mask.sum())
        ece += count / len(y_true) * abs(confidence - accuracy)
        details.append({
            "lower": float(lower), "upper": float(upper), "count": count,
            "mean_probability": confidence, "observed_rate": accuracy,
        })
    return float(ece), details


def _fit_calibrator(method, raw_probabilities, targets):
    raw_probabilities = np.asarray(raw_probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int8)
    if method == "platt":
        model = LogisticRegression(solver="lbfgs", C=1e6, random_state=42)
        model.fit(logit(np.clip(raw_probabilities, 1e-8, 1.0 - 1e-8)).reshape(-1, 1), targets)
        return {"method": method, "model": model}
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(raw_probabilities, targets)
        return {"method": method, "model": model}
    raise ValueError(f"Unsupported calibrator: {method}")


def apply_calibrator(calibrator, raw_probabilities):
    raw_probabilities = np.asarray(raw_probabilities, dtype=np.float64)
    if calibrator["method"] == "platt":
        features = logit(np.clip(raw_probabilities, 1e-8, 1.0 - 1e-8)).reshape(-1, 1)
        return calibrator["model"].predict_proba(features)[:, 1]
    if calibrator["method"] == "isotonic":
        return calibrator["model"].predict(raw_probabilities)
    raise ValueError(f"Unsupported calibrator: {calibrator['method']}")


def cross_validated_calibration(raw_probabilities, targets, groups, methods=("platt", "isotonic"),
                                folds=5, ece_bins=10, seed=42):
    raw_probabilities = np.asarray(raw_probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int8)
    groups = np.asarray(groups).astype(str)
    if not (len(raw_probabilities) == len(targets) == len(groups)):
        raise ValueError("probability, label, and group lengths differ")
    outputs = {}
    splitter = GroupKFold(n_splits=folds)
    for method in methods:
        calibrated = np.full(len(targets), np.nan, dtype=np.float64)
        assignments = np.zeros(len(targets), dtype=np.int8)
        for fit_indices, valid_indices in splitter.split(raw_probabilities, targets, groups):
            if set(groups[fit_indices]) & set(groups[valid_indices]):
                raise AssertionError("calibration GroupKFold domains overlap")
            calibrator = _fit_calibrator(method, raw_probabilities[fit_indices], targets[fit_indices])
            calibrated[valid_indices] = apply_calibrator(calibrator, raw_probabilities[valid_indices])
            assignments[valid_indices] += 1
        if not np.all(assignments == 1) or np.isnan(calibrated).any():
            raise AssertionError("every validation row must receive one held-out calibrated score")
        ece, reliability = _ece(targets, calibrated, ece_bins)
        outputs[method] = {
            "oof_probabilities": calibrated,
            "metrics": {
                "brier_score": float(brier_score_loss(targets, calibrated)),
                "log_loss": float(log_loss(targets, np.clip(calibrated, 1e-12, 1 - 1e-12))),
                "ece": ece,
            },
            "reliability_bins": reliability,
            "assignment_min": int(assignments.min()),
            "assignment_max": int(assignments.max()),
        }
    selected = min(
        methods,
        key=lambda method: (
            outputs[method]["metrics"]["log_loss"],
            outputs[method]["metrics"]["brier_score"],
        ),
    )
    final = _fit_calibrator(selected, raw_probabilities, targets)
    return selected, final, outputs


def choose_thresholds(y_true, calibrated_probabilities, max_block_fpr=0.001,
                      min_warn_recall=0.95):
    y_true = np.asarray(y_true, dtype=np.int8)
    probabilities = np.asarray(calibrated_probabilities, dtype=np.float64)
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    block_candidates = np.flatnonzero(fpr <= max_block_fpr)
    finite_candidates = [index for index in block_candidates if np.isfinite(thresholds[index])]
    block_index = min(finite_candidates, key=lambda index: thresholds[index]) if finite_candidates else 0
    block_threshold = float(thresholds[block_index])

    precision, recall, pr_thresholds = precision_recall_curve(y_true, probabilities)
    warn_candidates = np.flatnonzero(recall[:-1] >= min_warn_recall)
    if len(warn_candidates):
        warn_threshold = float(pr_thresholds[warn_candidates[-1]])
    else:
        warn_threshold = float(pr_thresholds[0]) if len(pr_thresholds) else 0.0
    warn_threshold = min(warn_threshold, block_threshold)
    return {
        "block_probability": block_threshold,
        "warn_probability": warn_threshold,
        "block_validation_fpr": float(fpr[block_index]),
        "block_validation_recall": float(tpr[block_index]),
        "warn_validation_recall": float(recall[warn_candidates[-1]]) if len(warn_candidates) else float(recall[-1]),
        "max_block_fpr_target": float(max_block_fpr),
        "min_warn_recall_target": float(min_warn_recall),
    }


def precision_at_recall(y_true, scores, target_recall=0.90):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    valid = np.flatnonzero(recall[:-1] >= target_recall)
    if not len(valid):
        return float("nan")
    return float(np.max(precision[valid]))


def join_hash_predictions(hash_ids, predictions):
    """Return predictions in hash_ids order; reject duplicate and missing SHA keys."""
    if predictions["sha256"].duplicated().any():
        raise ValueError("prediction rows have duplicate sha256 values")
    lookup = predictions.set_index("sha256")
    missing = [sha for sha in hash_ids if sha not in lookup.index]
    if missing:
        raise KeyError(f"{len(missing)} benchmark SHA-256 values did not map to dataset rows")
    return lookup.loc[list(hash_ids)].reset_index()