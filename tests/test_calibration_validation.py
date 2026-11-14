import numpy as np
import pandas as pd
import pytest

from src.evaluation.calibration_validation import (
    choose_thresholds,
    cross_validated_calibration,
    join_hash_predictions,
    precision_at_recall,
)


def test_grouped_calibration_assigns_each_domain_heldout_score_once():
    rng = np.random.default_rng(42)
    targets = np.tile([0, 1], 500)
    groups = np.repeat([f"domain-{index}" for index in range(50)], 20)
    raw = np.clip(0.15 + 0.7 * targets + rng.normal(0, 0.2, len(targets)), 0.001, 0.999)

    selected, calibrator, results = cross_validated_calibration(
        raw, targets, groups, folds=5,
    )

    assert selected in {"platt", "isotonic"}
    assert calibrator["method"] == selected
    for result in results.values():
        assert result["assignment_min"] == result["assignment_max"] == 1
        assert set(result["metrics"]) == {"brier_score", "log_loss", "ece"}


def test_operating_thresholds_meet_validation_targets():
    targets = np.array([0] * 1000 + [1] * 1000)
    probabilities = np.concatenate([
        np.linspace(0.001, 0.3, 1000),
        np.linspace(0.2, 0.999, 1000),
    ])

    thresholds = choose_thresholds(targets, probabilities, 0.01, 0.95)

    assert thresholds["block_validation_fpr"] <= 0.01
    assert thresholds["warn_validation_recall"] >= 0.95
    assert thresholds["warn_probability"] <= thresholds["block_probability"]


def test_hash_join_preserves_lookup_and_rejects_missing_or_duplicate_ids():
    predictions = pd.DataFrame({
        "sha256": ["a", "b"], "label": [0, 1], "probability": [0.1, 0.9],
    })

    joined = join_hash_predictions(["b", "a"], predictions)

    assert joined.sha256.tolist() == ["b", "a"]
    assert joined.probability.tolist() == [0.9, 0.1]
    with pytest.raises(KeyError):
        join_hash_predictions(["missing"], predictions)
    duplicate = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        join_hash_predictions(["a"], duplicate)


def test_benchmark_precision_at_recall_sanity():
    targets = np.array([0, 0, 1, 1])
    perfect = np.array([0.0, 0.1, 0.9, 1.0])
    assert precision_at_recall(targets, perfect, 0.90) == 1.0