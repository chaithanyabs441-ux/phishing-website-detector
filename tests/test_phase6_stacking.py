import numpy as np
import pandas as pd

from src.evaluation.phase6_stacking import (
    build_grouped_fold_ledger,
    build_meta_features,
    label_shuffle_ap,
    paired_ap_bootstrap,
)


def test_grouped_fold_ledger_is_disjoint_and_assigns_each_row_once():
    frame = pd.DataFrame({
        "target": np.tile([0, 1], 100),
        "group_domain": [f"domain-{index // 4}" for index in range(200)],
    })

    ledger, assignments = build_grouped_fold_ledger(frame, n_splits=5)

    assert len(ledger) == 5
    assert np.all(assignments == 1)
    groups = frame.group_domain.to_numpy()
    for fit_indices, score_indices in ledger:
        assert set(groups[fit_indices]).isdisjoint(groups[score_indices])


def test_meta_features_are_finite_logits():
    probabilities = np.array([[0.0, 1.0], [0.25, 0.75]])

    features = build_meta_features(probabilities)

    assert features.shape == probabilities.shape
    assert np.isfinite(features).all()
    assert features[0, 0] < -10
    assert features[0, 1] > 10


def test_shuffled_meta_labels_fall_to_chance_on_synthetic_data():
    rng = np.random.default_rng(42)
    latent = rng.normal(size=8000)
    targets = (latent > 0).astype(np.int8)
    features = np.column_stack([
        latent + rng.normal(0, 0.2, size=len(latent)),
        rng.normal(size=(len(latent), 20)),
    ])

    result = label_shuffle_ap(features, targets, seed=7)

    assert abs(result["difference_from_chance"]) < 0.1
    assert result["unshuffled_average_precision"] > 0.95
    assert result["unshuffled_average_precision"] - result["average_precision"] > 0.25


def test_paired_domain_bootstrap_reports_zero_for_identical_predictions():
    targets = np.tile([0, 1], 100)
    scores = np.linspace(-2, 2, len(targets))
    domain_codes = np.repeat(np.arange(20), 10)

    result = paired_ap_bootstrap(
        targets, scores, scores, domain_codes, 20, replicates=30, seed=9,
    )

    assert result["point_difference"] == 0.0
    assert result["ci_95"] == [0.0, 0.0]
    assert result["replicates_completed"] == 30