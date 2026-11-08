import numpy as np
import pandas as pd

from src.evaluation.phase6_stacking import (
    _set_fold_iterations,
    build_grouped_fold_ledger,
    build_meta_features,
    label_shuffle_ap,
    nested_meta_label_shuffle,
    paired_ap_bootstrap,
)
from src.evaluation.phase4_boosting import make_estimator


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
    groups = np.array([f"domain-{index // 100}" for index in range(len(targets))])
    features = np.column_stack([
        latent + rng.normal(0, 0.2, size=len(latent)),
        rng.normal(size=(len(latent), 20)),
    ])

    result = label_shuffle_ap(features, targets, groups, seed=7)

    assert abs(result["difference_from_chance"]) < 0.1
    assert result["unshuffled_average_precision"] > 0.95
    assert result["unshuffled_average_precision"] - result["average_precision"] > 0.25
    assert result["domain_overlap"] == 0


def test_nested_meta_shuffle_uses_disjoint_domains_and_falls_to_chance():
    rng = np.random.default_rng(42)
    targets = np.tile([0, 1], 2000)
    fit_groups = np.array([f"fit-{index // 20}" for index in range(3000)])
    validation_targets = np.tile([0, 1], 1000)
    validation_groups = np.array([f"validation-{index // 20}" for index in range(2000)])
    fit_features = np.column_stack([
        (targets * 2 - 1) + rng.normal(0, 0.25, len(targets)),
        rng.normal(size=(len(targets), 8)),
    ])
    validation_features = np.column_stack([
        (validation_targets * 2 - 1) + rng.normal(0, 0.25, len(validation_targets)),
        rng.normal(size=(len(validation_targets), 8)),
    ])

    result = nested_meta_label_shuffle(
        fit_features, targets, fit_groups, validation_features, validation_targets,
        validation_groups, seed=13, repeats=5,
    )

    assert result["domain_overlap"] == 0
    assert result["unshuffled_average_precision"] > 0.95
    assert result["drop_from_unshuffled"] > 0.25
    assert abs(result["difference_from_chance"]) < 0.08


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


def test_xgboost_oof_fixed_iteration_model_disables_early_stopping():
    model = make_estimator(
        "xgboost",
        {"learning_rate": 0.1, "max_depth": 2, "min_child_weight": 1.0,
         "subsample": 1.0, "colsample_bytree": 1.0, "reg_alpha": 0.0,
         "reg_lambda": 1.0},
        n_jobs=1,
    )

    model = _set_fold_iterations(model, "xgboost", best_iteration=7)

    assert model.get_params()["n_estimators"] == 7
    assert model.get_params()["early_stopping_rounds"] is None