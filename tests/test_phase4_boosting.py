import numpy as np
import pandas as pd
import pytest
from joblib import dump, load

from src.evaluation.phase4_boosting import (
    fit_with_early_stopping,
    generate_oof_text_scores,
    grouped_tuning_split,
    make_estimator,
    prepare_feature_frames,
)
from src.data.engineered_feature_dataset import CATEGORICAL_FEATURES, PHASE4_FEATURE_COLUMNS


def test_grouped_tuning_split_is_deterministic_bounded_and_domain_disjoint():
    frame = pd.DataFrame({
        "target": np.tile([0, 1], 100),
        "group_domain": [f"domain-{index // 2}" for index in range(200)],
    })

    fit_a, valid_a = grouped_tuning_split(
        frame, seed=17, fit_rows=60, validation_rows=20,
    )
    fit_b, valid_b = grouped_tuning_split(
        frame, seed=17, fit_rows=60, validation_rows=20,
    )

    assert np.array_equal(fit_a, fit_b)
    assert np.array_equal(valid_a, valid_b)
    assert len(fit_a) == 60
    assert len(valid_a) == 20
    assert set(frame.iloc[fit_a].group_domain).isdisjoint(frame.iloc[valid_a].group_domain)
    assert set(frame.iloc[fit_a].target) == {0, 1}
    assert set(frame.iloc[valid_a].target) == {0, 1}


def test_text_scores_are_grouped_oof_aligned_and_deterministic():
    targets = np.tile([0, 1], 50)
    groups = [f"domain-{index // 10}" for index in range(100)]
    train = pd.DataFrame({
        "target": targets,
        "group_domain": groups,
        "url": [
            f"https://site-{index // 10}.example/{'safe' if target == 0 else 'login'}/{index}"
            for index, target in enumerate(targets)
        ],
    })
    validation = pd.DataFrame({
        "url": ["https://fresh.example/safe", "https://fresh.example/login"]
    })
    options = {"ngram_range": (2, 3), "min_df": 1, "max_features": 5000}

    train_scores_a, valid_scores_a, metadata = generate_oof_text_scores(
        train, validation, folds=5, vectorizer_kwargs=options,
    )
    train_scores_b, valid_scores_b, _ = generate_oof_text_scores(
        train, validation, folds=5, vectorizer_kwargs=options,
    )

    assert len(train_scores_a) == len(train)
    assert len(valid_scores_a) == len(validation)
    assert metadata["oof_assignment_min"] == metadata["oof_assignment_max"] == 1
    assert np.array_equal(train_scores_a, train_scores_b)
    assert np.array_equal(valid_scores_a, valid_scores_b)


@pytest.mark.parametrize("model_name,params", [
    ("lightgbm", {
        "learning_rate": 0.1, "num_leaves": 15, "max_depth": 5,
        "min_child_samples": 10, "subsample": 0.9, "colsample_bytree": 0.9,
        "reg_alpha": 0.01, "reg_lambda": 1.0,
    }),
    ("xgboost", {
        "learning_rate": 0.1, "max_depth": 4, "min_child_weight": 1.0,
        "subsample": 0.9, "colsample_bytree": 0.9,
        "reg_alpha": 0.01, "reg_lambda": 1.0,
    }),
    ("catboost", {
        "learning_rate": 0.1, "depth": 5, "l2_leaf_reg": 3.0,
        "random_strength": 0.1, "bagging_temperature": 0.0,
    }),
])
def test_booster_determinism_roundtrip_and_single_batch_parity(model_name, params, tmp_path):
    row_indices = np.arange(400)
    train_target = (row_indices % 2).astype(np.int8)
    valid_target = (np.arange(100) % 2).astype(np.int8)

    def make_frame(indices, targets):
        frame = pd.DataFrame({
            column: (indices % 17) + targets * 5
            for column in PHASE4_FEATURE_COLUMNS
            if column not in CATEGORICAL_FEATURES
        })
        frame["registered_domain"] = [f"domain-{index % 7}.example" for index in indices]
        frame["public_suffix"] = ["com" if index % 2 else "net" for index in indices]
        frame["tld_category"] = "generic"
        frame["target"] = targets
        return frame

    train_frame = make_frame(row_indices, train_target)
    valid_frame = make_frame(np.arange(400, 500), valid_target)
    x_train, (x_valid,) = prepare_feature_frames(train_frame, valid_frame)
    models = []
    for _ in range(2):
        model = make_estimator(model_name, params, seed=19, n_jobs=1)
        fit_with_early_stopping(
            model_name, model, x_train, train_target, x_valid, valid_target,
        )
        models.append(model)

    prediction = models[0].predict_proba(x_valid)[:, 1]
    repeated = models[1].predict_proba(x_valid)[:, 1]
    assert np.allclose(prediction, repeated, rtol=0, atol=1e-12)
    assert np.allclose(
        models[0].predict_proba(x_valid.iloc[:1]),
        models[0].predict_proba(x_valid)[:1],
        rtol=0, atol=1e-12,
    )
    artifact = tmp_path / f"{model_name}.joblib"
    dump(models[0], artifact)
    restored = load(artifact)
    assert np.allclose(prediction, restored.predict_proba(x_valid)[:, 1], rtol=0, atol=1e-12)