import numpy as np
import pandas as pd
import pytest

from src.evaluation.phase5_trees import make_tree_model, tree_feature_columns


def test_tree_feature_selection_excludes_domain_identity_and_non_numeric_columns():
    frame = pd.DataFrame({
        "sha256": ["a", "b"],
        "url_length": [12, 24],
        "html_missing": [0, 1],
        "registered_domain": ["a.example", "b.example"],
        "public_suffix": ["com", "net"],
        "tld_category": ["generic", "generic"],
        "cleaned_text": ["safe", "login"],
        "target": [0, 1],
    })

    columns = tree_feature_columns(frame)

    assert columns == ["url_length", "html_missing"]


@pytest.mark.parametrize("model_name,params", [
    ("extra_trees", {"n_estimators": 20, "max_features": "sqrt", "min_samples_leaf": 2, "max_depth": 8}),
    ("hist_gradient_boosting", {"learning_rate": 0.1, "max_iter": 30, "max_leaf_nodes": 15, "min_samples_leaf": 5, "l2_regularization": 0.1}),
])
def test_tree_references_are_deterministic(model_name, params):
    rng = np.random.default_rng(42)
    features = rng.normal(size=(240, 8)).astype(np.float32)
    targets = (features[:, 0] + features[:, 1] > 0).astype(np.int8)
    models = []
    for _ in range(2):
        model = make_tree_model(model_name, params, seed=13, n_jobs=1)
        model.fit(features, targets)
        models.append(model)

    assert np.array_equal(
        models[0].predict_proba(features), models[1].predict_proba(features),
    )