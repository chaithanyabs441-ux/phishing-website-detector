import pandas as pd

from src.evaluation.domain_overlap_audit import audit_seen_unseen


def test_seen_unseen_audit_uses_prior_predictions_and_training_domains():
    identities = pd.DataFrame(
        {
            "sha256": ["a", "b", "c", "d"],
            "url": [
                "https://shop.example.com/login",
                "https://shop.example.com/home",
                "https://account.unseen.test/login",
                "https://account.unseen.test/home",
            ],
        },
    )
    predictions = pd.DataFrame(
        {
            "sha256": ["a", "b", "c", "d"],
            "label": [1, 0, 1, 0],
            "probability": [0.95, 0.1, 0.8, 0.2],
        },
    )

    result = audit_seen_unseen(
        identities,
        predictions,
        {"example.com"},
        block_threshold=0.9,
    )

    assert result["test_rows"] == 4
    assert result["training_domains"] == 1
    assert result["seen"]["rows"] == 2
    assert result["seen"]["domains"] == 1
    assert result["seen"]["average_precision"] == 1.0
    assert result["seen"]["recall_at_block"] == 1.0
    assert result["seen"]["fpr_at_block"] == 0.0
    assert result["unseen"]["rows"] == 2
    assert result["unseen"]["domains"] == 1
    assert result["unseen"]["average_precision"] == 1.0
    assert result["unseen"]["recall_at_block"] == 0.0
    assert result["unseen"]["fpr_at_block"] == 0.0
