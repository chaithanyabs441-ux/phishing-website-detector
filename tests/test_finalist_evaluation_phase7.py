import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.evaluation.locked_test.evaluate_finalist_phase7 import (
    _prepare_feature_batch,
    confusion_at_threshold,
    evaluate_benchmark_hashes,
)
from src.evaluation.phase4_boosting import prepare_feature_frames
from src.data.engineered_feature_dataset import PHASE4_FEATURE_COLUMNS
from src.features.html_features_v2 import extract_html_features_v2
from src.features.url_features import extract_url_features_v2


def _synthetic_training_rows():
    rows = []
    for url, label in (
        ("https://example.com/", 0),
        ("https://login.example.net/", 1),
    ):
        rows.append({
            **extract_url_features_v2(url),
            **extract_html_features_v2("<title>Account login</title><body>Verify</body>", url),
            "target": label,
        })
    return pd.DataFrame(rows)


def test_final_test_batch_uses_shared_phase4_feature_schema():
    training = _synthetic_training_rows()
    training_features, _ = prepare_feature_frames(training)
    numeric = [column for column in PHASE4_FEATURE_COLUMNS if column not in {"registered_domain", "public_suffix", "tld_category"}]
    medians = training_features[numeric].median()
    batch = pd.DataFrame({
        "sha256": ["test-a"],
        "url": ["https://new.example.org/login"],
        "label": ["phish"],
        "html": ["<title>Secure account</title><form action='https://other.example.net/login'></form>"],
    })

    prepared, domains = _prepare_feature_batch(batch, training_features, medians)

    assert prepared.columns.tolist() == PHASE4_FEATURE_COLUMNS
    assert len(prepared) == 1
    assert prepared.iloc[0].registered_domain == "__UNKNOWN__"
    assert prepared.iloc[0].public_suffix == "__UNKNOWN__"
    assert prepared.iloc[0].num_form_action_domain_mismatches == 1
    assert not prepared[numeric].isna().any().any()
    assert domains == ["example.org"]


def test_confusion_thresholds_match_binary_counts():
    result = confusion_at_threshold([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.2], 0.5)

    assert result == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}


def test_malformed_ipv6_page_url_uses_training_cache_parse_error_features():
    training = _synthetic_training_rows()
    training_features, _ = prepare_feature_frames(training)
    numeric = [
        column for column in PHASE4_FEATURE_COLUMNS
        if column not in {"registered_domain", "public_suffix", "tld_category"}
    ]
    medians = training_features[numeric].median()
    html = '<a href="/login">Sign in</a>'
    batch = pd.DataFrame({
        "sha256": ["malformed-ipv6"],
        "url": ["http://[::1"],
        "label": ["phish"],
        "html": [html],
    })

    prepared, _ = _prepare_feature_batch(batch, training_features, medians)

    assert prepared.iloc[0].html_parse_error == 1
    assert prepared.iloc[0].parsed_html_length == len(html)
    assert prepared.iloc[0].num_forms == 0


def test_hash_benchmark_evaluation_reports_missing_and_ambiguous_hashes(tmp_path):
    benchmark = tmp_path / "benchmark"
    (benchmark / "benchmark-005").mkdir(parents=True)
    (benchmark / "benign.txt").write_text("b\n", encoding="utf-8")
    (benchmark / "benchmark-005" / "0.txt").write_text(
        "p\nmissing\nduplicate\n", encoding="utf-8",
    )
    predictions = pd.DataFrame({
        "sha256": ["b", "p", "duplicate", "duplicate"],
        "label": [0, 1, 1, 1],
        "registered_domain": ["benign.example", "phish.example", "dup.example", "dup.example"],
        "probability": [0.1, 0.9, 0.8, 0.8],
    })

    results, summary, audit = evaluate_benchmark_hashes(predictions, benchmark)

    assert len(results) == 1
    assert results.iloc[0].n_benign == 1
    assert results.iloc[0].n_phish == 1
    assert results.iloc[0].n_phish_unmatched == 2
    assert audit["ambiguous_test_hashes"] == 1
    assert summary.iloc[0].n_benchmarks == 1


def test_benchmark_average_precision_sanity_strong_and_random_scores():
    rng = np.random.default_rng(42)
    targets = np.tile([0] * 19 + [1], 500)
    random_scores = rng.random(len(targets))

    assert average_precision_score(targets, targets) == 1.0
    assert abs(average_precision_score(targets, random_scores) - targets.mean()) < 0.02
