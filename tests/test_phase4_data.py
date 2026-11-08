import pandas as pd
import pytest

from src.evaluation.phase4_data import NUMERIC_HTML_FEATURES, assemble_phase4_data


def _fixtures():
    raw = pd.DataFrame({
        "sha256": ["a", "b"],
        "url": ["https://example.com/", "https://login.example.net/"],
        "label": ["benign", "phish"],
    })
    html = pd.DataFrame({
        "sha256": ["a", "b"],
        **{column: [0, 1] for column in NUMERIC_HTML_FEATURES},
    })
    temporal = pd.DataFrame({"sha256": ["a", "b"], "split": ["train", "validation"]})
    grouped = pd.DataFrame({
        "sha256": ["a", "b"], "registered_domain": ["example.com", "example.net"],
        "fold": [0, 1],
    })
    return raw, html, temporal, grouped


def test_assemble_phase4_data_joins_by_sha_and_preserves_frozen_splits():
    result = assemble_phase4_data(*_fixtures())

    assert result["sha256"].tolist() == ["a", "b"]
    assert result["target"].tolist() == [0, 1]
    assert result["split"].tolist() == ["train", "validation"]
    assert result["fold"].tolist() == [0, 1]
    assert result["group_domain"].tolist() == ["example.com", "example.net"]
    assert "cleaned_text" not in result.columns


def test_assemble_phase4_data_rejects_missing_or_duplicate_cache_ids():
    raw, html, temporal, grouped = _fixtures()
    with pytest.raises(ValueError, match="does not cover"):
        assemble_phase4_data(raw, html.iloc[:1], temporal, grouped)

    duplicate_html = pd.concat([html, html.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="repeated SHA-256"):
        assemble_phase4_data(raw, duplicate_html, temporal, grouped)