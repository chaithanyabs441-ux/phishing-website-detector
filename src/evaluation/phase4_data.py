"""Load training-only URL v2 and HTML v2 features for Phase 4 models."""

import json
from pathlib import Path

import pandas as pd

from src.features.html_features_v2 import HTML_FEATURE_COLUMNS_V2
from src.features.url_features import URL_FEATURE_COLUMNS_V2, extract_url_features_v2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "data"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
HTML_CACHE_ROOT = PROJECT_ROOT / "data" / "processed" / "features_v2"
HTML_FEATURE_HASH = "f5434603820e65699462bc276cc34b2c6ba4ad7ed72f9f061a1cc0d13d79e6d3"

NUMERIC_HTML_FEATURES = [column for column in HTML_FEATURE_COLUMNS_V2 if column != "cleaned_text"]
PHASE4_FEATURE_COLUMNS = [*URL_FEATURE_COLUMNS_V2, *NUMERIC_HTML_FEATURES]
CATEGORICAL_FEATURES = ["registered_domain", "public_suffix", "tld_category"]


def assemble_phase4_data(raw_rows, html_features, temporal_assignments, grouped_assignments):
    """One-to-one SHA joins for training rows and frozen Phase 2 assignments."""
    required = {
        "raw": (raw_rows, {"sha256", "url", "label"}),
        "html": (html_features, {"sha256", *NUMERIC_HTML_FEATURES}),
        "temporal": (temporal_assignments, {"sha256", "split"}),
        "grouped": (grouped_assignments, {"sha256", "registered_domain", "fold"}),
    }
    for name, (frame, columns) in required.items():
        missing_columns = columns - set(frame.columns)
        if missing_columns:
            raise ValueError(f"{name} data missing columns: {sorted(missing_columns)}")
        if frame["sha256"].duplicated().any():
            raise ValueError(f"{name} data has repeated SHA-256 values")

    merged = raw_rows[["sha256", "url", "label"]].merge(
        html_features[["sha256", *NUMERIC_HTML_FEATURES]],
        on="sha256", how="left", validate="one_to_one", indicator="_html_merge",
    )
    if (merged.pop("_html_merge") != "both").any():
        raise ValueError("HTML v2 cache does not cover every training SHA-256")
    merged = merged.merge(
        temporal_assignments[["sha256", "split"]], on="sha256", how="left",
        validate="one_to_one", indicator="_temporal_merge",
    )
    if (merged.pop("_temporal_merge") != "both").any():
        raise ValueError("Temporal assignments do not cover every training SHA-256")
    merged = merged.merge(
        grouped_assignments[["sha256", "registered_domain", "fold"]].rename(
            columns={"registered_domain": "group_domain"}
        ),
        on="sha256", how="left", validate="one_to_one", indicator="_grouped_merge",
    )
    if (merged.pop("_grouped_merge") != "both").any():
        raise ValueError("Grouped assignments do not cover every training SHA-256")
    merged["target"] = merged["label"].map({"benign": 0, "phish": 1})
    if merged["target"].isna().any():
        raise ValueError("Unexpected label in training data")
    return merged


def load_phase4_data():
    """Read training shards and the verified HTML v2 cache; never reads test files."""
    raw_files = sorted(RAW_DIR.glob("train-*.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"No train-*.parquet files in {RAW_DIR}")
    raw_rows = pd.concat(
        [pd.read_parquet(path, columns=["sha256", "url", "label"]) for path in raw_files],
        ignore_index=True,
    )
    raw_rows["sha256"] = raw_rows["sha256"].astype(str)
    url_features = pd.DataFrame(
        [extract_url_features_v2(url) for url in raw_rows["url"].fillna("").astype(str)],
        columns=URL_FEATURE_COLUMNS_V2,
    )
    url_features.insert(0, "sha256", raw_rows["sha256"].to_numpy())

    cache_dir = HTML_CACHE_ROOT / HTML_FEATURE_HASH
    manifest = json.loads((cache_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("feature_spec_hash") != HTML_FEATURE_HASH:
        raise ValueError("HTML v2 cache is incomplete or has a different feature specification")
    html_features = pd.concat(
        [pd.read_parquet(cache_dir / item["output_file"]) for item in manifest["shards"].values()],
        ignore_index=True,
    )
    if set(html_features["feature_spec_hash"].dropna().unique()) != {HTML_FEATURE_HASH}:
        raise ValueError("HTML cache contains an unexpected feature specification")
    html_features = html_features[["sha256", *NUMERIC_HTML_FEATURES]]

    temporal = pd.read_csv(SPLIT_DIR / "temporal_assignments.csv.gz", dtype={"sha256": str})
    grouped = pd.read_csv(
        SPLIT_DIR / "grouped_domain_folds.csv.gz",
        usecols=["sha256", "registered_domain", "fold"],
        dtype={"sha256": str, "registered_domain": str},
    )
    data = assemble_phase4_data(raw_rows, html_features, temporal, grouped)
    data = data.merge(url_features, on="sha256", how="left", validate="one_to_one")
    if data[URL_FEATURE_COLUMNS_V2].isna().all(axis=1).any():
        raise ValueError("URL v2 feature extraction produced an empty feature row")
    return data