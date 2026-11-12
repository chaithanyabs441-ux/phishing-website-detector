"""Describe locked-test performance for domains seen and unseen in training."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from sklearn.metrics import average_precision_score

from src.features.url_features import registered_domain_from_url


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "data"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "phase7_test_predictions.parquet"
THRESHOLDS_PATH = PROJECT_ROOT / "configs" / "thresholds.yaml"
REPORT_PATH = PROJECT_ROOT / "reports" / "phase_10a_domain_overlap_audit.md"


def load_train_domains(split_dir=SPLIT_DIR):
    """Return domains assigned to the frozen temporal training split only."""
    split_dir = Path(split_dir)
    temporal = pd.read_csv(
        split_dir / "temporal_assignments.csv.gz",
        usecols=["sha256", "split"],
        dtype={"sha256": str, "split": str},
    )
    grouped = pd.read_csv(
        split_dir / "grouped_domain_folds.csv.gz",
        usecols=["sha256", "registered_domain"],
        dtype={"sha256": str, "registered_domain": str},
    )
    if temporal["sha256"].duplicated().any() or grouped["sha256"].duplicated().any():
        raise ValueError("Frozen split assignments contain duplicate SHA-256 values")
    training = temporal.loc[temporal["split"] == "train", ["sha256"]].merge(
        grouped,
        on="sha256",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if (training["_merge"] != "both").any():
        raise ValueError("Training split is missing grouped registered-domain assignments")
    domains = training["registered_domain"].dropna().astype(str)
    return {domain for domain in domains if domain and domain != "__no_domain__"}


def read_test_identities(raw_dir=RAW_DIR, batch_size=10_000):
    """Read only SHA-256 and URL from official test shards, in batches."""
    files = sorted(Path(raw_dir).glob("test-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No official test Parquet shards found under {raw_dir}")
    pieces = []
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=["sha256", "url"],
        ):
            pieces.append(batch.to_pandas())
    identities = pd.concat(pieces, ignore_index=True)
    identities["sha256"] = identities["sha256"].astype(str)
    identities["url"] = identities["url"].fillna("").astype(str)
    return identities


def _metric_row(frame, seen):
    subset = frame.loc[frame["seen_in_train"] == seen]
    if subset.empty or subset["label"].nunique() != 2:
        raise ValueError("Each seen/unseen group must contain both labels")
    target = subset["label"].to_numpy(dtype=np.int8)
    score = subset["probability"].to_numpy(dtype=np.float64)
    predicted = score >= subset["block_threshold"].iloc[0]
    benign = target == 0
    phish = target == 1
    return {
        "rows": int(len(subset)),
        "domains": int(subset["registered_domain"].nunique()),
        "average_precision": float(average_precision_score(target, score)),
        "recall_at_block": float(np.sum(predicted & phish) / np.sum(phish)),
        "fpr_at_block": float(np.sum(predicted & benign) / np.sum(benign)),
    }


def audit_seen_unseen(test_identities, predictions, train_domains, block_threshold):
    """Join prior predictions to test identities and compute descriptive metrics."""
    required_predictions = {"sha256", "label", "probability"}
    missing = required_predictions - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction data missing required columns: {sorted(missing)}")
    if test_identities["sha256"].duplicated().any():
        raise ValueError("Test identity rows contain duplicate SHA-256 values")
    if predictions["sha256"].duplicated().any():
        raise ValueError("Prediction rows contain duplicate SHA-256 values")

    identities = test_identities[["sha256", "url"]].copy()
    identities["registered_domain"] = identities["url"].map(registered_domain_from_url)
    identities["registered_domain"] = identities["registered_domain"].fillna("").astype(str)
    identities["seen_in_train"] = identities["registered_domain"].map(
        lambda domain: bool(domain) and domain in train_domains
    )
    joined = identities.merge(
        predictions[["sha256", "label", "probability"]],
        on="sha256",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if (joined["_merge"] != "both").any():
        raise ValueError("Prior Phase 7 predictions do not cover every test SHA-256")
    joined = joined.drop(columns="_merge")
    joined["block_threshold"] = float(block_threshold)
    return {
        "test_rows": int(len(joined)),
        "training_domains": int(len(train_domains)),
        "seen": _metric_row(joined, True),
        "unseen": _metric_row(joined, False),
    }


def main():
    """Run the authorized descriptive audit and write its report."""
    thresholds = yaml.safe_load(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    identities = read_test_identities()
    predictions = pd.read_parquet(
        PREDICTIONS_PATH,
        columns=["sha256", "label", "probability"],
    )
    result = audit_seen_unseen(
        identities,
        predictions,
        load_train_domains(),
        thresholds["thresholds"]["block"],
    )
    result["block_threshold"] = float(thresholds["thresholds"]["block"])

    def fmt(metric):
        return f"{metric:.6f}"

    report = f"""# Phase 10A — Train-domain overlap audit

This is a descriptive slice of the already-authorized Phase 7 predictions. The official test shards were read only for `sha256` and `url`; labels and calibrated scores came from `{PREDICTIONS_PATH.relative_to(PROJECT_ROOT).as_posix()}`. Training domains came from the frozen temporal training assignments joined to grouped-domain assignments. No fitting or threshold selection was performed.

- Test rows joined: {result['test_rows']:,}
- Frozen training registered domains: {result['training_domains']:,}
- Existing validation-selected BLOCK threshold: {result['block_threshold']:.12g}

| Test domain group | Rows | Domains | AP | Recall at BLOCK | FPR at BLOCK |
|---|---:|---:|---:|---:|---:|
| Seen in training | {result['seen']['rows']:,} | {result['seen']['domains']:,} | {fmt(result['seen']['average_precision'])} | {fmt(result['seen']['recall_at_block'])} | {fmt(result['seen']['fpr_at_block'])} |
| Unseen in training | {result['unseen']['rows']:,} | {result['unseen']['domains']:,} | {fmt(result['unseen']['average_precision'])} | {fmt(result['unseen']['recall_at_block'])} | {fmt(result['unseen']['fpr_at_block'])} |

These test-set slice metrics are descriptive only and will not be used for model, feature, or threshold selection.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
