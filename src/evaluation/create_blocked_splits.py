"""Create frozen train-only temporal and registered-domain grouped splits."""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.url_features import registered_domain_from_url  # noqa: E402


RAW_DIR = PROJECT_ROOT / "data" / "raw" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"
REQUIRED_COLUMNS = ["sha256", "url", "label", "date"]


def load_train_rows(input_dir=RAW_DIR):
    files = sorted(Path(input_dir).glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No train-*.parquet files in {input_dir}")
    frames = [pd.read_parquet(path, columns=REQUIRED_COLUMNS) for path in files]
    frame = pd.concat(frames, ignore_index=True)
    frame["sha256"] = frame["sha256"].astype(str)
    frame["url"] = frame["url"].fillna("").astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    if frame[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Training rows contain missing sha256, label, or date values")
    return frame


def _group_domain(url, sha256):
    domain = registered_domain_from_url(url)
    if domain:
        return domain
    try:
        host = (urlsplit(url if "://" in url else "//" + url).hostname or "").lower()
    except ValueError:
        host = ""
    return host or f"__no_domain__:{sha256}"


def _choose_date_cutoff(frame, train_fraction):
    counts = frame.groupby(frame["date"].dt.floor("D")).size().sort_index()
    threshold = len(frame) * train_fraction
    cumulative = counts.cumsum()
    position = int((cumulative < threshold).sum())
    return counts.index[min(position, len(counts) - 1)]


def _path_template(url):
    try:
        parsed = urlsplit(url if "://" in url else "//" + url)
        host = (parsed.hostname or "").lower()
        path = parsed.path or "/"
    except ValueError:
        host, path = "", ""
    path = re.sub(r"(?i)[0-9a-f]{8}-[0-9a-f-]{27,}", "{uuid}", path)
    path = re.sub(r"\d{2,}", "{num}", path)
    path = re.sub(r"(?i)(?=[a-z0-9_-]{12,})(?=[a-z0-9_-]*[a-z])(?=[a-z0-9_-]*\d)[a-z0-9_-]+", "{id}", path)
    return f"{host}{path}"


def build_assignments(frame, train_fraction=0.8, n_splits=5):
    if not 0.5 <= train_fraction < 1:
        raise ValueError("train_fraction must be in [0.5, 1)")
    if frame["sha256"].duplicated().any():
        # Repeated row hashes are grouped into one temporal and CV assignment.
        pass

    domains = pd.Series(
        (_group_domain(url, sha) for url, sha in zip(frame["url"], frame["sha256"])),
        index=frame.index,
        dtype="string",
    )
    frame = frame.copy()
    frame["registered_domain"] = domains

    cutoff = _choose_date_cutoff(frame, train_fraction)
    by_sha = frame.groupby("sha256", sort=False).agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        registered_domain=("registered_domain", "first"),
        domain_count=("registered_domain", "nunique"),
    ).reset_index()
    spans_cutoff = (by_sha["first_date"] < cutoff) & (by_sha["last_date"] >= cutoff)
    by_sha["split"] = "purged_cross_boundary"
    by_sha.loc[by_sha["last_date"] < cutoff, "split"] = "train"
    by_sha.loc[by_sha["first_date"] >= cutoff, "split"] = "validation"
    by_sha.loc[spans_cutoff, "split"] = "purged_cross_boundary"

    unique_sha = by_sha[["sha256", "registered_domain"]].copy()
    splitter = GroupKFold(n_splits=n_splits)
    unique_sha["fold"] = -1
    dummy = pd.Series(0, index=unique_sha.index)
    for fold, (_, validation_indices) in enumerate(
        splitter.split(dummy, groups=unique_sha["registered_domain"])
    ):
        unique_sha.loc[validation_indices, "fold"] = fold

    audit = audit_training_data(frame, by_sha, cutoff, train_fraction, n_splits)
    audit["sha_domain_conflicts"] = int((by_sha["domain_count"] > 1).sum())
    grouped_rows = frame.merge(unique_sha[["sha256", "fold"]], on="sha256", how="left")
    grouped_stats = {}
    for fold in range(n_splits):
        validation_rows = grouped_rows[grouped_rows["fold"] == fold]
        training_rows = grouped_rows[grouped_rows["fold"] != fold]
        validation_domains = set(validation_rows["registered_domain"])
        training_domains = set(training_rows["registered_domain"])
        grouped_stats[str(fold)] = {
            "train_rows": int(len(training_rows)),
            "validation_rows": int(len(validation_rows)),
            "train_domains": int(len(training_domains)),
            "validation_domains": int(len(validation_domains)),
            "domain_overlap": int(len(training_domains & validation_domains)),
            "train_labels": {str(k): int(v) for k, v in training_rows["label"].value_counts().items()},
            "validation_labels": {str(k): int(v) for k, v in validation_rows["label"].value_counts().items()},
            "train_date_min": training_rows["date"].min().isoformat(),
            "train_date_max": training_rows["date"].max().isoformat(),
            "validation_date_min": validation_rows["date"].min().isoformat(),
            "validation_date_max": validation_rows["date"].max().isoformat(),
        }
    audit["grouped_fold_stats"] = grouped_stats
    return by_sha, unique_sha, audit


def audit_training_data(frame, assignments, cutoff, train_fraction, n_splits):
    labels = frame["label"].astype(str)
    frame = frame.copy()
    frame["label"] = labels
    split_lookup = assignments.set_index("sha256")["split"]
    frame["split"] = frame["sha256"].map(split_lookup)
    usable = frame[frame["split"].isin(["train", "validation"])].copy()
    url_groups = frame.groupby("url", dropna=False)
    conflicting_urls = int((url_groups["label"].nunique(dropna=False) > 1).sum())
    exact_duplicate_urls = int(frame["url"].duplicated(keep="first").sum())
    templates = frame["url"].map(_path_template)
    near_duplicate_rows = int(templates.duplicated(keep="first").sum())

    train_domains = set(usable.loc[usable["split"] == "train", "registered_domain"])
    validation_domains = set(usable.loc[usable["split"] == "validation", "registered_domain"])
    domain_overlap = train_domains & validation_domains
    by_split = {}
    for split, group in frame.groupby("split"):
        dates = group["date"]
        by_split[split] = {
            "rows": int(len(group)),
            "unique_sha256": int(group["sha256"].nunique()),
            "labels": {str(k): int(v) for k, v in group["label"].value_counts().items()},
            "date_min": dates.min().isoformat() if len(dates) else None,
            "date_max": dates.max().isoformat() if len(dates) else None,
        }

    return {
        "source": "data/raw/data/train-*.parquet only",
        "official_test_overlap_audit": "deferred: official test files are locked until Phase 7",
        "rows": int(len(frame)),
        "unique_sha256": int(frame["sha256"].nunique()),
        "cutoff_date_utc": cutoff.isoformat(),
        "requested_train_fraction": train_fraction,
        "actual_temporal_split": by_split,
        "cross_boundary_sha256_purged": int((assignments["split"] == "purged_cross_boundary").sum()),
        "exact_duplicate_url_rows_after_first": exact_duplicate_urls,
        "urls_with_conflicting_labels": conflicting_urls,
        "near_duplicate_host_path_template_rows_after_first": near_duplicate_rows,
        "near_duplicate_template_rate": near_duplicate_rows / max(1, len(frame)),
        "registered_domain_overlap_temporal": len(domain_overlap),
        "registered_domain_overlap_sample": sorted(domain_overlap)[:10],
        "grouped_folds": n_splits,
        "grouped_splitter": "deterministic GroupKFold; no random seed",
        "date_ranges_and_class_balance": by_split,
    }


def _write_gzip_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
        lineterminator="\n",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_splits(input_dir=RAW_DIR, output_dir=DEFAULT_OUTPUT_DIR, train_fraction=0.8, n_splits=5):
    started = datetime.now(timezone.utc)
    frame = load_train_rows(input_dir)
    temporal, grouped, audit = build_assignments(frame, train_fraction, n_splits)
    output_dir = Path(output_dir)
    temporal_path = output_dir / "temporal_assignments.csv.gz"
    grouped_path = output_dir / "grouped_domain_folds.csv.gz"
    audit_path = output_dir / "leakage_audit.json"
    temporal_hash = _write_gzip_csv(temporal[["sha256", "split"]], temporal_path)
    grouped_hash = _write_gzip_csv(grouped[["sha256", "registered_domain", "fold"]], grouped_path)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    audit_hash = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    manifest = {
        "manifest_version": 1,
        "source_directory": str(Path(input_dir).resolve().relative_to(PROJECT_ROOT)),
        "source_glob": "train-*.parquet",
        "rows": int(len(frame)),
        "unique_sha256": int(frame["sha256"].nunique()),
        "split_config": {
            "temporal_train_fraction": train_fraction,
            "temporal_cutoff_rule": "first UTC day whose cumulative row count reaches the requested train fraction; purge hashes spanning the cutoff",
            "grouping": "offline Public Suffix List registered domain",
            "grouped_folds": n_splits,
            "grouped_splitter": "deterministic GroupKFold; no random seed",
        },
        "files": {
            temporal_path.name: {"sha256": temporal_hash},
            grouped_path.name: {"sha256": grouped_hash},
            audit_path.name: {"sha256": audit_hash},
        },
        "created_utc": started.isoformat(),
        "git_commit": "cb02b34",
    }
    (output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Training rows: {len(frame)}; unique SHA-256: {frame['sha256'].nunique()}")
    print(f"Temporal cutoff UTC: {audit['cutoff_date_utc']}")
    print(f"Temporal train/validation/purged: {audit['actual_temporal_split'].get('train', {}).get('rows', 0)}/{audit['actual_temporal_split'].get('validation', {}).get('rows', 0)}/{audit['actual_temporal_split'].get('purged_cross_boundary', {}).get('rows', 0)} rows")
    print(f"Temporal registered-domain overlap: {audit['registered_domain_overlap_temporal']}")
    print(f"Exact duplicate URL rows: {audit['exact_duplicate_url_rows_after_first']}; label-conflict URLs: {audit['urls_with_conflicting_labels']}")
    print(f"Near-duplicate host/path template rows: {audit['near_duplicate_host_path_template_rows_after_first']}")
    print(f"Wrote splits: {temporal_path.name}, {grouped_path.name}")
    print(f"Wrote hashes: {output_dir / 'MANIFEST.json'}")
    print(f"Elapsed seconds: {(datetime.now(timezone.utc) - started).total_seconds():.2f}")
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    create_splits(args.input_dir, args.output_dir, args.train_fraction, args.folds)


if __name__ == "__main__":
    main()
