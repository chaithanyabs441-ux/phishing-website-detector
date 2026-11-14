"""Measure missing/blank HTML by label in train shards only; never parse pages."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "data"
OUTPUT = PROJECT_ROOT / "experiments" / "phase3_html_missing_audit.json"


def audit(input_dir=RAW_DIR, batch_size=256):
    started = time.monotonic()
    counts = {}
    files = sorted(Path(input_dir).glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No train-*.parquet files in {input_dir}")
    rows = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["label", "html"]):
            frame = batch.to_pandas()
            missing = frame["html"].isna() | frame["html"].astype("string").str.strip().eq("")
            for label, group in frame.groupby("label", dropna=False):
                label_name = str(label)
                entry = counts.setdefault(label_name, {"rows": 0, "missing_html": 0})
                indices = group.index
                entry["rows"] += int(len(group))
                entry["missing_html"] += int(missing.loc[indices].sum())
            rows += len(frame)
    for entry in counts.values():
        entry["missing_rate"] = entry["missing_html"] / max(1, entry["rows"])
    result = {
        "source": "data/raw/data/train-*.parquet only",
        "files": len(files),
        "rows": rows,
        "missing_html_by_label": counts,
        "batch_size": batch_size,
        "elapsed_seconds": time.monotonic() - started,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "official_test_accessed": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Training files: {len(files)}; rows: {rows}")
    print(f"Missing HTML by label: {json.dumps(counts, sort_keys=True)}")
    print(f"Elapsed seconds: {result['elapsed_seconds']:.2f}")
    print(f"Wrote audit: {OUTPUT}")
    return result


if __name__ == "__main__":
    audit()
