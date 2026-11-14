"""Build a resumable, training-only parquet cache for HTML feature v2."""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.html_features_v2 import (  # noqa: E402
    FEATURE_SPEC_HASH_V2,
    FEATURE_SPEC_V2,
    HTML_FEATURE_COLUMNS_V2,
    MAX_HTML_CHARS,
    extract_html_features_v2,
)


RAW_DIR = PROJECT_ROOT / "data" / "raw" / "data"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "data" / "processed" / "features_v2"
EXPERIMENT_RECORD = PROJECT_ROOT / "experiments" / "phase3_html_sample.json"
SOURCE_COLUMNS = ["sha256", "url", "label", "html"]
OUTPUT_COLUMNS = ["sha256", "url", "label", *HTML_FEATURE_COLUMNS_V2, "feature_spec_hash"]


def _digest(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _extract_record(html, url, sha256):
    try:
        return extract_html_features_v2(html, url)
    except Exception as error:
        html_text = html if isinstance(html, str) else ""
        parsed_length = min(len(html_text), MAX_HTML_CHARS)
        return {
            **{column: 0 for column in HTML_FEATURE_COLUMNS_V2 if column != "cleaned_text"},
            "html_missing": int(not html_text.strip()),
            "html_length": len(html_text),
            "parsed_html_length": parsed_length,
            "html_truncated": int(len(html_text) > MAX_HTML_CHARS),
            "html_parse_error": 1,
            "cleaned_text": "",
            "_error_type": type(error).__name__,
            "_sha256": str(sha256),
        }


def _frame_from_records(records):
    rows = []
    parse_errors = 0
    for sha256, url, label, html in records:
        features = _extract_record(html, url, sha256)
        if features.get("_error_type"):
            parse_errors += 1
            features.pop("_error_type", None)
            features.pop("_sha256", None)
        rows.append({
            "sha256": str(sha256),
            "url": "" if url is None else str(url),
            "label": str(label),
            **features,
            "feature_spec_hash": FEATURE_SPEC_HASH_V2,
        })
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return frame, parse_errors


def _log_line(path, message):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")


def _process_shard(shard_path, version_dir, row_batch_size):
    shard_path = Path(shard_path)
    output_path = version_dir / f"{shard_path.stem}_features_v2.parquet"
    temporary_path = output_path.with_suffix(".tmp.parquet")
    progress_path = version_dir / "progress" / f"{shard_path.stem}.log"
    row_count = 0
    error_count = 0
    writer = None
    started = time.monotonic()
    parquet = pq.ParquetFile(shard_path)
    try:
        for batch in parquet.iter_batches(batch_size=row_batch_size, columns=SOURCE_COLUMNS):
            source = batch.to_pandas()
            records = source[["sha256", "url", "label", "html"]].itertuples(index=False, name=None)
            frame, errors = _frame_from_records(records)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary_path, table.schema, compression="zstd")
            writer.write_table(table)
            row_count += len(frame)
            error_count += errors
            if row_count % 5000 < row_batch_size:
                _log_line(progress_path, f"rows={row_count}/{parquet.metadata.num_rows} parse_errors={error_count}")
        if writer is None:
            raise ValueError(f"No rows found in training shard {shard_path.name}")
    finally:
        if writer is not None:
            writer.close()
    os.replace(temporary_path, output_path)
    result = {
        "source_file": shard_path.name,
        "output_file": output_path.name,
        "rows": row_count,
        "parse_errors": error_count,
        "missing_by_label": {},
        "output_sha256": _digest(output_path),
        "wall_seconds": time.monotonic() - started,
    }
    out = pq.read_table(output_path, columns=["label", "html_missing"]).to_pandas()
    result["missing_by_label"] = {
        str(label): {"rows": int(len(group)), "missing_html": int(group["html_missing"].sum())}
        for label, group in out.groupby("label")
    }
    return result


def _read_sample(shards, sample_rows):
    records = []
    source_names = []
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=min(1000, sample_rows), columns=SOURCE_COLUMNS):
            frame = batch.to_pandas()
            records.extend(frame.itertuples(index=False, name=None))
            source_names.append(shard.name)
            if len(records) >= sample_rows:
                return records[:sample_rows], sorted(set(source_names))
    return records, sorted(set(source_names))


def _sample_extract(records, n_jobs, chunk_size=250):
    from joblib import Parallel, delayed

    chunks = [records[start:start + chunk_size] for start in range(0, len(records), chunk_size)]
    extracted = Parallel(n_jobs=n_jobs, backend="loky", batch_size=1)(
        delayed(_frame_from_records)(chunk) for chunk in chunks
    )
    frames = [frame for frame, _ in extracted]
    errors = sum(count for _, count in extracted)
    return pd.concat(frames, ignore_index=True), errors


def _write_sample(cache_root, records, source_names, n_jobs):
    sample_started = time.monotonic()
    cache_root = Path(cache_root)
    sample_dir = cache_root / FEATURE_SPEC_HASH_V2 / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    frame, parse_errors = _sample_extract(records, n_jobs)
    output_path = sample_dir / f"sample_{len(frame)}.parquet"
    frame.to_parquet(output_path, index=False, compression="zstd")
    elapsed = time.monotonic() - sample_started
    missing_by_label = {
        str(label): {
            "rows": int(len(group)),
            "missing_html": int(group["html_missing"].sum()),
            "missing_rate": float(group["html_missing"].mean()),
        }
        for label, group in frame.groupby("label")
    }
    manifest = {
        "feature_spec": FEATURE_SPEC_V2,
        "feature_spec_hash": FEATURE_SPEC_HASH_V2,
        "sample_mode": True,
        "rows": len(frame),
        "source_files": source_names,
        "n_jobs": n_jobs,
        "parse_errors": parse_errors,
        "missing_html_by_label": missing_by_label,
        "elapsed_seconds": elapsed,
        "rows_per_second": len(frame) / max(elapsed, 1e-9),
        "output_file": output_path.name,
        "output_sha256": _digest(output_path),
        "official_test_accessed": False,
    }
    sample_manifest_path = sample_dir / f"sample_{len(frame)}_MANIFEST.json"
    sample_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    EXPERIMENT_RECORD.parent.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_RECORD.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Sample rows: {len(frame)} from training files: {', '.join(source_names)}")
    print(f"Feature spec SHA-256: {FEATURE_SPEC_HASH_V2}")
    print(f"Parse errors: {parse_errors}")
    print(f"Missing HTML by label: {json.dumps(missing_by_label, sort_keys=True)}")
    print(f"Throughput: {manifest['rows_per_second']:.2f} rows/s; elapsed {elapsed:.2f}s")
    print(f"Wrote sample cache: {output_path}")
    return manifest


def build_cache(input_dir=RAW_DIR, cache_root=DEFAULT_CACHE_ROOT, n_jobs=2, timeout_seconds=1800,
                row_batch_size=500, sample_rows=None):
    shards = sorted(Path(input_dir).glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No training shards at {input_dir}")
    if sample_rows:
        records, source_names = _read_sample(shards, sample_rows)
        return _write_sample(cache_root, records, source_names, n_jobs)

    version_dir = Path(cache_root) / FEATURE_SPEC_HASH_V2
    version_dir.mkdir(parents=True, exist_ok=True)
    progress_log = version_dir / "progress.log"
    output_manifest_path = version_dir / "MANIFEST.json"
    existing_manifest = {}
    if output_manifest_path.exists():
        try:
            existing_manifest = json.loads(output_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_manifest = {}
    completed = existing_manifest.get("shards", {})
    current_names = {shard.name for shard in shards}
    results = {name: result for name, result in completed.items() if name in current_names}
    work = []
    for shard in shards:
        output_path = version_dir / f"{shard.stem}_features_v2.parquet"
        cached = completed.get(shard.name, {})
        valid_cached = False
        if output_path.exists() and cached.get("output_sha256"):
            valid_cached = _digest(output_path) == cached["output_sha256"]
        if valid_cached:
            _log_line(progress_log, f"skip_complete shard={shard.name}")
        else:
            work.append(shard)

    run_started = time.monotonic()
    from joblib import Parallel, delayed

    parallel = Parallel(n_jobs=n_jobs, backend="loky", timeout=timeout_seconds,
                        return_as="generator_unordered")
    for index, result in enumerate(
        parallel(delayed(_process_shard)(path, version_dir, row_batch_size) for path in work),
        start=1,
    ):
        results[result["source_file"]] = result
        _log_line(progress_log, f"completed={result['source_file']} rows={result['rows']} seconds={result['wall_seconds']:.2f}")
        print(f"Completed {index}/{len(work)}: {result['source_file']} ({result['rows']} rows, {result['wall_seconds']:.1f}s)", flush=True)
        partial_manifest = {
            "feature_spec": FEATURE_SPEC_V2,
            "feature_spec_hash": FEATURE_SPEC_HASH_V2,
            "source_glob": "train-*.parquet",
            "rows": sum(int(item["rows"]) for item in results.values()),
            "n_jobs": n_jobs,
            "row_batch_size": row_batch_size,
            "shard_timeout_seconds": timeout_seconds,
            "parse_errors": sum(int(item.get("parse_errors", 0)) for item in results.values()),
            "shards": results,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "official_test_accessed": False,
            "complete": False,
        }
        output_manifest_path.write_text(
            json.dumps(partial_manifest, indent=2) + "\n", encoding="utf-8"
        )

    total_rows = sum(int(result["rows"]) for result in results.values())
    total_parse_errors = sum(int(result.get("parse_errors", 0)) for result in results.values())
    missing_totals = {}
    for result in results.values():
        for label, counts in result.get("missing_by_label", {}).items():
            aggregate = missing_totals.setdefault(label, {"rows": 0, "missing_html": 0})
            aggregate["rows"] += counts["rows"]
            aggregate["missing_html"] += counts["missing_html"]
    for counts in missing_totals.values():
        counts["missing_rate"] = counts["missing_html"] / max(1, counts["rows"])

    manifest = {
        "feature_spec": FEATURE_SPEC_V2,
        "feature_spec_hash": FEATURE_SPEC_HASH_V2,
        "source_glob": "train-*.parquet",
        "rows": total_rows,
        "n_jobs": n_jobs,
        "row_batch_size": row_batch_size,
        "shard_timeout_seconds": timeout_seconds,
        "parse_errors": total_parse_errors,
        "missing_html_by_label": missing_totals,
        "shards": results,
        "elapsed_seconds_this_run": time.monotonic() - run_started,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "official_test_accessed": False,
        "complete": all(shard.name in results for shard in shards),
    }
    output_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _log_line(progress_log, f"manifest_written complete={manifest['complete']} rows={total_rows}")
    print(f"Cache complete: {manifest['complete']}; rows: {total_rows}; parse errors: {total_parse_errors}")
    print(f"Missing HTML by label: {json.dumps(missing_totals, sort_keys=True)}")
    print(f"Feature spec hash: {FEATURE_SPEC_HASH_V2}")
    print(f"Elapsed this run: {manifest['elapsed_seconds_this_run']:.2f}s")
    print(f"Manifest: {output_manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--row-batch-size", type=int, default=500)
    parser.add_argument("--sample", type=int, default=None, help="Process only the first N training rows")
    args = parser.parse_args()
    build_cache(args.input_dir, args.cache_root, args.n_jobs, args.timeout_seconds,
                args.row_batch_size, args.sample)


if __name__ == "__main__":
    main()
