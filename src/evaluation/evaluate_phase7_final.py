"""One-pass locked-test evaluation and SHA-mapped PhreshPhish benchmarks."""

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from joblib import load
from scipy.special import logit
from sklearn.metrics import average_precision_score

from src.evaluation.evaluate_svm_blocked import _domain_bootstrap, _metric_bundle
from src.evaluation.phase4_boosting import prepare_feature_frames
from src.evaluation.phase4_data import (
    CATEGORICAL_FEATURES,
    PHASE4_FEATURE_COLUMNS,
    PROJECT_ROOT,
    load_phase4_data,
)
from src.evaluation.phase7_calibration import apply_calibrator, join_hash_predictions, precision_at_recall
from src.features.html_features_v2 import (
    HTML_FEATURE_COLUMNS_V2,
    MAX_HTML_CHARS,
    _empty_features,
    extract_html_features_v2,
)
from src.features.url_features import URL_FEATURE_COLUMNS_V2, extract_url_features_v2


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "phase7_calibration.yaml"
RESULT_PATH = PROJECT_ROOT / "experiments" / "phase7_final_evaluation.json"
TEST_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "phase7_test_predictions.parquet"
BENCHMARK_RESULTS_PATH = PROJECT_ROOT / "experiments" / "phase7_benchmark_results.csv"
BENCHMARK_SUMMARY_PATH = PROJECT_ROOT / "experiments" / "phase7_benchmark_summary.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "data"
SOURCE_COLUMNS = ["sha256", "url", "label", "html"]
NUMERIC_HTML_COLUMNS = [column for column in HTML_FEATURE_COLUMNS_V2 if column != "cleaned_text"]


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def confusion_at_threshold(targets, probabilities, threshold):
    targets = np.asarray(targets, dtype=np.int8)
    predicted = np.asarray(probabilities) >= threshold
    tp = int(np.sum(predicted & (targets == 1)))
    fp = int(np.sum(predicted & (targets == 0)))
    tn = int(np.sum(~predicted & (targets == 0)))
    fn = int(np.sum(~predicted & (targets == 1)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _prepare_feature_batch(batch_frame, train_features, numeric_medians):
    urls = batch_frame["url"].fillna("").astype(str).tolist()
    html_values = batch_frame["html"].fillna("").astype(str).tolist()
    url_records = [extract_url_features_v2(url) for url in urls]
    records = []
    for url, html, url_features in zip(urls, html_values, url_records):
        try:
            html_features = extract_html_features_v2(html, url)
        except Exception:
            # Match the training cache's error-row representation. Malformed
            # page URLs can make urljoin raise while resolving HTML resources.
            html_features = _empty_features(
                html_length=len(html),
                parsed_length=min(len(html), MAX_HTML_CHARS),
                html_missing=int(not html.strip()),
                truncated=int(len(html) > MAX_HTML_CHARS),
                parse_error=1,
            )
        records.append({**url_features, **html_features})
    features = pd.DataFrame(records, columns=[*URL_FEATURE_COLUMNS_V2, *NUMERIC_HTML_COLUMNS])
    for column in CATEGORICAL_FEATURES:
        categories = train_features[column].cat.categories
        values = features[column].fillna("__MISSING__").astype(str)
        values = values.where(values.isin(categories), "__UNKNOWN__")
        features[column] = pd.Categorical(values, categories=categories)
    numeric = [column for column in PHASE4_FEATURE_COLUMNS if column not in CATEGORICAL_FEATURES]
    features[numeric] = features[numeric].fillna(numeric_medians).fillna(0)
    return features[PHASE4_FEATURE_COLUMNS], [
        record.get("registered_domain") or f"__no_domain__:{sha}"
        for record, sha in zip(url_records, batch_frame.sha256.astype(str))
    ]


def score_test_once(config_path=None, batch_size=500):
    """Read official test shards in one streaming pass and write score-only predictions."""
    started = time.monotonic()
    config_path = Path(config_path or DEFAULT_CONFIG)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    training = load_phase4_data()
    train = training[training["split"] == "train"].reset_index(drop=True)
    train_features, _ = prepare_feature_frames(train)
    numeric = [column for column in PHASE4_FEATURE_COLUMNS if column not in CATEGORICAL_FEATURES]
    numeric_medians = train_features[numeric].median()

    model_path = PROJECT_ROOT / config["model_artifact"]
    calibrator_path = PROJECT_ROOT / "models" / "phase7" / "catboost_engineered_calibrator.joblib"
    thresholds_path = PROJECT_ROOT / "configs" / "thresholds.yaml"
    if not model_path.exists() or not calibrator_path.exists() or not thresholds_path.exists():
        raise FileNotFoundError("Finalist, calibrator, or threshold config is missing")
    model = load(model_path)
    calibrator = load(calibrator_path)
    thresholds = yaml.safe_load(thresholds_path.read_text(encoding="utf-8"))

    test_files = sorted(RAW_DIR.glob("test-*.parquet"))
    if not test_files:
        raise FileNotFoundError(f"No official test shards under {RAW_DIR}")
    outputs = []
    total_rows = 0
    for shard_index, path in enumerate(test_files, start=1):
        parquet = pq.ParquetFile(path)
        shard_rows = 0
        for batch in parquet.iter_batches(batch_size=batch_size, columns=SOURCE_COLUMNS):
            rows = batch.to_pandas()
            features, domains = _prepare_feature_batch(rows, train_features, numeric_medians)
            raw_probabilities = model.predict_proba(features)[:, 1]
            calibrated = apply_calibrator(calibrator, raw_probabilities)
            outputs.append(pd.DataFrame({
                "sha256": rows.sha256.astype(str).to_numpy(),
                "label": rows.label.map({"benign": 0, "phish": 1}).astype(np.int8).to_numpy(),
                "registered_domain": domains,
                "raw_probability": raw_probabilities.astype(np.float32),
                "probability": np.asarray(calibrated, dtype=np.float32),
            }))
            shard_rows += len(rows)
            total_rows += len(rows)
        print(f"test shard {shard_index}/{len(test_files)}: rows={shard_rows}", flush=True)

    predictions = pd.concat(outputs, ignore_index=True)
    duplicate_hashes = int(predictions.sha256.duplicated(keep=False).sum())
    TEST_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(TEST_PREDICTIONS_PATH, index=False, compression="zstd")
    return predictions, {
        "test_shards": [path.name for path in test_files],
        "test_rows": total_rows,
        "duplicate_sha256_rows": duplicate_hashes,
        "test_predictions_sha256": _sha256_file(TEST_PREDICTIONS_PATH),
        "model_sha256": _sha256_file(model_path),
        "calibrator_sha256": _sha256_file(calibrator_path),
        "threshold_config_sha256": _sha256_file(thresholds_path),
        "elapsed_seconds": time.monotonic() - started,
        "thresholds": thresholds["thresholds"],
    }


def evaluate_benchmark_hashes(predictions, benchmark_root):
    benchmark_root = Path(benchmark_root)
    duplicate_ids = set(predictions.loc[predictions.sha256.duplicated(keep=False), "sha256"].astype(str))
    benchmark_predictions = predictions[~predictions.sha256.astype(str).isin(duplicate_ids)]
    test_ids = set(benchmark_predictions.sha256.astype(str))
    dataset_predictions = benchmark_predictions.set_index("sha256")
    benign_hashes = [
        line.strip() for line in (benchmark_root / "benign.txt").read_text(
            encoding="utf-8", errors="ignore",
        ).splitlines() if line.strip()
    ]
    benign_unique = list(dict.fromkeys(benign_hashes))
    benign_mapped = [sha for sha in benign_unique if sha in test_ids and int(dataset_predictions.loc[sha, "label"]) == 0]
    benign_unknown = [sha for sha in benign_unique if sha not in test_ids]
    benign_wrong_label = [sha for sha in benign_unique if sha in test_ids and int(dataset_predictions.loc[sha, "label"]) != 0]
    rate_map = {
        "benchmark-005": 0.0005,
        "benchmark-010": 0.001,
        "benchmark-050": 0.005,
        "benchmark-100": 0.01,
        "benchmark-500": 0.05,
    }
    rows = []
    for folder, base_rate in rate_map.items():
        for path in sorted((benchmark_root / folder).glob("*.txt")):
            original_hashes = [
                line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
                if line.strip()
            ]
            unique_hashes = list(dict.fromkeys(original_hashes))
            phish_mapped = [sha for sha in unique_hashes if sha in test_ids and int(dataset_predictions.loc[sha, "label"]) == 1]
            phish_unknown = [sha for sha in unique_hashes if sha not in test_ids]
            phish_wrong_label = [sha for sha in unique_hashes if sha in test_ids and int(dataset_predictions.loc[sha, "label"]) != 1]
            ids = list(dict.fromkeys(benign_mapped + phish_mapped))
            joined = join_hash_predictions(ids, benchmark_predictions)
            targets = joined.label.to_numpy(dtype=np.int8)
            scores = joined.probability.to_numpy(dtype=np.float64)
            rows.append({
                "benchmark": path.name,
                "folder": folder,
                "base_rate": base_rate,
                "actual_rate": float(targets.mean()) if len(targets) else float("nan"),
                "n_benign": int((targets == 0).sum()),
                "n_phish": int((targets == 1).sum()),
                "n_total": int(len(targets)),
                "n_benign_input": len(benign_unique),
                "n_benign_unmatched": len(benign_unknown),
                "n_benign_wrong_label": len(benign_wrong_label),
                "n_phish_input": len(unique_hashes),
                "n_phish_unmatched": len(phish_unknown),
                "n_phish_wrong_label": len(phish_wrong_label),
                "n_ambiguous_test_hashes": int(len(duplicate_ids & set(unique_hashes))),
                "average_precision": float(average_precision_score(targets, scores)),
                "precision_at_recall_90": precision_at_recall(targets, scores, 0.90),
            })
    results = pd.DataFrame(rows)
    summary = results.groupby("base_rate").agg(
        n_benchmarks=("benchmark", "count"),
        average_precision_mean=("average_precision", "mean"),
        average_precision_std=("average_precision", "std"),
        precision_at_recall_90_mean=("precision_at_recall_90", "mean"),
        precision_at_recall_90_std=("precision_at_recall_90", "std"),
        actual_rate_mean=("actual_rate", "mean"),
        total_unmatched=("n_phish_unmatched", "sum"),
        total_wrong_label=("n_phish_wrong_label", "sum"),
    ).reset_index()
    return results, summary, {
        "unique_benign_hashes": len(benign_unique),
        "benign_hashes_mapped_as_benign": len(benign_mapped),
        "benign_hashes_unmatched": len(benign_unknown),
        "benign_hashes_wrong_label": len(benign_wrong_label),
        "benchmark_files": int(len(results)),
        "ambiguous_test_hashes": len(duplicate_ids),
        "phish_hashes_unmatched": int(results.n_phish_unmatched.sum()),
        "phish_hashes_wrong_label": int(results.n_phish_wrong_label.sum()),
    }


def run_final_evaluation(config_path=None, batch_size=500):
    started = time.monotonic()
    config_path = Path(config_path or DEFAULT_CONFIG)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    predictions, access = score_test_once(config_path, batch_size)
    calibration_record = json.loads((PROJECT_ROOT / "experiments" / "phase7_calibration.json").read_text(encoding="utf-8"))
    selected_method = calibration_record["selected_calibrator"]["method"]
    calibrated_validation = calibration_record["selected_calibrator"]["oof_metrics"]
    thresholds = access["thresholds"]
    targets = predictions.label.to_numpy(dtype=np.int8)
    probabilities = predictions.probability.to_numpy(dtype=np.float64)
    groups = predictions.registered_domain.fillna("").astype(str).to_numpy()
    unique_domains, domain_codes = np.unique(groups, return_inverse=True)
    score = logit(np.clip(probabilities, 1e-12, 1 - 1e-12))
    point = _metric_bundle(targets, score)
    intervals, completed = _domain_bootstrap(
        targets, score, domain_codes, len(unique_domains),
        int(config["bootstrap"]["replicates"]), int(config["seed"]),
    )
    threshold_confusions = {
        name: confusion_at_threshold(targets, probabilities, float(threshold))
        for name, threshold in {
            "probability_0.5": 0.5,
            "warn": thresholds["warn"],
            "block": thresholds["block"],
        }.items()
    }
    benchmark_results, benchmark_summary, benchmark_audit = evaluate_benchmark_hashes(
        predictions, PROJECT_ROOT / config["benchmark"]["root"],
    )
    BENCHMARK_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    benchmark_results.to_csv(BENCHMARK_RESULTS_PATH, index=False)
    benchmark_summary.to_csv(BENCHMARK_SUMMARY_PATH, index=False)
    output = {
        "experiment": "phase7_locked_test_and_hash_benchmark",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
        ).strip(),
        "pre_registration_commit": "86e804f",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "split_manifest_sha256": hashlib.sha256((PROJECT_ROOT / "data" / "splits" / "MANIFEST.json").read_bytes()).hexdigest(),
        "finalist": config["selected_model"],
        "calibration_method": selected_method,
        "calibration_validation_metrics": calibrated_validation,
        "test_rows": int(len(predictions)),
        "test_domains": int(len(unique_domains)),
        "test_metrics_probability_0.5": point,
        "domain_bootstrap_95ci": intervals,
        "bootstrap_replicates_completed": completed,
        "confusion_matrices": threshold_confusions,
        "selected_thresholds": thresholds,
        "test_access": access,
        "benchmark_audit": benchmark_audit,
        "benchmark_results_csv": str(BENCHMARK_RESULTS_PATH.relative_to(PROJECT_ROOT)),
        "benchmark_summary_csv": str(BENCHMARK_SUMMARY_PATH.relative_to(PROJECT_ROOT)),
        "benchmark_summary": benchmark_summary.to_dict(orient="records"),
        "official_test_accessed": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    RESULT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"test rows={len(predictions)}; AP={point['average_precision']:.6f}; ROC-AUC={point['roc_auc']:.6f}")
    print(f"benchmark files={benchmark_audit['benchmark_files']}; unmatched phish hashes={benchmark_audit['phish_hashes_unmatched']}")
    print(f"Wrote final Phase 7 record: {RESULT_PATH}; elapsed={output['elapsed_seconds']:.2f}s")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    run_final_evaluation(args.config, args.batch_size)


if __name__ == "__main__":
    main()
