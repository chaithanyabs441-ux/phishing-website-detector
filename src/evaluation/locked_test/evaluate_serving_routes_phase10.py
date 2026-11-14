"""One-pass evaluation of the registered Phase 10 serving routes."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.engineered_feature_dataset import (
    CATEGORICAL_FEATURES,
    PHASE4_FEATURE_COLUMNS,
    PROJECT_ROOT,
    load_phase4_data,
)
from src.features.html_features_v2 import (
    HTML_FEATURE_COLUMNS_V2,
    MAX_HTML_CHARS,
    _empty_features,
    extract_html_features_v2,
)
from src.features.url_features import (
    URL_FEATURE_COLUMNS_V2,
    extract_url_features_v2,
    registered_domain_from_url,
)
from src.serving.model_registry import ModelRegistry, _apply_calibrator


RAW_DIR = PROJECT_ROOT / "data" / "raw" / "data"
THRESHOLDS_PATH = PROJECT_ROOT / "configs" / "thresholds.yaml"
RESULT_PATH = PROJECT_ROOT / "experiments" / "phase10_final_evaluation.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "phase_10_final_evaluation.md"
SOURCE_COLUMNS = ["sha256", "url", "label", "html"]
NUMERIC_HTML_COLUMNS = [
    column for column in HTML_FEATURE_COLUMNS_V2 if column != "cleaned_text"
]
BOOTSTRAP_REPLICATES = 200
BOOTSTRAP_SEED = 42
LOW_PREVALENCE = 0.0005


def _feature_frames(rows):
    """Build both serving routes with the same transforms used by ModelRegistry."""
    urls = rows["url"].fillna("").astype(str).tolist()
    html_values = rows["html"].fillna("").astype(str).tolist()
    url_records = [extract_url_features_v2(url) for url in urls]
    url_frame = pd.DataFrame(url_records, columns=URL_FEATURE_COLUMNS_V2)

    html_records = []
    for url, html in zip(urls, html_values):
        try:
            html_record = extract_html_features_v2(html, url)
        except Exception:
            html_record = _empty_features(
                html_length=len(html),
                parsed_length=min(len(html), MAX_HTML_CHARS),
                html_missing=int(not html.strip()),
                truncated=int(len(html) > MAX_HTML_CHARS),
                parse_error=1,
            )
        html_records.append(html_record)

    url_html_frame = pd.DataFrame(
        [{**url_record, **html_record} for url_record, html_record in zip(url_records, html_records)],
        columns=[*URL_FEATURE_COLUMNS_V2, *NUMERIC_HTML_COLUMNS],
    )
    for frame in (url_frame, url_html_frame):
        categorical = [column for column in CATEGORICAL_FEATURES if column in frame]
        for column in categorical:
            frame[column] = frame[column].fillna("__MISSING__").astype(str)
        numeric = [column for column in frame.columns if column not in categorical]
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce").fillna(0)

    return url_frame, url_html_frame, url_records


def _domain_bootstrap_ci(targets, scores, domain_codes, domain_count, replicates, seed):
    rng = np.random.default_rng(seed)
    ap_values = []
    auc_values = []
    for _ in range(replicates):
        sampled = rng.integers(0, domain_count, size=domain_count)
        multiplicity = np.bincount(sampled, minlength=domain_count)
        weights = multiplicity[domain_codes]
        included_targets = targets[weights > 0]
        if len(np.unique(included_targets)) < 2:
            continue
        ap_values.append(
            average_precision_score(targets, scores, sample_weight=weights),
        )
        auc_values.append(roc_auc_score(targets, scores, sample_weight=weights))
    if not ap_values:
        raise ValueError("Domain bootstrap produced no valid replicates")
    return {
        "average_precision": [
            float(np.quantile(ap_values, 0.025)),
            float(np.quantile(ap_values, 0.975)),
        ],
        "roc_auc": [
            float(np.quantile(auc_values, 0.025)),
            float(np.quantile(auc_values, 0.975)),
        ],
        "valid_replicates": len(ap_values),
    }


def _rates_at_threshold(targets, scores, threshold):
    predicted = scores >= threshold
    positive = targets == 1
    negative = ~positive
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & negative))
    fn = int(np.sum(~predicted & positive))
    tn = int(np.sum(~predicted & negative))
    recall = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    precision = tp / max(1, tp + fp)
    prevalence = LOW_PREVALENCE
    reweighted_precision = prevalence * recall / max(
        1e-15,
        prevalence * recall + (1 - prevalence) * fpr,
    )
    alerts_per_1000 = 1000 * (
        prevalence * recall + (1 - prevalence) * fpr
    )
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "precision_at_0_05_percent_prevalence": reweighted_precision,
        "alerts_per_1000_at_0_05_percent_prevalence": alerts_per_1000,
    }


def _metrics_for_slice(targets, scores, domains, threshold):
    if len(targets) == 0 or len(np.unique(targets)) < 2:
        return {"rows": int(len(targets)), "status": "insufficient_class_variation"}
    rates = _rates_at_threshold(targets, scores, threshold)
    return {
        "rows": int(len(targets)),
        "domains": int(pd.Series(domains).nunique()),
        "average_precision": float(average_precision_score(targets, scores)),
        "roc_auc": float(roc_auc_score(targets, scores)),
        "block_recall": rates["recall"],
        "block_fpr": rates["fpr"],
    }


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_official_test_once(batch_size=250):
    """Score both locked Phase 10 routes in one streaming pass."""
    started = time.monotonic()
    registry = ModelRegistry(PROJECT_ROOT)
    threshold_config = yaml.safe_load(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    model_config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "serving.yaml").read_text(encoding="utf-8"),
    )
    training = load_phase4_data()
    training_domains = set(
        training.loc[training["split"] == "train", "registered_domain"]
        .dropna().astype(str),
    )
    paths = sorted(RAW_DIR.glob("test-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No official test shards under {RAW_DIR}")

    targets_parts = []
    score_parts = {"url_only": [], "url_html": []}
    domain_parts = []
    seen_sha256 = set()
    duplicate_sha256_rows = 0
    total_rows = 0
    for shard_index, path in enumerate(paths, start=1):
        shard_rows = 0
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=SOURCE_COLUMNS):
            rows = batch.to_pandas()
            labels = rows["label"].astype(str).map({"benign": 0, "phish": 1})
            if labels.isna().any():
                raise ValueError("Unexpected label in official test shard")
            targets = labels.to_numpy(dtype=np.int8)
            sha_values = rows["sha256"].fillna("").astype(str).tolist()
            duplicate_sha256_rows += sum(sha in seen_sha256 for sha in sha_values)
            duplicate_sha256_rows += len(sha_values) - len(set(sha_values))
            seen_sha256.update(sha_values)

            url_frame, url_html_frame, url_records = _feature_frames(rows)
            for name, features in (
                ("url_only", url_frame),
                ("url_html", url_html_frame[PHASE4_FEATURE_COLUMNS]),
            ):
                raw = registry.primary_models[name].predict_proba(features)[:, 1]
                calibrated = _apply_calibrator(registry.calibrators[name], raw)
                score_parts[name].append(np.asarray(calibrated, dtype=np.float64))

            domains = []
            for url, sha, record in zip(rows.url.fillna("").astype(str), sha_values, url_records):
                domain = record.get("registered_domain") or registered_domain_from_url(url)
                domains.append(domain or f"__no_domain__:{sha}")
            targets_parts.append(targets)
            domain_parts.extend(domains)
            total_rows += len(rows)
            shard_rows += len(rows)
        print(
            f"official test shard {shard_index}/{len(paths)}: rows={shard_rows}",
            flush=True,
        )

    targets = np.concatenate(targets_parts)
    domains = np.asarray(domain_parts, dtype=object)
    domain_codes, unique_domains = pd.factorize(domains, sort=False)
    domain_count = len(unique_domains)
    seen_mask = np.asarray([domain in training_domains for domain in domains], dtype=bool)
    results = {
        "protocol": "one streaming pass; score only; no fitting, tuning, threshold changes, or selection",
        "test_rows": total_rows,
        "test_shards": [path.name for path in paths],
        "duplicate_sha256_rows": int(duplicate_sha256_rows),
        "registered_domains": int(domain_count),
        "phishing_prevalence": float(targets.mean()),
        "training_registered_domains": int(len(training_domains)),
        "artifact_sha256": {
            "thresholds": _hash_file(THRESHOLDS_PATH),
            "manifest": _hash_file(PROJECT_ROOT / "models" / "MANIFEST.json"),
            **{
                f"{route}_{artifact}": _hash_file(
                    PROJECT_ROOT / model_config["serving"][f"{route}_{artifact}"],
                )
                for route in ("url_only", "url_html")
                for artifact in ("model", "calibrator")
            },
        },
        "bootstrap": {
            "method": "registered-domain percentile bootstrap",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "routes": {},
        "seen_unseen_domains": {},
        "official_test_accessed": True,
        "elapsed_seconds": 0.0,
    }
    for route in ("url_only", "url_html"):
        scores = np.concatenate(score_parts[route])
        thresholds = threshold_config["models"][route]["thresholds"]
        results["routes"][route] = {
            "feature_version": threshold_config["models"][route]["feature_version"],
            "calibration_method": threshold_config["models"][route]["calibration_method"],
            "average_precision": float(average_precision_score(targets, scores)),
            "roc_auc": float(roc_auc_score(targets, scores)),
            "domain_bootstrap_ci": _domain_bootstrap_ci(
                targets,
                scores,
                domain_codes,
                domain_count,
                BOOTSTRAP_REPLICATES,
                BOOTSTRAP_SEED,
            ),
            "operating_points": {
                level: _rates_at_threshold(targets, scores, float(threshold))
                for level, threshold in thresholds.items()
            },
        }
        results["seen_unseen_domains"][route] = {}
        for group_name, group_mask in (
            ("seen_in_training", seen_mask),
            ("unseen_in_training", ~seen_mask),
        ):
            results["seen_unseen_domains"][route][group_name] = _metrics_for_slice(
                targets[group_mask],
                scores[group_mask],
                domains[group_mask],
                float(thresholds["block"]),
            )

    results["elapsed_seconds"] = time.monotonic() - started
    return results


def _write_report(results):
    lines = [
        "# Phase 10 — Authorized final test evaluation",
        "",
        "Both registered serving routes were scored during one streaming pass over the official test shards. The pass did not fit models or calibrators, select or change thresholds, or use test results for model selection.",
        "",
        f"Rows: {results['test_rows']:,}; registered domains: {results['registered_domains']:,}; phishing prevalence: {results['phishing_prevalence']:.3%}; duplicate SHA-256 rows: {results['duplicate_sha256_rows']:,}.",
        "",
        "Average precision and ROC-AUC intervals use 200 registered-domain bootstrap replicates. Threshold metrics use the cutoffs committed before this test pass. Low-prevalence values reweight test class-conditional rates to 0.05% and are estimates, not additional observations.",
        "",
        "| Route | Calibration | AP (95% domain CI) | ROC-AUC (95% domain CI) |",
        "|---|---|---:|---:|",
    ]
    for route, metrics in results["routes"].items():
        ap_ci = metrics["domain_bootstrap_ci"]["average_precision"]
        auc_ci = metrics["domain_bootstrap_ci"]["roc_auc"]
        lines.append(
            f"| {route} | {metrics['calibration_method']} | "
            f"{metrics['average_precision']:.6f} [{ap_ci[0]:.6f}, {ap_ci[1]:.6f}] | "
            f"{metrics['roc_auc']:.6f} [{auc_ci[0]:.6f}, {auc_ci[1]:.6f}] |",
        )
    lines.extend([
        "",
        "## Operating points",
        "",
        "| Route | Level | Threshold | Precision | Recall | FPR | Precision at 0.05% prevalence | Alerts / 1,000 at 0.05% prevalence |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for route, metrics in results["routes"].items():
        for level, point in metrics["operating_points"].items():
            lines.append(
                f"| {route} | {level.upper()} | {point['threshold']:.8f} | "
                f"{point['precision']:.6f} | {point['recall']:.6f} | {point['fpr']:.6f} | "
                f"{point['precision_at_0_05_percent_prevalence']:.6f} | "
                f"{point['alerts_per_1000_at_0_05_percent_prevalence']:.4f} |",
            )
    lines.extend([
        "",
        "## Seen and unseen training domains",
        "",
        "These groups are descriptive slices of this authorized Phase 10 evaluation. They were not used to tune, select, or change the models or thresholds.",
        "",
        "| Route | Domain group | Rows | Domains | AP | ROC-AUC | BLOCK recall | BLOCK FPR |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for route, groups in results["seen_unseen_domains"].items():
        for group_name, metrics in groups.items():
            lines.append(
                f"| {route} | {group_name.replace('_', ' ')} | {metrics['rows']:,} | "
                f"{metrics.get('domains', 0):,} | {metrics.get('average_precision', float('nan')):.6f} | "
                f"{metrics.get('roc_auc', float('nan')):.6f} | "
                f"{metrics.get('block_recall', float('nan')):.6f} | "
                f"{metrics.get('block_fpr', float('nan')):.6f} |",
            )
    lines.extend([
        "",
        f"Elapsed time: {results['elapsed_seconds']:.1f} seconds.",
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    results = score_official_test_once()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    _write_report(results)
    print(f"Wrote report: {REPORT_PATH}")
    print(f"Wrote summary: {RESULT_PATH}")


if __name__ == "__main__":
    main()
