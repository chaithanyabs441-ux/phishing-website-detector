"""Leakage-safe temporal holdout baseline for the existing three SVM designs."""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    PREVALENCES as PREVALENCES,
    _domain_bootstrap as _domain_bootstrap,
    _metric_bundle as _metric_bundle,
)
from src.features.url_features import URL_FEATURE_COLUMNS_V1, extract_url_features_v1  # noqa: E402


RAW_DIR = PROJECT_ROOT / "data" / "raw" / "data"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
HTML_DIR = PROJECT_ROOT / "data" / "processed" / "html_features"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "phase2_svm_baseline.json"
CONFIG_PATH = PROJECT_ROOT / "configs" / "svm_blocked_baseline.yaml"


def _load_data():
    raw_files = sorted(RAW_DIR.glob("train-*.parquet"))
    rows = pd.concat(
        [pd.read_parquet(path, columns=["sha256", "url", "label", "date"]) for path in raw_files],
        ignore_index=True,
    )
    html_files = sorted(HTML_DIR.glob("train-*_html_features.csv"))
    html_columns = joblib.load(MODEL_DIR / "html_feature_columns.joblib")
    feature_columns = ["url", "label", *html_columns]
    html = pd.concat(
        [pd.read_csv(path, usecols=feature_columns) for path in html_files],
        ignore_index=True,
    )
    if html["url"].duplicated().any():
        raise ValueError("Cached HTML features contain repeated URLs; refusing an ambiguous join")
    merged = rows.merge(
        html,
        on=["url", "label"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing = int((merged["_merge"] != "both").sum())
    if missing:
        raise ValueError(f"Cached HTML features missing for {missing} training rows")
    merged = merged.drop(columns="_merge")

    temporal = pd.read_csv(SPLIT_DIR / "temporal_assignments.csv.gz", dtype={"sha256": str})
    merged = merged.merge(temporal, on="sha256", how="inner", validate="one_to_one")
    grouped = pd.read_csv(
        SPLIT_DIR / "grouped_domain_folds.csv.gz",
        usecols=["sha256", "registered_domain", "fold"],
        dtype={"sha256": str, "registered_domain": str},
    )
    merged = merged.merge(grouped, on="sha256", how="inner", validate="one_to_one")
    merged = merged[merged["split"].isin(["train", "validation"])].reset_index(drop=True)
    merged["target"] = merged["label"].map({"benign": 0, "phish": 1})
    if merged["target"].isna().any():
        raise ValueError("Unexpected training label encountered")
    return merged, html_columns


def run_evaluation(bootstrap_replicates=200, seed=42):
    started = time.monotonic()
    data, html_columns = _load_data()
    train = data[data["split"] == "train"].copy()
    validation = data[data["split"] == "validation"].copy()
    y_train = train["target"].to_numpy(dtype=np.int8)
    y_valid = validation["target"].to_numpy(dtype=np.int8)
    train_urls = train["url"].astype(str)
    validation_urls = validation["url"].astype(str)

    vectorizer_start = time.monotonic()
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 5), min_df=2,
        max_features=200000, sublinear_tf=True,
    )
    X_train_text = vectorizer.fit_transform(train_urls)
    X_valid_text = vectorizer.transform(validation_urls)
    vectorizer_seconds = time.monotonic() - vectorizer_start

    url_train = pd.DataFrame(
        [extract_url_features_v1(url) for url in train_urls],
        columns=URL_FEATURE_COLUMNS_V1,
    )
    url_valid = pd.DataFrame(
        [extract_url_features_v1(url) for url in validation_urls],
        columns=URL_FEATURE_COLUMNS_V1,
    )
    url_scaler = StandardScaler()
    X_train_url = csr_matrix(url_scaler.fit_transform(url_train))
    X_valid_url = csr_matrix(url_scaler.transform(url_valid))

    html_scaler = StandardScaler()
    X_train_html = csr_matrix(html_scaler.fit_transform(train[html_columns].astype(float)))
    X_valid_html = csr_matrix(html_scaler.transform(validation[html_columns].astype(float)))

    variants = {
        "url_tfidf": (X_train_text, X_valid_text),
        "url_engineered": (
            hstack([X_train_text, X_train_url], format="csr"),
            hstack([X_valid_text, X_valid_url], format="csr"),
        ),
        "url_html": (
            hstack([X_train_text, X_train_url, X_train_html], format="csr"),
            hstack([X_valid_text, X_valid_url, X_valid_html], format="csr"),
        ),
    }

    domains = validation["registered_domain"].fillna("").astype(str)
    unique_domains, domain_codes = np.unique(domains, return_inverse=True)
    results = {}
    for index, (name, (X_train, X_valid)) in enumerate(variants.items()):
        fit_start = time.monotonic()
        model = LinearSVC(C=1.0, class_weight="balanced", dual=True, max_iter=5000)
        model.fit(X_train, y_train)
        fit_seconds = time.monotonic() - fit_start
        scores = model.decision_function(X_valid)
        point = _metric_bundle(y_valid, scores)
        ci, completed = _domain_bootstrap(
            y_valid, scores, domain_codes, len(unique_domains), bootstrap_replicates,
            seed + index,
        )
        results[name] = {
            "metrics": point,
            "domain_bootstrap_95ci": ci,
            "bootstrap_replicates_completed": completed,
            "training_seconds": fit_seconds,
            "validation_rows": int(len(y_valid)),
            "training_rows": int(len(y_train)),
            "feature_matrix_shape": list(X_train.shape),
        }
        print(
            f"{name}: AP={point['average_precision']:.6f}, "
            f"ROC-AUC={point['roc_auc']:.6f}, "
            f"precision={point['precision']:.6f}, recall={point['recall']:.6f}, "
            f"F1={point['f1']:.6f}, FPR={point['false_positive_rate']:.6f}"
        )
        del model, scores

    split_hash = hashlib.sha256((SPLIT_DIR / "MANIFEST.json").read_bytes()).hexdigest()
    split_manifest = json.loads((SPLIT_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
    output = {
        "experiment": "phase2_temporal_holdout_svm_baseline",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": "cb02b34",
        "seed": seed,
        "split_manifest_sha256": split_hash,
        "split_files_sha256": {
            name: item["sha256"] for name, item in split_manifest["files"].items()
        },
        "temporal_cutoff_utc": json.loads((SPLIT_DIR / "leakage_audit.json").read_text(encoding="utf-8"))["cutoff_date_utc"],
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "validation_domains": int(len(unique_domains)),
        "vectorizer_seconds": vectorizer_seconds,
        "bootstrap": {
            "resampling_unit": "registered_domain",
            "requested_replicates": bootstrap_replicates,
            "confidence_level": 0.95,
        },
        "params": {
            "vectorizer": {
                "analyzer": "char", "ngram_range": [2, 5], "min_df": 2,
                "max_features": 200000, "sublinear_tf": True,
            },
            "classifier": {
                "type": "LinearSVC", "C": 1.0, "class_weight": "balanced",
                "dual": True, "max_iter": 5000,
            },
            "threshold": 0.0,
        },
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "models": results,
        "wall_seconds": time.monotonic() - started,
        "artifacts_saved": False,
        "official_test_accessed": False,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Vectorizer fit/transform seconds: {vectorizer_seconds:.2f}")
    print(f"Registered-domain bootstrap: {bootstrap_replicates} requested replicates")
    print(f"Wrote experiment record: {OUTPUT_PATH}")
    print(f"Total wall seconds: {output['wall_seconds']:.2f}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_evaluation(args.bootstrap_replicates, args.seed)


if __name__ == "__main__":
    main()
