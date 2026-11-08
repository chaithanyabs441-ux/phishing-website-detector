"""Five-fold registered-domain OOF validation of the three SVM designs."""

import hashlib
import json
import sys
import time
import warnings
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

from src.evaluation.evaluate_svm_blocked import (  # noqa: E402
    HTML_DIR,
    MODEL_DIR,
    RAW_DIR,
    SPLIT_DIR,
    _domain_bootstrap,
    _metric_bundle,
)
from src.features.url_features import URL_FEATURE_COLUMNS_V1, extract_url_features_v1  # noqa: E402


OUTPUT_GROUPED = PROJECT_ROOT / "experiments" / "phase2_grouped_cv.json"
CHECKPOINT_PATH = PROJECT_ROOT / "experiments" / "phase2_grouped_cv_checkpoint.npz"
CONFIG_PATH = PROJECT_ROOT / "configs" / "svm_blocked_baseline.yaml"


def run_grouped_cv(bootstrap_replicates=200, seed=42, resume=True):
    started = time.monotonic()
    raw_files = sorted(RAW_DIR.glob("train-*.parquet"))
    data = pd.concat(
        [pd.read_parquet(path, columns=["sha256", "url", "label"]) for path in raw_files],
        ignore_index=True,
    )
    html_columns = joblib.load(MODEL_DIR / "html_feature_columns.joblib")
    html_files = sorted(HTML_DIR.glob("train-*_html_features.csv"))
    html = pd.concat(
        [pd.read_csv(path, usecols=["url", "label", *html_columns]) for path in html_files],
        ignore_index=True,
    )
    if html["url"].duplicated().any():
        raise ValueError("Cached HTML features contain repeated URLs; refusing an ambiguous join")
    data = data.merge(html, on=["url", "label"], how="left", validate="one_to_one")
    if data[html_columns].isna().any().any():
        raise ValueError("HTML feature cache has missing values after the URL/label join")
    grouped = pd.read_csv(
        SPLIT_DIR / "grouped_domain_folds.csv.gz",
        dtype={"sha256": str, "registered_domain": str},
    )
    data = data.merge(grouped, on="sha256", how="inner", validate="one_to_one")
    data["target"] = data["label"].map({"benign": 0, "phish": 1})
    if data["target"].isna().any():
        raise ValueError("Unexpected label in training shards")

    urls = data["url"].astype(str).tolist()
    y = data["target"].to_numpy(dtype=np.int8)
    folds = data["fold"].to_numpy(dtype=np.int8)
    domains = data["registered_domain"].astype(str).to_numpy()
    domain_values, domain_codes = np.unique(domains, return_inverse=True)
    split_manifest_path = SPLIT_DIR / "MANIFEST.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    split_manifest_hash = hashlib.sha256(split_manifest_path.read_bytes()).hexdigest()
    url_features = pd.DataFrame(
        [extract_url_features_v1(url) for url in urls],
        columns=URL_FEATURE_COLUMNS_V1,
    )
    oof_scores = {
        "url_tfidf": np.full(len(data), np.nan, dtype=np.float32),
        "url_engineered": np.full(len(data), np.nan, dtype=np.float32),
        "url_html": np.full(len(data), np.nan, dtype=np.float32),
    }
    fit_seconds = {name: 0.0 for name in oof_scores}
    convergence_warnings = []
    fold_details = {}
    completed_folds = set()
    elapsed_before_resume = 0.0

    if resume and CHECKPOINT_PATH.exists():
        with np.load(CHECKPOINT_PATH, allow_pickle=False) as checkpoint:
            checkpoint_manifest_hash = str(checkpoint["split_manifest_hash"].item())
            if checkpoint_manifest_hash != split_manifest_hash:
                raise ValueError("Grouped OOF checkpoint belongs to different split files")
            if not np.array_equal(checkpoint["y"], y):
                raise ValueError("Grouped OOF checkpoint labels do not match the current training rows")
            completed_folds = set(checkpoint["completed_folds"].astype(int).tolist())
            for name in oof_scores:
                oof_scores[name] = checkpoint[f"scores_{name}"].copy()
            fit_seconds = json.loads(str(checkpoint["fit_seconds"].item()))
            fold_details = json.loads(str(checkpoint["fold_details"].item()))
            convergence_warnings = json.loads(str(checkpoint["convergence_warnings"].item()))
            elapsed_before_resume = float(checkpoint["elapsed_seconds"].item())
        print(f"Resuming completed folds: {sorted(completed_folds)}", flush=True)

    for fold in range(5):
        if fold in completed_folds:
            continue
        valid_indices = np.flatnonzero(folds == fold)
        train_indices = np.flatnonzero(folds != fold)
        print(
            f"Fold {fold + 1}/5: train={len(train_indices)}, "
            f"validation={len(valid_indices)}",
            flush=True,
        )
        train_urls = pd.Series(urls).iloc[train_indices]
        valid_urls = pd.Series(urls).iloc[valid_indices]
        vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 5), min_df=2,
            max_features=200000, sublinear_tf=True,
        )
        vectorizer_start = time.monotonic()
        text_train = vectorizer.fit_transform(train_urls)
        text_valid = vectorizer.transform(valid_urls)
        vectorizer_seconds = time.monotonic() - vectorizer_start

        url_scaler = StandardScaler()
        url_train = csr_matrix(url_scaler.fit_transform(url_features.iloc[train_indices]))
        url_valid = csr_matrix(url_scaler.transform(url_features.iloc[valid_indices]))
        html_scaler = StandardScaler()
        html_train = csr_matrix(
            html_scaler.fit_transform(data.iloc[train_indices][html_columns].astype(float))
        )
        html_valid = csr_matrix(
            html_scaler.transform(data.iloc[valid_indices][html_columns].astype(float))
        )
        matrices = {
            "url_tfidf": (text_train, text_valid),
            "url_engineered": (
                hstack([text_train, url_train], format="csr"),
                hstack([text_valid, url_valid], format="csr"),
            ),
            "url_html": (
                hstack([text_train, url_train, html_train], format="csr"),
                hstack([text_valid, url_valid, html_valid], format="csr"),
            ),
        }
        for name, (X_train, X_valid) in matrices.items():
            model = LinearSVC(C=1.0, class_weight="balanced", dual=True, max_iter=5000)
            fit_start = time.monotonic()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(X_train, y[train_indices])
            fit_seconds[name] += time.monotonic() - fit_start
            for warning in caught:
                if "ConvergenceWarning" in warning.category.__name__:
                    convergence_warnings.append({
                        "fold": fold,
                        "model": name,
                        "message": str(warning.message),
                    })
            oof_scores[name][valid_indices] = model.decision_function(X_valid).astype(np.float32)
            print(f"  {name} fit complete", flush=True)
            del model
        fold_details[str(fold)] = {
            "train_rows": int(len(train_indices)),
            "validation_rows": int(len(valid_indices)),
            "vectorizer_seconds": vectorizer_seconds,
            "validation_domains": int(len(np.unique(domains[valid_indices]))),
        }
        completed_folds.add(fold)
        checkpoint_temp = CHECKPOINT_PATH.with_suffix(".tmp.npz")
        np.savez_compressed(
            checkpoint_temp,
            y=y,
            split_manifest_hash=np.asarray(split_manifest_hash),
            completed_folds=np.asarray(sorted(completed_folds), dtype=np.int8),
            scores_url_tfidf=oof_scores["url_tfidf"],
            scores_url_engineered=oof_scores["url_engineered"],
            scores_url_html=oof_scores["url_html"],
            fit_seconds=np.asarray(json.dumps(fit_seconds)),
            fold_details=np.asarray(json.dumps(fold_details)),
            convergence_warnings=np.asarray(json.dumps(convergence_warnings)),
            elapsed_seconds=np.asarray(
                elapsed_before_resume + time.monotonic() - started
            ),
        )
        checkpoint_temp.replace(CHECKPOINT_PATH)
        print(f"Checkpoint saved after fold {fold + 1}", flush=True)
        del vectorizer, text_train, text_valid, url_train, url_valid
        del html_train, html_valid, matrices

    if any(np.isnan(scores).any() for scores in oof_scores.values()):
        raise RuntimeError("Not every row received exactly one out-of-fold score")
    results = {}
    for index, (name, scores) in enumerate(oof_scores.items()):
        point = _metric_bundle(y, scores)
        ci, completed = _domain_bootstrap(
            y, scores, domain_codes, len(domain_values), bootstrap_replicates, seed + index,
        )
        results[name] = {
            "metrics": point,
            "domain_bootstrap_95ci": ci,
            "bootstrap_replicates_completed": completed,
            "training_seconds_across_folds": fit_seconds[name],
            "oof_rows": int(len(scores)),
        }
        print(
            f"{name} grouped OOF: AP={point['average_precision']:.6f}, "
            f"ROC-AUC={point['roc_auc']:.6f}, precision={point['precision']:.6f}, "
            f"recall={point['recall']:.6f}, F1={point['f1']:.6f}, "
            f"FPR={point['false_positive_rate']:.6f}",
            flush=True,
        )

    output = {
        "experiment": "phase2_registered_domain_grouped_5fold_oof_svm_baseline",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": "cb02b34",
        "seed": seed,
        "split_manifest_sha256": split_manifest_hash,
        "split_files_sha256": {name: info["sha256"] for name, info in split_manifest["files"].items()},
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "rows": int(len(data)),
        "registered_domains": int(len(domain_values)),
        "folds": fold_details,
        "models": results,
        "convergence_warnings": convergence_warnings,
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
        "wall_seconds": elapsed_before_resume + time.monotonic() - started,
        "artifacts_saved": False,
        "official_test_accessed": False,
        "splitter_leakage_check": "each validation domain belongs to exactly one held-out fold; every row gets one OOF score",
    }
    OUTPUT_GROUPED.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_GROUPED.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    print(f"Convergence warnings: {len(convergence_warnings)}", flush=True)
    print(f"Wrote experiment record: {OUTPUT_GROUPED}", flush=True)
    print(f"Grouped CV wall seconds: {output['wall_seconds']:.2f}", flush=True)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    run_grouped_cv(args.bootstrap_replicates, args.seed, resume=not args.no_resume)
