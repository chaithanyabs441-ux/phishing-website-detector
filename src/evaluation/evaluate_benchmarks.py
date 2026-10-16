import os
import gc
import glob
import time
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import average_precision_score, precision_recall_curve


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================
# FIND BENCHMARK DIRECTORY
# ============================================================

possible_roots = [
    os.path.join(PROJECT_ROOT, "benchmark"),
    os.path.join(PROJECT_ROOT, "data", "benchmark"),
    os.path.join(PROJECT_ROOT, "data", "benchmarks"),
    os.path.join(PROJECT_ROOT, "data", "benchmarks", "benchmark"),
]

BENCHMARK_ROOT = None

for root in possible_roots:
    benign_file = os.path.join(root, "benign.txt")

    if os.path.exists(benign_file):
        BENCHMARK_ROOT = root
        break


if BENCHMARK_ROOT is None:
    print("ERROR: Could not find benchmark directory.")
    print()
    print("Expected a folder containing:")
    print("  benign.txt")
    print("  benchmark-005/")
    print("  benchmark-010/")
    print("  benchmark-050/")
    print("  benchmark-100/")
    print("  benchmark-500/")
    raise SystemExit(1)


print("=" * 70)
print("PHRESH PHISH BENCHMARK EVALUATION")
print("=" * 70)
print("Project root   :", PROJECT_ROOT)
print("Benchmark root :", BENCHMARK_ROOT)
print()


# ============================================================
# MODEL FILES
# ============================================================

print("Loading models...")

url_tfidf = joblib.load(
    os.path.join(MODEL_DIR, "url_tfidf.joblib")
)

url_svm = joblib.load(
    os.path.join(MODEL_DIR, "url_linear_svm.joblib")
)

combined_tfidf = joblib.load(
    os.path.join(MODEL_DIR, "combined_tfidf.joblib")
)

combined_svm = joblib.load(
    os.path.join(MODEL_DIR, "combined_linear_svm.joblib")
)

url_scaler = joblib.load(
    os.path.join(MODEL_DIR, "url_engineered_scaler.joblib")
)


# ============================================================
# FEATURE ORDER
# ============================================================

FEATURE_COLUMNS = [
    "url_length",
    "hostname_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_digits",
    "num_special_chars",
    "num_subdomains",
    "num_query_params",
    "has_at_symbol",
    "has_ip_address",
    "has_https",
    "suspicious_word_count",
]


# ============================================================
# URL FEATURE EXTRACTION
# ============================================================

from urllib.parse import urlparse
import re


SUSPICIOUS_WORDS = [
    "login",
    "signin",
    "verify",
    "verification",
    "secure",
    "account",
    "update",
    "confirm",
    "password",
    "bank",
    "wallet",
    "payment",
    "invoice",
    "auth",
]


def extract_url_features(url):

    if not isinstance(url, str):
        url = ""

    try:
        parsed = urlparse(url)

        hostname = parsed.hostname or ""
        path = parsed.path or ""
        query = parsed.query or ""

    except Exception:
        hostname = ""
        path = ""
        query = ""

    url_lower = url.lower()

    num_query_params = 0

    if query:
        num_query_params = query.count("&") + 1

    suspicious_word_count = sum(
        word in url_lower
        for word in SUSPICIOUS_WORDS
    )

    try:
        ip_pattern = (
            r"^(?:\d{1,3}\.){3}\d{1,3}$"
        )

        has_ip = int(
            re.match(ip_pattern, hostname) is not None
        )

    except Exception:
        has_ip = 0

    if hostname:
        subdomain_parts = hostname.split(".")
        num_subdomains = max(
            0,
            len(subdomain_parts) - 2
        )
    else:
        num_subdomains = 0

    return {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": sum(
            not c.isalnum() for c in url
        ),
        "num_subdomains": num_subdomains,
        "num_query_params": num_query_params,
        "has_at_symbol": int("@" in url),
        "has_ip_address": has_ip,
        "has_https": int(url_lower.startswith("https://")),
        "suspicious_word_count": suspicious_word_count,
    }


def extract_feature_dataframe(urls):

    rows = []

    for url in urls:
        rows.append(
            extract_url_features(url)
        )

    return pd.DataFrame(
        rows,
        columns=FEATURE_COLUMNS
    )


# ============================================================
# P@R=0.90
# ============================================================

def precision_at_recall_90(y_true, scores):

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        scores
    )

    valid = np.where(recall >= 0.90)[0]

    if len(valid) == 0:
        return np.nan

    # Choose the point closest to recall = 0.90
    best_index = valid[
        np.argmin(
            np.abs(recall[valid] - 0.90)
        )
    ]

    return float(precision[best_index])


# ============================================================
# LOAD URL LIST
# ============================================================

def load_urls(path):

    urls = []

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                urls.append(line)

    return urls


# ============================================================
# LOAD BENIGN DATA
# ============================================================

benign_path = os.path.join(
    BENCHMARK_ROOT,
    "benign.txt"
)

print("Loading benign benchmark URLs...")

benign_urls = load_urls(benign_path)

print(
    "Benign URLs    :",
    len(benign_urls)
)

print()


# ============================================================
# PRECOMPUTE BENIGN SCORES
#
# IMPORTANT:
# The same benign benchmark set is reused across the
# benchmark datasets.
#
# So we calculate it ONCE instead of 975 times.
# ============================================================

print("=" * 70)
print("PRECOMPUTING BENIGN SCORES")
print("=" * 70)


# -----------------------------
# EXPERIMENT 1
# URL ONLY
# -----------------------------

print("Exp1 URL TF-IDF...")

benign_url_matrix = url_tfidf.transform(
    benign_urls
)

benign_exp1_scores = url_svm.decision_function(
    benign_url_matrix
)

del benign_url_matrix
gc.collect()


# -----------------------------
# EXPERIMENT 2
# URL + ENGINEERED FEATURES
# -----------------------------

print("Exp2 URL + engineered features...")

benign_tfidf = combined_tfidf.transform(
    benign_urls
)

benign_features = extract_feature_dataframe(
    benign_urls
)

# Keep feature names so StandardScaler does not warn.
benign_scaled = url_scaler.transform(
    benign_features
)

benign_scaled = csr_matrix(
    benign_scaled
)

benign_combined = hstack(
    [
        benign_tfidf,
        benign_scaled
    ],
    format="csr"
)

benign_exp2_scores = combined_svm.decision_function(
    benign_combined
)

del benign_tfidf
del benign_features
del benign_scaled
del benign_combined

gc.collect()

print("Benign preprocessing complete.")
print()


# ============================================================
# DISCOVER BENCHMARK FILES
# ============================================================

rate_folders = [
    ("benchmark-005", 0.0005),
    ("benchmark-010", 0.0010),
    ("benchmark-050", 0.0050),
    ("benchmark-100", 0.0100),
    ("benchmark-500", 0.0500),
]


benchmark_files = []

for folder, rate in rate_folders:

    folder_path = os.path.join(
        BENCHMARK_ROOT,
        folder
    )

    if not os.path.isdir(folder_path):
        print(
            "WARNING: missing folder:",
            folder_path
        )
        continue

    files = sorted(
        glob.glob(
            os.path.join(
                folder_path,
                "*.txt"
            )
        )
    )

    for path in files:

        benchmark_files.append(
            (
                path,
                folder,
                rate
            )
        )


print(
    "Benchmark files found:",
    len(benchmark_files)
)

print()


# ============================================================
# OUTPUT
# ============================================================

results = []

output_csv = os.path.join(
    LOG_DIR,
    "phresh_benchmark_results.csv"
)


# ============================================================
# MAIN EVALUATION
# ============================================================

total = len(benchmark_files)

start_time = time.time()


for index, (
    benchmark_path,
    folder,
    base_rate
) in enumerate(
    benchmark_files,
    start=1
):

    filename = os.path.basename(
        benchmark_path
    )

    print(
        f"[{index}/{total}] "
        f"{folder}/{filename}"
    )

    # ----------------------------------------
    # Load phishing URLs
    # ----------------------------------------

    phish_urls = load_urls(
        benchmark_path
    )

    n_benign = len(benign_urls)
    n_phish = len(phish_urls)

    actual_rate = (
        n_phish /
        (n_benign + n_phish)
    )

    print(
        f"Phishing rate : {actual_rate:.6f}"
    )

    print(
        f"Benign rows   : {n_benign}"
    )

    print(
        f"Phish rows    : {n_phish}"
    )

    print(
        f"Total rows    : "
        f"{n_benign + n_phish}"
    )


    # ========================================================
    # EXPERIMENT 1
    # ========================================================

    phish_url_matrix = url_tfidf.transform(
        phish_urls
    )

    phish_exp1_scores = url_svm.decision_function(
        phish_url_matrix
    )

    del phish_url_matrix

    exp1_scores = np.concatenate(
        [
            benign_exp1_scores,
            phish_exp1_scores
        ]
    )

    y_true = np.concatenate(
        [
            np.zeros(n_benign, dtype=int),
            np.ones(n_phish, dtype=int)
        ]
    )

    exp1_ap = average_precision_score(
        y_true,
        exp1_scores
    )

    exp1_p90 = precision_at_recall_90(
        y_true,
        exp1_scores
    )


    # ========================================================
    # EXPERIMENT 2
    # ========================================================

    phish_tfidf = combined_tfidf.transform(
        phish_urls
    )

    phish_features = extract_feature_dataframe(
        phish_urls
    )

    phish_scaled = url_scaler.transform(
        phish_features
    )

    phish_scaled = csr_matrix(
        phish_scaled
    )

    phish_combined = hstack(
        [
            phish_tfidf,
            phish_scaled
        ],
        format="csr"
    )

    phish_exp2_scores = combined_svm.decision_function(
        phish_combined
    )

    del phish_tfidf
    del phish_features
    del phish_scaled
    del phish_combined

    gc.collect()


    exp2_scores = np.concatenate(
        [
            benign_exp2_scores,
            phish_exp2_scores
        ]
    )

    exp2_ap = average_precision_score(
        y_true,
        exp2_scores
    )

    exp2_p90 = precision_at_recall_90(
        y_true,
        exp2_scores
    )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    ap_difference = (
        exp2_ap - exp1_ap
    )

    results.append(
        {
            "benchmark": filename,
            "base_rate": base_rate,
            "actual_rate": actual_rate,
            "n_benign": n_benign,
            "n_phish": n_phish,
            "total_rows": n_benign + n_phish,
            "exp1_ap": exp1_ap,
            "exp1_p_at_r90": exp1_p90,
            "exp2_ap": exp2_ap,
            "exp2_p_at_r90": exp2_p90,
            "ap_difference_exp2_minus_exp1":
                ap_difference,
        }
    )


    # ----------------------------------------
    # Print current result
    # ----------------------------------------

    print(
        f"Exp1 AP       : {exp1_ap:.6f}"
    )

    print(
        f"Exp1 P@R90    : {exp1_p90:.6f}"
    )

    print(
        f"Exp2 AP       : {exp2_ap:.6f}"
    )

    print(
        f"Exp2 P@R90    : {exp2_p90:.6f}"
    )

    print(
        f"AP difference : {ap_difference:+.6f}"
    )

    print()


    # ----------------------------------------
    # Save checkpoint after every benchmark
    # ----------------------------------------

    pd.DataFrame(results).to_csv(
        output_csv,
        index=False
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    output_csv,
    index=False
)


print("=" * 70)
print("BENCHMARK EVALUATION COMPLETE")
print("=" * 70)

print(
    "Results saved:",
    output_csv
)

print()


# ============================================================
# SUMMARY BY BASE RATE
# ============================================================

summary = (
    results_df
    .groupby("base_rate")
    .agg(
        n_benchmarks=("benchmark", "count"),

        exp1_ap_mean=("exp1_ap", "mean"),
        exp1_ap_std=("exp1_ap", "std"),

        exp2_ap_mean=("exp2_ap", "mean"),
        exp2_ap_std=("exp2_ap", "std"),

        ap_difference_mean=(
            "ap_difference_exp2_minus_exp1",
            "mean"
        ),

        exp1_p90_mean=(
            "exp1_p_at_r90",
            "mean"
        ),

        exp2_p90_mean=(
            "exp2_p_at_r90",
            "mean"
        ),
    )
    .reset_index()
)


summary_csv = os.path.join(
    LOG_DIR,
    "phresh_benchmark_summary.csv"
)

summary.to_csv(
    summary_csv,
    index=False
)


print("=" * 70)
print("SUMMARY BY PHISHING BASE RATE")
print("=" * 70)

print(
    summary.to_string(
        index=False
    )
)

print()

print(
    "Summary saved:",
    summary_csv
)

print()

elapsed = time.time() - start_time

print(
    f"Total evaluation time: "
    f"{elapsed / 60:.2f} minutes"
)