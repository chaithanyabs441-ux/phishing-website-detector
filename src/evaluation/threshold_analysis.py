import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    average_precision_score
)

import joblib


DATA_PATH = Path("data/raw/data")
MODEL_PATH = Path("models")


def load_test_data():

    files = sorted(DATA_PATH.glob("test-*.parquet"))

    print("Loading test data...")
    print("Files:", len(files))

    data = []

    for file in files:

        df = pd.read_parquet(
            file,
            columns=["url", "label"]
        )

        data.append(df)

    data = pd.concat(
        data,
        ignore_index=True
    )

    data["label"] = data["label"].map({
        "benign": 0,
        "phish": 1
    })

    data = data.dropna(
        subset=["url", "label"]
    )

    return data


def main():

    data = load_test_data()

    X_test = data["url"].astype(str)
    y_test = data["label"].values

    print("\nLoading Experiment 2 model...")

    vectorizer = joblib.load(
        MODEL_PATH / "combined_tfidf.joblib"
    )

    scaler = joblib.load(
        MODEL_PATH / "url_feature_scaler.joblib"
    )

    model = joblib.load(
        MODEL_PATH / "combined_linear_svm.joblib"
    )

    print("\nCreating TF-IDF features...")

    X_test_tfidf = vectorizer.transform(X_test)

    print("TF-IDF shape:", X_test_tfidf.shape)

    # Import URL feature extractor
    import sys

    sys.path.append(
        str(
            Path(__file__).resolve().parents[1]
            / "features"
        )
    )

    from url_features import extract_url_features

    print("\nExtracting URL features...")

    test_features = pd.DataFrame(
        [
            extract_url_features(url)
            for url in X_test
        ]
    )

    test_features_scaled = scaler.transform(
        test_features
    )

    from scipy.sparse import csr_matrix, hstack

    test_features_sparse = csr_matrix(
        test_features_scaled
    )

    X_test_combined = hstack([
        X_test_tfidf,
        test_features_sparse
    ])

    print(
        "Combined shape:",
        X_test_combined.shape
    )

    print("\nGetting model decision scores...")

    scores = model.decision_function(
        X_test_combined
    )

    print(
        "Average Precision:",
        round(
            average_precision_score(
                y_test,
                scores
            ),
            4
        )
    )

    # -------------------------------------------------
    # THRESHOLD ANALYSIS
    # -------------------------------------------------

    thresholds = [
        -1.0,
        -0.75,
        -0.50,
        -0.25,
        -0.10,
        0.00,
        0.10,
        0.25,
        0.50,
        0.75,
        1.00
    ]

    results = []

    print("\n")
    print("=" * 90)
    print("THRESHOLD ANALYSIS")
    print("=" * 90)

    print(
        f"{'Threshold':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'FPR':>10}"
        f"{'FN':>10}"
        f"{'FP':>10}"
    )

    print("-" * 90)

    for threshold in thresholds:

        predictions = (
            scores >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions
        ).ravel()

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        fpr = fp / (fp + tn)

        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "fpr": fpr,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "true_negatives": tn
        })

        print(
            f"{threshold:>10.2f}"
            f"{precision:>12.4f}"
            f"{recall:>10.4f}"
            f"{fpr:>10.4f}"
            f"{fn:>10}"
            f"{fp:>10}"
        )

    # -------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        MODEL_PATH / "threshold_results.csv",
        index=False
    )

    print("\nSaved:")
    print(
        "models/threshold_results.csv"
    )


if __name__ == "__main__":
    main()