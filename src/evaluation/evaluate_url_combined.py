import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
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

    print("\nTest dataset:")
    print("Rows:", len(data))

    print("\nLabels:")
    print(data["label"].value_counts())

    X_test = data["url"].astype(str)
    y_test = data["label"]

    print("\nLoading Experiment 2 models...")

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

    print(
        "TF-IDF shape:",
        X_test_tfidf.shape
    )

    print("\nExtracting URL features...")

    from scipy.sparse import csr_matrix, hstack

    from pathlib import Path
    import sys

    sys.path.append(
        str(
            Path(__file__).resolve().parents[1]
            / "features"
        )
    )

    from url_features import extract_url_features

    test_features = pd.DataFrame(
        [
            extract_url_features(url)
            for url in X_test
        ]
    )

    test_features_scaled = scaler.transform(
        test_features
    )

    test_features_sparse = csr_matrix(
        test_features_scaled
    )

    print(
        "URL feature shape:",
        test_features_sparse.shape
    )

    print("\nCombining features...")

    X_test_combined = hstack([
        X_test_tfidf,
        test_features_sparse
    ])

    print(
        "Combined shape:",
        X_test_combined.shape
    )

    print("\nMaking predictions...")

    predictions = model.predict(
        X_test_combined
    )

    scores = model.decision_function(
        X_test_combined
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT 2 - OFFICIAL TEST RESULTS")
    print("=" * 60)

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "benign",
                "phish"
            ]
        )
    )

    print("Confusion matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    ap = average_precision_score(
        y_test,
        scores
    )

    print(
        "\nAverage Precision (AP):",
        round(ap, 4)
    )


if __name__ == "__main__":
    main()