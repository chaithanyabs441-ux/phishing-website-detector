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

    data = pd.concat(data, ignore_index=True)

    data["label"] = data["label"].map({
        "benign": 0,
        "phish": 1
    })

    data = data.dropna(subset=["url", "label"])

    return data


def main():
    data = load_test_data()

    print("\nTest dataset:")
    print("Rows:", len(data))
    print("\nLabels:")
    print(data["label"].value_counts())

    X_test = data["url"].astype(str)
    y_test = data["label"]

    print("\nLoading saved model...")

    vectorizer = joblib.load(
        MODEL_PATH / "url_tfidf.joblib"
    )

    model = joblib.load(
        MODEL_PATH / "url_linear_svm.joblib"
    )

    print("Creating TF-IDF features...")

    X_test_tfidf = vectorizer.transform(X_test)

    print("Test feature shape:", X_test_tfidf.shape)

    print("\nMaking predictions...")

    predictions = model.predict(X_test_tfidf)
    scores = model.decision_function(X_test_tfidf)

    print("\n" + "=" * 60)
    print("OFFICIAL TEST RESULTS")
    print("=" * 60)

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["benign", "phish"]
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