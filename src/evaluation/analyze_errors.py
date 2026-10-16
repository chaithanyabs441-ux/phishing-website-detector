import pandas as pd
from pathlib import Path

from sklearn.metrics import confusion_matrix
import joblib


DATA_PATH = Path("data/raw/data")
MODEL_PATH = Path("models")


def load_test_data():
    files = sorted(DATA_PATH.glob("test-*.parquet"))

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

    print("Loading test data...")

    data = load_test_data()

    X_test = data["url"].astype(str)
    y_test = data["label"]

    print("Loading model...")

    vectorizer = joblib.load(
        MODEL_PATH / "url_tfidf.joblib"
    )

    model = joblib.load(
        MODEL_PATH / "url_linear_svm.joblib"
    )

    print("Creating TF-IDF features...")

    X_test_tfidf = vectorizer.transform(X_test)

    print("Predicting...")

    predictions = model.predict(X_test_tfidf)
    scores = model.decision_function(X_test_tfidf)

    data["prediction"] = predictions
    data["score"] = scores

    # False positives:
    # Actual benign, predicted phishing
    false_positives = data[
        (data["label"] == 0) &
        (data["prediction"] == 1)
    ].copy()

    # False negatives:
    # Actual phishing, predicted benign
    false_negatives = data[
        (data["label"] == 1) &
        (data["prediction"] == 0)
    ].copy()

    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    print("\nFalse positives:", len(false_positives))
    print("False negatives:", len(false_negatives))

    print("\n--- FALSE POSITIVE EXAMPLES ---")

    for url in false_positives["url"].head(20):
        print(url)

    print("\n--- FALSE NEGATIVE EXAMPLES ---")

    for url in false_negatives["url"].head(20):
        print(url)

    # Save examples for later analysis
    false_positives[
        ["url", "label", "prediction", "score"]
    ].head(1000).to_csv(
        "models/false_positives.csv",
        index=False
    )

    false_negatives[
        ["url", "label", "prediction", "score"]
    ].head(1000).to_csv(
        "models/false_negatives.csv",
        index=False
    )

    print("\nSaved:")
    print("models/false_positives.csv")
    print("models/false_negatives.csv")


if __name__ == "__main__":
    main()