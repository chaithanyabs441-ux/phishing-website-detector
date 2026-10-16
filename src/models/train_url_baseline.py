import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    confusion_matrix
)
import joblib


DATA_PATH = Path("data/raw/data")
MODEL_PATH = Path("models")

MODEL_PATH.mkdir(parents=True, exist_ok=True)


def load_training_data():
    files = sorted(DATA_PATH.glob("train-*.parquet"))

    print("Loading training data...")
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
    data = load_training_data()

    print("\nDataset:")
    print("Rows:", len(data))
    print("\nLabels:")
    print(data["label"].value_counts())

    X = data["url"].astype(str)
    y = data["label"]

    # Development validation split.
    # The official test set remains completely untouched.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining rows:", len(X_train))
    print("Validation rows:", len(X_valid))

    print("\nCreating character TF-IDF features...")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_features=200000,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_valid_tfidf = vectorizer.transform(X_valid)

    print("Training feature shape:", X_train_tfidf.shape)
    print("Validation feature shape:", X_valid_tfidf.shape)

    print("\nTraining Linear SVM...")

    model = LinearSVC(
        C=1.0,
        class_weight="balanced"
    )

    model.fit(X_train_tfidf, y_train)

    print("\nMaking predictions...")

    predictions = model.predict(X_valid_tfidf)
    scores = model.decision_function(X_valid_tfidf)

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    print("\nClassification report:")
    print(
        classification_report(
            y_valid,
            predictions,
            target_names=["benign", "phish"]
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_valid, predictions))

    ap = average_precision_score(y_valid, scores)

    print("\nAverage Precision (AP):", round(ap, 4))

    print("\nSaving model...")

    joblib.dump(
        vectorizer,
        MODEL_PATH / "url_tfidf.joblib"
    )

    joblib.dump(
        model,
        MODEL_PATH / "url_linear_svm.joblib"
    )

    print("Saved:")
    print(" ", MODEL_PATH / "url_tfidf.joblib")
    print(" ", MODEL_PATH / "url_linear_svm.joblib")


if __name__ == "__main__":
    main()