import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    confusion_matrix
)

import joblib

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "features"))

from url_features import extract_url_features


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

    data = data.dropna(
        subset=["url", "label"]
    )

    return data


def main():

    data = load_training_data()

    print("\nDataset:")
    print("Rows:", len(data))

    X = data["url"].astype(str)
    y = data["label"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining rows:", len(X_train))
    print("Validation rows:", len(X_valid))

    # -------------------------------------------------
    # PART 1: CHARACTER TF-IDF
    # -------------------------------------------------

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

    print(
        "TF-IDF training shape:",
        X_train_tfidf.shape
    )

    # -------------------------------------------------
    # PART 2: HAND-CRAFTED URL FEATURES
    # -------------------------------------------------

    print("\nExtracting URL features...")

    train_features = pd.DataFrame(
        [
            extract_url_features(url)
            for url in X_train
        ]
    )

    valid_features = pd.DataFrame(
        [
            extract_url_features(url)
            for url in X_valid
        ]
    )

    print(
        "URL feature shape:",
        train_features.shape
    )

    print("\nURL features:")
    print(list(train_features.columns))

    # Scale numerical features
    scaler = StandardScaler()

    train_features_scaled = scaler.fit_transform(
        train_features
    )

    valid_features_scaled = scaler.transform(
        valid_features
    )

    # Convert to sparse matrix
    from scipy.sparse import csr_matrix, hstack

    train_features_sparse = csr_matrix(
        train_features_scaled
    )

    valid_features_sparse = csr_matrix(
        valid_features_scaled
    )

    # -------------------------------------------------
    # PART 3: COMBINE BOTH
    # -------------------------------------------------

    print("\nCombining TF-IDF + URL features...")

    X_train_combined = hstack([
        X_train_tfidf,
        train_features_sparse
    ])

    X_valid_combined = hstack([
        X_valid_tfidf,
        valid_features_sparse
    ])

    print(
        "Combined training shape:",
        X_train_combined.shape
    )

    print(
        "Combined validation shape:",
        X_valid_combined.shape
    )

    # -------------------------------------------------
    # PART 4: TRAIN MODEL
    # -------------------------------------------------

    print("\nTraining Linear SVM...")

    model = LinearSVC(
        C=1.0,
        class_weight="balanced",
        dual=True,
        max_iter=5000
    )

    model.fit(
        X_train_combined,
        y_train
    )

    # -------------------------------------------------
    # PART 5: EVALUATION
    # -------------------------------------------------

    print("\nMaking predictions...")

    predictions = model.predict(
        X_valid_combined
    )

    scores = model.decision_function(
        X_valid_combined
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT 2 RESULTS")
    print("=" * 60)

    print("\nClassification report:")

    print(
        classification_report(
            y_valid,
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
            y_valid,
            predictions
        )
    )

    ap = average_precision_score(
        y_valid,
        scores
    )

    print(
        "\nAverage Precision (AP):",
        round(ap, 4)
    )

    # -------------------------------------------------
    # PART 6: SAVE
    # -------------------------------------------------

    print("\nSaving Experiment 2 model...")

    joblib.dump(
        vectorizer,
        MODEL_PATH / "combined_tfidf.joblib"
    )

    joblib.dump(
        scaler,
        MODEL_PATH / "url_feature_scaler.joblib"
    )

    joblib.dump(
        model,
        MODEL_PATH / "combined_linear_svm.joblib"
    )

    print("\nSaved:")
    print(
        "models/combined_tfidf.joblib"
    )
    print(
        "models/url_feature_scaler.joblib"
    )
    print(
        "models/combined_linear_svm.joblib"
    )


if __name__ == "__main__":
    main()