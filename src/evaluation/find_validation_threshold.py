import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import joblib


DATA_PATH = Path("data/raw/data")
MODEL_PATH = Path("models")


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

    data = load_training_data()

    X = data["url"].astype(str)
    y = data["label"]

    # Same validation split used during Experiment 2
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nValidation rows:", len(X_valid))

    # -------------------------------------------------
    # LOAD EXPERIMENT 2 MODEL
    # -------------------------------------------------

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

    # -------------------------------------------------
    # CREATE TF-IDF FEATURES
    # -------------------------------------------------

    print("\nCreating TF-IDF features...")

    X_valid_tfidf = vectorizer.transform(
        X_valid
    )

    # -------------------------------------------------
    # CREATE URL FEATURES
    # -------------------------------------------------

    print("Extracting URL features...")

    import sys

    sys.path.append(
        str(
            Path(__file__).resolve().parents[1]
            / "features"
        )
    )

    from url_features import extract_url_features

    valid_features = pd.DataFrame(
        [
            extract_url_features(url)
            for url in X_valid
        ]
    )

    valid_features_scaled = scaler.transform(
        valid_features
    )

    from scipy.sparse import csr_matrix, hstack

    valid_features_sparse = csr_matrix(
        valid_features_scaled
    )

    # -------------------------------------------------
    # COMBINE FEATURES
    # -------------------------------------------------

    X_valid_combined = hstack([
        X_valid_tfidf,
        valid_features_sparse
    ])

    print(
        "Combined validation shape:",
        X_valid_combined.shape
    )

    # -------------------------------------------------
    # GET DECISION SCORES
    # -------------------------------------------------

    print("\nGetting model scores...")

    scores = model.decision_function(
        X_valid_combined
    )

    y_valid = y_valid.values

    # -------------------------------------------------
    # SEARCH THRESHOLDS
    # -------------------------------------------------

    # Fine-grained thresholds
    thresholds = np.arange(
        -1.00,
        1.01,
        0.01
    )

    results = []

    for threshold in thresholds:

        predictions = (
            scores >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_valid,
            predictions
        ).ravel()

        fpr = fp / (fp + tn)

        recall = tp / (tp + fn)

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0
        )

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

    results_df = pd.DataFrame(results)

    # -------------------------------------------------
    # APPLY OUR CONSTRAINT
    # -------------------------------------------------

    acceptable = results_df[
        results_df["fpr"] <= 0.05
    ]

    if len(acceptable) == 0:

        print(
            "\nNo threshold satisfies FPR <= 5%."
        )

        return

    # Maximum recall while FPR <= 5%
    best = acceptable.sort_values(
        ["recall", "precision"],
        ascending=[False, False]
    ).iloc[0]

    # -------------------------------------------------
    # PRINT RESULT
    # -------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION THRESHOLD SEARCH")
    print("=" * 70)

    print("\nConstraint:")
    print("FPR <= 5%")

    print("\nSelected threshold:")
    print(
        round(best["threshold"], 2)
    )

    print("\nValidation results at selected threshold:")

    print(
        "Precision:",
        round(best["precision"], 4)
    )

    print(
        "Recall:",
        round(best["recall"], 4)
    )

    print(
        "FPR:",
        round(best["fpr"], 4)
    )

    print(
        "False positives:",
        int(best["false_positives"])
    )

    print(
        "False negatives:",
        int(best["false_negatives"])
    )

    print(
        "True positives:",
        int(best["true_positives"])
    )

    print(
        "True negatives:",
        int(best["true_negatives"])
    )

    # -------------------------------------------------
    # SHOW NEARBY THRESHOLDS
    # -------------------------------------------------

    print("\n" + "=" * 70)
    print("NEARBY THRESHOLDS")
    print("=" * 70)

    nearby = results_df[
        (results_df["threshold"] >= best["threshold"] - 0.05) &
        (results_df["threshold"] <= best["threshold"] + 0.05)
    ]

    print(
        nearby[
            [
                "threshold",
                "precision",
                "recall",
                "fpr",
                "false_positives",
                "false_negatives"
            ]
        ].to_string(index=False)
    )

    # -------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------

    results_df.to_csv(
        MODEL_PATH / "validation_threshold_results.csv",
        index=False
    )

    with open(
        MODEL_PATH / "selected_threshold.txt",
        "w"
    ) as file:

        file.write(
            str(round(best["threshold"], 2))
        )

    print("\nSaved:")
    print(
        "models/validation_threshold_results.csv"
    )
    print(
        "models/selected_threshold.txt"
    )


if __name__ == "__main__":
    main()