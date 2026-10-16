from pathlib import Path
import sys
import gc
import joblib
import traceback
from datetime import datetime

import pandas as pd
import numpy as np

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HTML_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "html_features"
)

MODEL_FOLDER = PROJECT_ROOT / "models"

LOG_FOLDER = PROJECT_ROOT / "logs"

MODEL_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

LOG_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

LOG_FILE = LOG_FOLDER / "train_url_html_log.txt"


# ============================================================
# LOGGING
# ============================================================

def log(message=""):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    line = f"[{timestamp}] {message}"

    print(line)

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(line + "\n")


def log_error():

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write("\n")
        file.write("=" * 70 + "\n")
        file.write("FULL ERROR TRACEBACK\n")
        file.write("=" * 70 + "\n")

        file.write(
            traceback.format_exc()
        )

        file.write("\n")
        file.write("=" * 70 + "\n")


# ============================================================
# START LOG
# ============================================================

with open(
    LOG_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "PHISH//X - EXPERIMENT 3 LOG\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )


try:

    log("=" * 70)
    log("EXPERIMENT 3: URL + HTML PHISHING DETECTION")
    log("=" * 70)

    log(f"Project root: {PROJECT_ROOT}")
    log(f"HTML folder: {HTML_FOLDER}")
    log(f"Model folder: {MODEL_FOLDER}")
    log(f"Log file: {LOG_FILE}")


    # ========================================================
    # IMPORT URL FEATURE FUNCTION
    # ========================================================

    log("")
    log("STEP 0: Loading URL feature function")

    sys.path.append(
        str(PROJECT_ROOT / "src" / "features")
    )

    from url_features import extract_url_features

    log("URL feature function loaded successfully")


    # ========================================================
    # SETTINGS
    # ========================================================

    RANDOM_STATE = 42

    HTML_COLUMNS = [
        "html_length",
        "num_forms",
        "num_inputs",
        "num_links",
        "num_images",
        "num_scripts",
        "num_iframes",
        "num_password_inputs",
        "num_email_inputs",
        "num_hidden_inputs",
        "num_external_links",
        "num_external_form_actions",
        "has_password_field",
        "has_form",
        "has_external_form_action"
    ]

    EXPECTED_ROWS = 498255


    # ========================================================
    # LOAD HTML FEATURE FILES
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 1: Loading HTML feature files")
    log("=" * 70)

    html_files = sorted(
        HTML_FOLDER.glob(
            "train-*_html_features.csv"
        )
    )

    log(
        f"HTML feature files found: {len(html_files)}"
    )

    if len(html_files) != 56:

        log(
            "WARNING: Expected 56 HTML feature files."
        )


    all_data = []

    for number, file in enumerate(
        html_files,
        start=1
    ):

        log(
            f"[{number}/{len(html_files)}] "
            f"Loading {file.name}"
        )

        df = pd.read_csv(file)

        log(
            f"{file.name} rows: {len(df)}"
        )

        all_data.append(df)


    log("")
    log("Combining HTML feature files...")

    data = pd.concat(
        all_data,
        ignore_index=True
    )

    del all_data
    gc.collect()

    log(
        f"Combined dataset rows: {len(data)}"
    )

    log(
        f"Combined dataset columns: {len(data.columns)}"
    )


    # ========================================================
    # DATASET SIZE CHECK
    # ========================================================

    log("")
    log("Checking dataset row count...")

    if len(data) != EXPECTED_ROWS:

        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"but found {len(data)} rows."
        )

    log("ROW COUNT CHECK: OK")


    # ========================================================
    # COLUMN CHECK
    # ========================================================

    log("")
    log("Checking required columns...")

    required_columns = [
        "url",
        "label"
    ] + HTML_COLUMNS

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns: "
            + str(missing_columns)
        )

    log("COLUMN CHECK: OK")


    # ========================================================
    # LABEL PREPARATION
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 2: Preparing labels")
    log("=" * 70)

    log("Original labels:")

    log(
        str(data["label"].value_counts())
    )

    data["label"] = data["label"].map(
        {
            "benign": 0,
            "phish": 1
        }
    )

    if data["label"].isnull().any():

        raise ValueError(
            "Some labels could not be converted."
        )

    y = data["label"].astype(int).values

    log("Labels converted successfully")
    log("benign = 0")
    log("phish  = 1")


    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 3: Creating train/validation split")
    log("=" * 70)

    indices = np.arange(
        len(data)
    )

    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    log(
        f"Training rows  : {len(train_indices)}"
    )

    log(
        f"Validation rows: {len(val_indices)}"
    )


    # ========================================================
    # URL DATA
    # ========================================================

    log("")
    log("STEP 4: Preparing URL data")

    urls = data["url"].astype(str)

    train_urls = urls.iloc[
        train_indices
    ]

    val_urls = urls.iloc[
        val_indices
    ]

    log("URL data prepared")


    # ========================================================
    # URL ENGINEERED FEATURES
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 5: Extracting engineered URL features")
    log("=" * 70)

    url_feature_rows = []

    total_rows = len(urls)

    for i, url in enumerate(urls):

        if i % 10000 == 0:

            log(
                f"URL features processed: "
                f"{i}/{total_rows}"
            )

        features = extract_url_features(url)

        url_feature_rows.append(
            features
        )


    url_features_df = pd.DataFrame(
        url_feature_rows
    )

    del url_feature_rows
    gc.collect()

    log(
        f"URL engineered feature shape: "
        f"{url_features_df.shape}"
    )


    # ========================================================
    # SPLIT ENGINEERED FEATURES
    # ========================================================

    X_url_engineered_train_values = (
        url_features_df.iloc[
            train_indices
        ]
    )

    X_url_engineered_val_values = (
        url_features_df.iloc[
            val_indices
        ]
    )


    # ========================================================
    # SCALE URL ENGINEERED FEATURES
    # ========================================================

    log("")
    log("STEP 6: Scaling URL engineered features")

    url_scaler = StandardScaler()

    X_url_engineered_train = (
        url_scaler.fit_transform(
            X_url_engineered_train_values
        )
    )

    X_url_engineered_val = (
        url_scaler.transform(
            X_url_engineered_val_values
        )
    )

    X_url_engineered_train = csr_matrix(
        X_url_engineered_train
    )

    X_url_engineered_val = csr_matrix(
        X_url_engineered_val
    )

    log(
        f"URL engineered train shape: "
        f"{X_url_engineered_train.shape}"
    )

    log(
        f"URL engineered validation shape: "
        f"{X_url_engineered_val.shape}"
    )


    # ========================================================
    # URL TF-IDF
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 7: Creating URL character TF-IDF")
    log("=" * 70)

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_features=200000,
        sublinear_tf=True
    )

    log("Fitting TF-IDF on training URLs only...")

    X_url_tfidf_train = (
        vectorizer.fit_transform(
            train_urls
        )
    )

    log(
        f"Training TF-IDF shape: "
        f"{X_url_tfidf_train.shape}"
    )

    log("Transforming validation URLs...")

    X_url_tfidf_val = (
        vectorizer.transform(
            val_urls
        )
    )

    log(
        f"Validation TF-IDF shape: "
        f"{X_url_tfidf_val.shape}"
    )


    # ========================================================
    # HTML FEATURES
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 8: Preparing HTML structural features")
    log("=" * 70)

    X_html_values = data[
        HTML_COLUMNS
    ].astype(float)

    X_html_train_values = (
        X_html_values.iloc[
            train_indices
        ]
    )

    X_html_val_values = (
        X_html_values.iloc[
            val_indices
        ]
    )

    log("Fitting HTML scaler on training data only...")

    html_scaler = StandardScaler()

    X_html_train = (
        html_scaler.fit_transform(
            X_html_train_values
        )
    )

    X_html_val = (
        html_scaler.transform(
            X_html_val_values
        )
    )

    X_html_train = csr_matrix(
        X_html_train
    )

    X_html_val = csr_matrix(
        X_html_val
    )

    log(
        f"HTML training shape: "
        f"{X_html_train.shape}"
    )

    log(
        f"HTML validation shape: "
        f"{X_html_val.shape}"
    )


    # ========================================================
    # COMBINE FEATURES
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 9: Combining URL + engineered URL + HTML")
    log("=" * 70)

    X_train = hstack(
        [
            X_url_tfidf_train,
            X_url_engineered_train,
            X_html_train
        ]
    ).tocsr()

    X_val = hstack(
        [
            X_url_tfidf_val,
            X_url_engineered_val,
            X_html_val
        ]
    ).tocsr()

    log(
        f"Training feature matrix: "
        f"{X_train.shape}"
    )

    log(
        f"Validation feature matrix: "
        f"{X_val.shape}"
    )


    # ========================================================
    # LABEL SPLIT
    # ========================================================

    y_train = y[
        train_indices
    ]

    y_val = y[
        val_indices
    ]


    # ========================================================
    # TRAIN MODEL
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 10: TRAINING LINEAR SVM")
    log("=" * 70)

    model = LinearSVC(
        C=1.0,
        class_weight="balanced",
        dual=True,
        max_iter=5000
    )

    log("Starting model training...")

    model.fit(
        X_train,
        y_train
    )

    log("MODEL TRAINING COMPLETE")


    # ========================================================
    # VALIDATION PREDICTIONS
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 11: Evaluating validation set")
    log("=" * 70)

    y_pred = model.predict(
        X_val
    )

    y_scores = model.decision_function(
        X_val
    )

    log("Predictions generated")


    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    precision = precision_score(
        y_val,
        y_pred
    )

    recall = recall_score(
        y_val,
        y_pred
    )

    average_precision = (
        average_precision_score(
            y_val,
            y_scores
        )
    )

    cm = confusion_matrix(
        y_val,
        y_pred
    )


    # ========================================================
    # RESULTS
    # ========================================================

    log("")
    log("=" * 70)
    log("EXPERIMENT 3 RESULTS")
    log("=" * 70)

    log(
        f"Accuracy          : {accuracy:.4f}"
    )

    log(
        f"Precision         : {precision:.4f}"
    )

    log(
        f"Recall            : {recall:.4f}"
    )

    log(
        f"Average Precision : {average_precision:.4f}"
    )

    log("")
    log("Confusion Matrix:")
    log(str(cm))

    log("")
    log(
        "Matrix format:"
    )

    log(
        "[[True Negative, False Positive],"
    )

    log(
        " [False Negative, True Positive]]"
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 12: SAVING MODEL")
    log("=" * 70)

    joblib.dump(
        vectorizer,
        MODEL_FOLDER / "url_html_tfidf.joblib"
    )

    log("Saved URL TF-IDF")

    joblib.dump(
        model,
        MODEL_FOLDER / "url_html_linear_svm.joblib"
    )

    log("Saved Linear SVM")

    joblib.dump(
        url_scaler,
        MODEL_FOLDER / "url_engineered_scaler.joblib"
    )

    log("Saved URL scaler")

    joblib.dump(
        html_scaler,
        MODEL_FOLDER / "html_scaler.joblib"
    )

    log("Saved HTML scaler")

    joblib.dump(
        HTML_COLUMNS,
        MODEL_FOLDER / "html_feature_columns.joblib"
    )

    log("Saved HTML feature columns")


    # ========================================================
    # FINISH
    # ========================================================

    log("")
    log("=" * 70)
    log("EXPERIMENT 3 COMPLETE")
    log("=" * 70)

    log(
        f"Log saved to: {LOG_FILE}"
    )


except Exception:

    log("")
    log("=" * 70)
    log("EXPERIMENT 3 FAILED")
    log("=" * 70)

    log_error()

    log(
        f"Check the full error in: {LOG_FILE}"
    )

    raise