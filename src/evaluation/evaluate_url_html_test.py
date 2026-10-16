from pathlib import Path
import sys
import gc
import joblib
import traceback
from datetime import datetime

import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from scipy.sparse import hstack, csr_matrix
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

TEST_FOLDER = PROJECT_ROOT / "data" / "raw" / "data"
MODEL_FOLDER = PROJECT_ROOT / "models"
LOG_FOLDER = PROJECT_ROOT / "logs"

LOG_FOLDER.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_FOLDER / "evaluate_url_html_test_log.txt"

CHECKPOINT_FILE = (
    LOG_FOLDER / "url_html_test_checkpoint.csv"
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 250


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
        file.write(traceback.format_exc())
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
        "PHISH//X - URL + HTML OFFICIAL TEST EVALUATION\n"
    )

    file.write("=" * 70 + "\n\n")


# ============================================================
# MAIN
# ============================================================

try:

    log("=" * 70)
    log("URL + HTML OFFICIAL TEST EVALUATION")
    log("=" * 70)

    log(f"Project root: {PROJECT_ROOT}")
    log(f"Test folder: {TEST_FOLDER}")
    log(f"Model folder: {MODEL_FOLDER}")
    log(f"Checkpoint: {CHECKPOINT_FILE}")
    log(f"Batch size: {BATCH_SIZE}")


    # ========================================================
    # STEP 1
    # LOAD FEATURE FUNCTIONS
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 1: Loading feature functions")
    log("=" * 70)

    sys.path.append(
        str(PROJECT_ROOT / "src" / "features")
    )

    from url_features import extract_url_features
    from html_features import extract_html_features

    log(
        "URL feature function loaded successfully"
    )

    log(
        "HTML feature function loaded successfully"
    )


    # ========================================================
    # HTML FEATURES
    # ========================================================

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


    # ========================================================
    # STEP 2
    # LOAD MODEL
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 2: Loading saved Experiment 3 model")
    log("=" * 70)

    vectorizer = joblib.load(
        MODEL_FOLDER /
        "url_html_tfidf.joblib"
    )

    log("Loaded URL TF-IDF")

    model = joblib.load(
        MODEL_FOLDER /
        "url_html_linear_svm.joblib"
    )

    log("Loaded Linear SVM")

    url_scaler = joblib.load(
        MODEL_FOLDER /
        "url_engineered_scaler.joblib"
    )

    log("Loaded URL scaler")

    html_scaler = joblib.load(
        MODEL_FOLDER /
        "html_scaler.joblib"
    )

    log("Loaded HTML scaler")


    # ========================================================
    # LOAD SAVED HTML COLUMN ORDER
    # ========================================================

    html_columns_file = (
        MODEL_FOLDER /
        "html_feature_columns.joblib"
    )

    if html_columns_file.exists():

        saved_html_columns = joblib.load(
            html_columns_file
        )

        log(
            f"Loaded saved HTML columns: "
            f"{len(saved_html_columns)}"
        )

    else:

        saved_html_columns = HTML_COLUMNS

        log(
            "Saved HTML column file not found."
        )

        log(
            "Using predefined HTML columns."
        )


    log(
        "All model components loaded successfully"
    )


    # ========================================================
    # STEP 3
    # FIND TEST FILES
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 3: Finding official test files")
    log("=" * 70)

    test_files = sorted(
        TEST_FOLDER.glob(
            "test-*.parquet"
        )
    )

    log(
        f"Test files found: "
        f"{len(test_files)}"
    )

    if len(test_files) != 21:

        log(
            "WARNING: Expected 21 test files."
        )


    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    completed_files = set()

    checkpoint_rows = []

    if CHECKPOINT_FILE.exists():

        checkpoint = pd.read_csv(
            CHECKPOINT_FILE
        )

        completed_files = set(
            checkpoint["file"].astype(str)
        )

        checkpoint_rows = (
            checkpoint.to_dict("records")
        )

        log(
            f"Checkpoint found."
        )

        log(
            f"Completed files: "
            f"{len(completed_files)}"
        )

    else:

        log(
            "No previous checkpoint found."
        )

        log(
            "Starting from test-000."
        )


    # ========================================================
    # PROCESS FILES
    # ========================================================

    for file_number, file in enumerate(
        test_files,
        start=1
    ):

        # ----------------------------------------------------
        # SKIP COMPLETED FILE
        # ----------------------------------------------------

        if file.name in completed_files:

            log("")
            log(
                f"SKIPPING [{file_number}/"
                f"{len(test_files)}]: "
                f"{file.name}"
            )

            continue


        log("")
        log("=" * 70)
        log(
            f"PROCESSING TEST FILE "
            f"[{file_number}/"
            f"{len(test_files)}]: "
            f"{file.name}"
        )
        log("=" * 70)


        # ====================================================
        # OPEN PARQUET FILE
        # ====================================================

        log("")
        log(
            "Opening Parquet file..."
        )

        parquet_file = pq.ParquetFile(
            file
        )

        total_file_rows = (
            parquet_file.metadata.num_rows
        )

        log(
            f"Rows in file: "
            f"{total_file_rows}"
        )


        file_y_test = []
        file_y_pred = []
        file_y_scores = []


        # ====================================================
        # PROCESS BATCHES
        # ====================================================

        batch_number = 0

        for batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=[
                "url",
                "label",
                "html"
            ]
        ):

            batch_number += 1

            log("")
            log(
                f"{file.name} | "
                f"Processing batch "
                f"{batch_number}"
            )


            # ------------------------------------------------
            # CONVERT ONLY SMALL BATCH TO PANDAS
            # ------------------------------------------------

            test_data = (
                batch.to_pandas()
            )

            rows = len(test_data)

            log(
                f"Batch rows: {rows}"
            )


            # =================================================
            # LABELS
            # =================================================

            y_test = (
                test_data["label"]
                .map({
                    "benign": 0,
                    "phish": 1
                })
            )

            if y_test.isnull().any():

                raise ValueError(
                    f"Unknown labels found "
                    f"in {file.name}"
                )

            y_test = (
                y_test
                .astype(int)
                .values
            )


            # =================================================
            # URLS
            # =================================================

            test_urls = (
                test_data["url"]
                .astype(str)
            )


            # =================================================
            # URL TF-IDF
            # =================================================

            X_url_tfidf = (
                vectorizer.transform(
                    test_urls
                )
            )

            log(
                f"URL TF-IDF: "
                f"{X_url_tfidf.shape}"
            )


            # =================================================
            # ENGINEERED URL FEATURES
            # =================================================

            url_feature_rows = []

            for url in test_urls:

                features = (
                    extract_url_features(
                        url
                    )
                )

                url_feature_rows.append(
                    features
                )


            url_features_df = (
                pd.DataFrame(
                    url_feature_rows
                )
            )


            X_url_engineered = (
                url_scaler.transform(
                    url_features_df
                )
            )

            X_url_engineered = csr_matrix(
                X_url_engineered
            )


            # =================================================
            # HTML FEATURES
            # =================================================

            html_feature_rows = []

            for i, row in enumerate(
                test_data.itertuples(
                    index=False
                )
            ):

                if i % 100 == 0:

                    log(
                        f"{file.name} | "
                        f"HTML features: "
                        f"{i}/{rows}"
                    )


                features = (
                    extract_html_features(
                        row.html,
                        row.url
                    )
                )

                html_feature_rows.append(
                    features
                )


            html_features_df = (
                pd.DataFrame(
                    html_feature_rows
                )
            )


            # =================================================
            # CHECK HTML COLUMNS
            # =================================================

            missing_columns = [
                column
                for column in saved_html_columns
                if column not in
                html_features_df.columns
            ]

            if missing_columns:

                raise ValueError(
                    "Missing HTML feature columns: "
                    + str(missing_columns)
                )


            # =================================================
            # SCALE HTML FEATURES
            # =================================================

            X_html_values = (
                html_features_df[
                    saved_html_columns
                ].astype(float)
            )

            X_html = (
                html_scaler.transform(
                    X_html_values
                )
            )

            X_html = csr_matrix(
                X_html
            )


            # =================================================
            # COMBINE FEATURES
            # =================================================

            X_test = hstack(
                [
                    X_url_tfidf,
                    X_url_engineered,
                    X_html
                ]
            ).tocsr()

            log(
                f"Final batch matrix: "
                f"{X_test.shape}"
            )


            # =================================================
            # PREDICTIONS
            # =================================================

            y_pred = model.predict(
                X_test
            )

            y_scores = (
                model.decision_function(
                    X_test
                )
            )


            # =================================================
            # STORE BATCH RESULTS
            # =================================================

            file_y_test.extend(
                y_test.tolist()
            )

            file_y_pred.extend(
                y_pred.tolist()
            )

            file_y_scores.extend(
                y_scores.tolist()
            )


            # =================================================
            # CLEAR BATCH MEMORY
            # =================================================

            del batch
            del test_data
            del test_urls
            del y_test
            del X_url_tfidf
            del url_feature_rows
            del url_features_df
            del X_url_engineered
            del html_feature_rows
            del html_features_df
            del X_html_values
            del X_html
            del X_test
            del y_pred
            del y_scores

            gc.collect()


        # ====================================================
        # FILE FINISHED
        # ====================================================

        file_accuracy = accuracy_score(
            file_y_test,
            file_y_pred
        )

        file_precision = precision_score(
            file_y_test,
            file_y_pred
        )

        file_recall = recall_score(
            file_y_test,
            file_y_pred
        )


        log("")
        log(
            f"{file.name} COMPLETE"
        )

        log(
            f"Rows: "
            f"{len(file_y_test)}"
        )

        log(
            f"Accuracy: "
            f"{file_accuracy:.4f}"
        )

        log(
            f"Precision: "
            f"{file_precision:.4f}"
        )

        log(
            f"Recall: "
            f"{file_recall:.4f}"
        )


        # ====================================================
        # SAVE CHECKPOINT
        # ====================================================

        checkpoint_rows.append({

            "file":
            file.name,

            "rows":
            len(file_y_test),

            "accuracy":
            file_accuracy,

            "precision":
            file_precision,

            "recall":
            file_recall,

            "y_test":
            ",".join(
                map(
                    str,
                    file_y_test
                )
            ),

            "y_pred":
            ",".join(
                map(
                    str,
                    file_y_pred
                )
            ),

            "y_scores":
            ",".join(
                map(
                    str,
                    file_y_scores
                )
            )
        })


        checkpoint_df = pd.DataFrame(
            checkpoint_rows
        )

        checkpoint_df.to_csv(
            CHECKPOINT_FILE,
            index=False
        )


        log(
            f"Checkpoint saved: "
            f"{CHECKPOINT_FILE}"
        )


        # ====================================================
        # CLEAR FILE MEMORY
        # ====================================================

        del file_y_test
        del file_y_pred
        del file_y_scores

        gc.collect()


    # ========================================================
    # LOAD ALL CHECKPOINT RESULTS
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 12: Combining checkpoint results")
    log("=" * 70)


    checkpoint = pd.read_csv(
        CHECKPOINT_FILE
    )


    all_y_test = []
    all_y_pred = []
    all_y_scores = []


    for _, row in checkpoint.iterrows():

        y_test_values = [
            int(x)
            for x in str(
                row["y_test"]
            ).split(",")
        ]

        y_pred_values = [
            int(x)
            for x in str(
                row["y_pred"]
            ).split(",")
        ]

        y_score_values = [
            float(x)
            for x in str(
                row["y_scores"]
            ).split(",")
        ]


        all_y_test.extend(
            y_test_values
        )

        all_y_pred.extend(
            y_pred_values
        )

        all_y_scores.extend(
            y_score_values
        )


    y_test_final = np.array(
        all_y_test
    )

    y_pred_final = np.array(
        all_y_pred
    )

    y_scores_final = np.array(
        all_y_scores
    )


    # ========================================================
    # ROW COUNT
    # ========================================================

    log(
        f"Total evaluated rows: "
        f"{len(y_test_final)}"
    )


    EXPECTED_TEST_ROWS = 168060

    if len(y_test_final) != EXPECTED_TEST_ROWS:

        raise ValueError(
            f"Expected "
            f"{EXPECTED_TEST_ROWS} "
            f"rows but evaluated "
            f"{len(y_test_final)}."
        )

    log(
        "TEST ROW COUNT CHECK: OK"
    )


    # ========================================================
    # FINAL METRICS
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 13: Calculating official test metrics")
    log("=" * 70)


    accuracy = accuracy_score(
        y_test_final,
        y_pred_final
    )

    precision = precision_score(
        y_test_final,
        y_pred_final
    )

    recall = recall_score(
        y_test_final,
        y_pred_final
    )

    average_precision = (
        average_precision_score(
            y_test_final,
            y_scores_final
        )
    )

    cm = confusion_matrix(
        y_test_final,
        y_pred_final
    )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    log("")
    log("=" * 70)
    log("URL + HTML — OFFICIAL TEST RESULTS")
    log("=" * 70)

    log(
        f"Accuracy          : "
        f"{accuracy:.4f}"
    )

    log(
        f"Precision         : "
        f"{precision:.4f}"
    )

    log(
        f"Recall            : "
        f"{recall:.4f}"
    )

    log(
        f"Average Precision : "
        f"{average_precision:.4f}"
    )

    log("")

    log("Confusion Matrix:")

    log(
        str(cm)
    )


    tn, fp, fn, tp = cm.ravel()


    log("")

    log(
        f"True Negatives  : "
        f"{tn}"
    )

    log(
        f"False Positives : "
        f"{fp}"
    )

    log(
        f"False Negatives : "
        f"{fn}"
    )

    log(
        f"True Positives  : "
        f"{tp}"
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    log("")
    log("=" * 70)
    log("OFFICIAL TEST EVALUATION COMPLETE")
    log("=" * 70)

    log(
        f"Files evaluated: "
        f"{len(test_files)}"
    )

    log(
        f"Rows evaluated: "
        f"{len(y_test_final)}"
    )

    log(
        f"Checkpoint saved: "
        f"{CHECKPOINT_FILE}"
    )

    log(
        f"Log saved to: "
        f"{LOG_FILE}"
    )


except Exception:

    log("")
    log("=" * 70)
    log("OFFICIAL TEST EVALUATION FAILED")
    log("=" * 70)

    log_error()

    log(
        f"Check the full error in: "
        f"{LOG_FILE}"
    )

    raise