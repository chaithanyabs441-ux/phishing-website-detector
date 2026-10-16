from pathlib import Path
import sys
import gc
import joblib
import traceback
from datetime import datetime

import pandas as pd
import numpy as np

from scipy.sparse import hstack, csr_matrix

from sklearn.metrics import average_precision_score


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_FOLDER = PROJECT_ROOT / "data" / "raw" / "data"
MODEL_FOLDER = PROJECT_ROOT / "models"
LOG_FOLDER = PROJECT_ROOT / "logs"

LOG_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

LOG_FILE = (
    LOG_FOLDER /
    "bootstrap_exp2_exp3_log.txt"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

N_BOOTSTRAPS = 5000

CONFIDENCE_LEVEL = 0.95


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
        "PHISH//X - EXP2 vs EXP3 PAIRED BOOTSTRAP\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )


# ============================================================
# MAIN
# ============================================================

try:

    log("=" * 70)
    log("EXP2 vs EXP3 PAIRED BOOTSTRAP")
    log("=" * 70)

    log(
        f"Bootstrap samples: "
        f"{N_BOOTSTRAPS}"
    )

    log(
        f"Random seed: "
        f"{RANDOM_SEED}"
    )

    log(
        f"Project root: "
        f"{PROJECT_ROOT}"
    )

    log(
        f"Test folder: "
        f"{TEST_FOLDER}"
    )

    log(
        f"Model folder: "
        f"{MODEL_FOLDER}"
    )


    # ========================================================
    # STEP 1
    # LOAD URL FEATURE FUNCTION
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 1: Loading URL feature function")
    log("=" * 70)

    sys.path.append(
        str(
            PROJECT_ROOT /
            "src" /
            "features"
        )
    )

    from url_features import extract_url_features

    log(
        "URL feature function loaded successfully"
    )


    # ========================================================
    # STEP 2
    # LOAD EXPERIMENT 2 MODEL
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 2: Loading Experiment 2 model")
    log("=" * 70)

    # --------------------------------------------------------
    # IMPORTANT:
    # These filenames match the actual files in your
    # models folder.
    # --------------------------------------------------------

    vectorizer = joblib.load(
        MODEL_FOLDER /
        "combined_tfidf.joblib"
    )

    log(
        "Loaded Experiment 2 TF-IDF"
    )

    model = joblib.load(
        MODEL_FOLDER /
        "combined_linear_svm.joblib"
    )

    log(
        "Loaded Experiment 2 Linear SVM"
    )

    url_scaler = joblib.load(
        MODEL_FOLDER /
        "url_engineered_scaler.joblib"
    )

    log(
        "Loaded Experiment 2 URL scaler"
    )

    log(
        "All Experiment 2 model components "
        "loaded successfully"
    )


    # ========================================================
    # STEP 3
    # LOAD EXPERIMENT 3 RESULTS
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 3: Loading Experiment 3 test results")
    log("=" * 70)

    checkpoint_file = (
        LOG_FOLDER /
        "url_html_test_checkpoint.csv"
    )

    if not checkpoint_file.exists():

        raise FileNotFoundError(
            "Experiment 3 checkpoint was not found:\n"
            + str(checkpoint_file)
        )

    log(
        f"Checkpoint found: "
        f"{checkpoint_file}"
    )

    checkpoint = pd.read_csv(
        checkpoint_file
    )

    log(
        f"Checkpoint files: "
        f"{len(checkpoint)}"
    )


    # --------------------------------------------------------
    # CHECKPOINT COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "file",
        "rows",
        "accuracy",
        "precision",
        "recall",
        "y_test",
        "y_pred",
        "y_scores"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in checkpoint.columns
    ]

    if missing_columns:

        raise ValueError(
            "Checkpoint is missing columns: "
            + str(missing_columns)
        )

    log(
        "Checkpoint columns verified"
    )


    # ========================================================
    # EXTRACT EXP3 LABELS AND SCORES
    # ========================================================

    exp3_y_test = []

    exp3_scores = []


    for _, row in checkpoint.iterrows():

        y_values = [
            int(value)
            for value in str(
                row["y_test"]
            ).split(",")
        ]

        score_values = [
            float(value)
            for value in str(
                row["y_scores"]
            ).split(",")
        ]

        exp3_y_test.extend(
            y_values
        )

        exp3_scores.extend(
            score_values
        )


    exp3_y_test = np.array(
        exp3_y_test,
        dtype=int
    )

    exp3_scores = np.array(
        exp3_scores,
        dtype=float
    )


    log(
        f"Experiment 3 labels loaded: "
        f"{len(exp3_y_test)}"
    )

    log(
        f"Experiment 3 scores loaded: "
        f"{len(exp3_scores)}"
    )


    # ========================================================
    # STEP 4
    # VERIFY EXP3 ROW COUNT
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 4: Verifying Experiment 3 results")
    log("=" * 70)

    EXPECTED_ROWS = 168060


    if len(exp3_y_test) != EXPECTED_ROWS:

        raise ValueError(
            f"Expected {EXPECTED_ROWS} "
            f"Experiment 3 labels, but found "
            f"{len(exp3_y_test)}"
        )


    if len(exp3_scores) != EXPECTED_ROWS:

        raise ValueError(
            f"Expected {EXPECTED_ROWS} "
            f"Experiment 3 scores, but found "
            f"{len(exp3_scores)}"
        )


    log(
        "Experiment 3 row count: OK"
    )


    # ========================================================
    # STEP 5
    # LOAD OFFICIAL TEST URLS AND LABELS
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 5: Loading official test URLs")
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


    all_urls = []

    all_labels = []


    for number, file in enumerate(
        test_files,
        start=1
    ):

        log(
            f"[{number}/{len(test_files)}] "
            f"Loading {file.name}"
        )

        part = pd.read_parquet(
            file,
            columns=[
                "url",
                "label"
            ]
        )


        # ----------------------------------------------------
        # URLS
        # ----------------------------------------------------

        all_urls.extend(
            part["url"]
            .astype(str)
            .tolist()
        )


        # ----------------------------------------------------
        # LABELS
        # ----------------------------------------------------

        labels = (
            part["label"]
            .map({
                "benign": 0,
                "phish": 1
            })
        )


        if labels.isnull().any():

            raise ValueError(
                f"Unknown label found "
                f"in {file.name}"
            )


        all_labels.extend(
            labels
            .astype(int)
            .tolist()
        )


        del part

        gc.collect()


    test_urls = np.array(
        all_urls,
        dtype=object
    )

    y_test = np.array(
        all_labels,
        dtype=int
    )


    del all_urls
    del all_labels

    gc.collect()


    log(
        f"Official test URLs: "
        f"{len(test_urls)}"
    )

    log(
        f"Official test labels: "
        f"{len(y_test)}"
    )


    # ========================================================
    # STEP 6
    # VERIFY SAME TEST LABELS
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 6: Verifying test-set alignment")
    log("=" * 70)


    if len(y_test) != EXPECTED_ROWS:

        raise ValueError(
            f"Expected {EXPECTED_ROWS} "
            f"test labels, but found "
            f"{len(y_test)}"
        )


    if not np.array_equal(
        y_test,
        exp3_y_test
    ):

        raise ValueError(
            "Experiment 2 test labels and "
            "Experiment 3 test labels "
            "do not match."
        )


    log(
        "Test labels match Experiment 3 exactly."
    )

    log(
        "Test-set alignment: OK"
    )


    # ========================================================
    # STEP 7
    # EXPERIMENT 2 TF-IDF
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 7: Creating Experiment 2 TF-IDF features")
    log("=" * 70)

    X_tfidf = vectorizer.transform(
        test_urls
    )

    log(
        f"Experiment 2 TF-IDF shape: "
        f"{X_tfidf.shape}"
    )


    # ========================================================
    # STEP 8
    # EXPERIMENT 2 ENGINEERED URL FEATURES
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 8: Creating Experiment 2 engineered URL features")
    log("=" * 70)

    feature_rows = []


    for i, url in enumerate(
        test_urls
    ):

        if i % 10000 == 0:

            log(
                f"URL features processed: "
                f"{i}/{len(test_urls)}"
            )


        feature_rows.append(
            extract_url_features(
                url
            )
        )


    feature_df = pd.DataFrame(
        feature_rows
    )


    del feature_rows

    gc.collect()


    log(
        f"Engineered URL features shape: "
        f"{feature_df.shape}"
    )


    # ========================================================
    # SCALE ENGINEERED FEATURES
    # ========================================================

    X_engineered = (
        url_scaler.transform(
            feature_df
        )
    )

    X_engineered = csr_matrix(
        X_engineered
    )


    log(
        f"Scaled URL features shape: "
        f"{X_engineered.shape}"
    )


    # ========================================================
    # STEP 9
    # COMBINE EXP2 FEATURES
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 9: Combining Experiment 2 features")
    log("=" * 70)

    X_exp2 = hstack(
        [
            X_tfidf,
            X_engineered
        ]
    ).tocsr()


    log(
        f"Experiment 2 final matrix: "
        f"{X_exp2.shape}"
    )


    # ========================================================
    # STEP 10
    # GENERATE EXP2 DECISION SCORES
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 10: Generating Experiment 2 scores")
    log("=" * 70)

    exp2_scores = (
        model.decision_function(
            X_exp2
        )
    )


    log(
        f"Experiment 2 scores generated: "
        f"{len(exp2_scores)}"
    )


    # ========================================================
    # STEP 11
    # VERIFY SCORE COUNTS
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 11: Verifying score alignment")
    log("=" * 70)


    if len(exp2_scores) != EXPECTED_ROWS:

        raise ValueError(
            f"Expected {EXPECTED_ROWS} "
            f"Experiment 2 scores, but found "
            f"{len(exp2_scores)}"
        )


    if len(exp3_scores) != EXPECTED_ROWS:

        raise ValueError(
            f"Expected {EXPECTED_ROWS} "
            f"Experiment 3 scores, but found "
            f"{len(exp3_scores)}"
        )


    log(
        "Experiment 2 score count: OK"
    )

    log(
        "Experiment 3 score count: OK"
    )

    log(
        "Score alignment: OK"
    )


    # ========================================================
    # STEP 12
    # CALCULATE OBSERVED AP
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 12: Calculating observed AP")
    log("=" * 70)


    exp2_ap = average_precision_score(
        y_test,
        exp2_scores
    )


    exp3_ap = average_precision_score(
        y_test,
        exp3_scores
    )


    observed_difference = (
        exp3_ap -
        exp2_ap
    )


    log(
        f"Experiment 2 AP: "
        f"{exp2_ap:.6f}"
    )

    log(
        f"Experiment 3 AP: "
        f"{exp3_ap:.6f}"
    )

    log(
        f"Observed AP difference "
        f"(Exp3 - Exp2): "
        f"{observed_difference:.6f}"
    )


    # ========================================================
    # STEP 13
    # PAIRED BOOTSTRAP
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 13: Running paired bootstrap")
    log("=" * 70)


    rng = np.random.default_rng(
        RANDOM_SEED
    )


    n = len(y_test)


    bootstrap_differences = np.empty(
        N_BOOTSTRAPS,
        dtype=float
    )


    for i in range(
        N_BOOTSTRAPS
    ):

        if i % 500 == 0:

            log(
                f"Bootstrap iteration: "
                f"{i}/{N_BOOTSTRAPS}"
            )


        # ----------------------------------------------------
        # SAME SAMPLE FOR BOTH MODELS
        # ----------------------------------------------------

        indices = rng.integers(
            0,
            n,
            size=n
        )


        y_sample = (
            y_test[
                indices
            ]
        )


        exp2_sample = (
            exp2_scores[
                indices
            ]
        )


        exp3_sample = (
            exp3_scores[
                indices
            ]
        )


        # ----------------------------------------------------
        # AP
        # ----------------------------------------------------

        try:

            ap2 = (
                average_precision_score(
                    y_sample,
                    exp2_sample
                )
            )


            ap3 = (
                average_precision_score(
                    y_sample,
                    exp3_sample
                )
            )


            bootstrap_differences[i] = (
                ap3 - ap2
            )


        except ValueError:

            bootstrap_differences[i] = np.nan


    # --------------------------------------------------------
    # REMOVE INVALID BOOTSTRAP SAMPLES
    # --------------------------------------------------------

    bootstrap_differences = (
        bootstrap_differences[
            ~np.isnan(
                bootstrap_differences
            )
        ]
    )


    log(
        f"Valid bootstrap samples: "
        f"{len(bootstrap_differences)}"
    )


    # ========================================================
    # STEP 14
    # CONFIDENCE INTERVAL
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 14: Calculating 95% bootstrap confidence interval")
    log("=" * 70)


    alpha = (
        1 -
        CONFIDENCE_LEVEL
    )


    lower = np.percentile(
        bootstrap_differences,
        100 * alpha / 2
    )


    upper = np.percentile(
        bootstrap_differences,
        100 * (
            1 -
            alpha / 2
        )
    )


    # --------------------------------------------------------
    # BOOTSTRAP PROBABILITY
    # --------------------------------------------------------

    probability_exp3_better = (
        np.mean(
            bootstrap_differences > 0
        )
    )


    # ========================================================
    # STEP 15
    # FINAL RESULTS
    # ========================================================

    log("")
    log("=" * 70)
    log("EXP2 vs EXP3 — PAIRED BOOTSTRAP RESULTS")
    log("=" * 70)


    log(
        f"Test rows: "
        f"{n}"
    )


    log(
        f"Bootstrap samples: "
        f"{len(bootstrap_differences)}"
    )


    log("")


    log(
        f"Experiment 2 AP: "
        f"{exp2_ap:.6f}"
    )


    log(
        f"Experiment 3 AP: "
        f"{exp3_ap:.6f}"
    )


    log("")


    log(
        f"Observed AP difference: "
        f"{observed_difference:.6f}"
    )


    log(
        f"95% bootstrap CI: "
        f"[{lower:.6f}, {upper:.6f}]"
    )


    log("")


    log(
        f"Bootstrap probability "
        f"(Exp3 > Exp2): "
        f"{probability_exp3_better:.4f}"
    )


    # ========================================================
    # STEP 16
    # INTERPRETATION
    # ========================================================

    log("")
    log("=" * 70)
    log("INTERPRETATION")
    log("=" * 70)


    if lower > 0:

        log(
            "The 95% bootstrap confidence interval "
            "is entirely above zero."
        )

        log(
            "This supports a positive AP difference "
            "between Experiment 3 and Experiment 2 "
            "on this test set."
        )

    elif upper < 0:

        log(
            "The 95% bootstrap confidence interval "
            "is entirely below zero."
        )

        log(
            "Experiment 2 has higher AP than "
            "Experiment 3 on this test set."
        )

    else:

        log(
            "The 95% bootstrap confidence interval "
            "includes zero."
        )

        log(
            "The observed AP difference should "
            "not be described as statistically "
            "confirmed."
        )


    # ========================================================
    # STEP 17
    # SAVE BOOTSTRAP DISTRIBUTION
    # ========================================================

    bootstrap_file = (
        LOG_FOLDER /
        "exp2_exp3_bootstrap_differences.csv"
    )


    pd.DataFrame({
        "ap_difference":
        bootstrap_differences
    }).to_csv(
        bootstrap_file,
        index=False
    )


    log(
        f"Bootstrap distribution saved: "
        f"{bootstrap_file}"
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    log("")
    log("=" * 70)
    log("BOOTSTRAP ANALYSIS COMPLETE")
    log("=" * 70)


    log(
        f"Log saved to: "
        f"{LOG_FILE}"
    )


except Exception:

    log("")
    log("=" * 70)
    log("BOOTSTRAP ANALYSIS FAILED")
    log("=" * 70)

    log_error()

    log(
        f"Check the full error in: "
        f"{LOG_FILE}"
    )

    raise