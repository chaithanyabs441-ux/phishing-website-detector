from pathlib import Path
import pandas as pd
import traceback
from datetime import datetime


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FOLDER = PROJECT_ROOT / "data"

LOG_FOLDER = PROJECT_ROOT / "logs"
LOG_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

LOG_FILE = (
    LOG_FOLDER /
    "benchmark_inspection_log.txt"
)


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


# ============================================================
# START
# ============================================================

with open(
    LOG_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "PHISH//X - PHRESH PHISH BENCHMARK INSPECTION\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )


try:

    log("=" * 70)
    log("PHRESH PHISH BENCHMARK INSPECTION")
    log("=" * 70)

    log(
        f"Project root: {PROJECT_ROOT}"
    )

    log(
        f"Data folder: {DATA_FOLDER}"
    )


    # ========================================================
    # FIND DATA FILES
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 1: Searching data folder")
    log("=" * 70)


    extensions = {
        ".parquet",
        ".csv",
        ".json",
        ".jsonl"
    }


    files = []

    for file in DATA_FOLDER.rglob("*"):

        if not file.is_file():
            continue

        if file.suffix.lower() in extensions:

            files.append(file)


    files = sorted(files)


    log(
        f"Data files found: "
        f"{len(files)}"
    )


    for file in files:

        relative = file.relative_to(
            PROJECT_ROOT
        )

        size_mb = (
            file.stat().st_size /
            (1024 * 1024)
        )

        log(
            f"{relative} | "
            f"{size_mb:.2f} MB"
        )


    # ========================================================
    # INSPECT PARQUET FILES
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 2: Inspecting Parquet files")
    log("=" * 70)


    parquet_files = [
        file
        for file in files
        if file.suffix.lower() == ".parquet"
    ]


    for file in parquet_files:

        log("")
        log(
            f"FILE: "
            f"{file.relative_to(PROJECT_ROOT)}"
        )

        try:

            df = pd.read_parquet(
                file,
                engine="pyarrow"
            )

            log(
                f"Rows: "
                f"{len(df)}"
            )

            log(
                f"Columns: "
                f"{list(df.columns)}"
            )


            # ------------------------------------------------
            # LABEL INFORMATION
            # ------------------------------------------------

            if "label" in df.columns:

                counts = (
                    df["label"]
                    .value_counts()
                )

                log(
                    f"Labels:\n"
                    f"{counts.to_string()}"
                )


                total = len(df)

                if total > 0:

                    for label, count in (
                        counts.items()
                    ):

                        percentage = (
                            count /
                            total *
                            100
                        )

                        log(
                            f"  {label}: "
                            f"{percentage:.6f}%"
                        )


            # ------------------------------------------------
            # TARGET INFORMATION
            # ------------------------------------------------

            if "target" in df.columns:

                log(
                    f"Target values: "
                    f"{df['target'].nunique(dropna=True)}"
                )


            del df


        except Exception:

            log(
                "Could not inspect this file."
            )

            log(
                traceback.format_exc()
            )


    # ========================================================
    # INSPECT CSV FILES
    # ========================================================

    log("")
    log("=" * 70)
    log("STEP 3: Inspecting CSV files")
    log("=" * 70)


    csv_files = [
        file
        for file in files
        if file.suffix.lower() == ".csv"
    ]


    for file in csv_files:

        log("")

        log(
            f"FILE: "
            f"{file.relative_to(PROJECT_ROOT)}"
        )

        try:

            df = pd.read_csv(
                file,
                nrows=1000
            )

            log(
                f"Columns: "
                f"{list(df.columns)}"
            )

            log(
                f"First 1000 rows loaded successfully"
            )

            if "label" in df.columns:

                log(
                    "Sample labels:"
                )

                log(
                    str(
                        df["label"]
                        .value_counts()
                    )
                )


        except Exception:

            log(
                "Could not inspect this CSV."
            )

            log(
                traceback.format_exc()
            )


    # ========================================================
    # FINISHED
    # ========================================================

    log("")
    log("=" * 70)
    log("BENCHMARK INSPECTION COMPLETE")
    log("=" * 70)

    log(
        f"Log saved to: "
        f"{LOG_FILE}"
    )


except Exception:

    log("")
    log("=" * 70)
    log("BENCHMARK INSPECTION FAILED")
    log("=" * 70)

    log(
        traceback.format_exc()
    )

    raise