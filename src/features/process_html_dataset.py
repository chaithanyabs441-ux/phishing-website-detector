from pathlib import Path
import sys
import gc
import traceback
from datetime import datetime

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from html_features import extract_html_features


input_folder = Path("data/raw/data")
output_folder = Path("data/processed/html_features")
output_folder.mkdir(parents=True, exist_ok=True)

log_file = Path("html_log.txt")


def log(message):
    message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"

    print(message)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")


log("=" * 70)
log("HTML FEATURE EXTRACTION STARTED")
log("=" * 70)

files = sorted(input_folder.glob("train-*.parquet"))

log(f"TRAIN FILES FOUND: {len(files)}")

for file_number, file in enumerate(files, start=1):

    output_file = output_folder / (file.stem + "_html_features.csv")

    if output_file.exists():
        log(f"[{file_number}/{len(files)}] ALREADY EXISTS: {file.name}")
        continue

    log("=" * 70)
    log(f"STARTING FILE [{file_number}/{len(files)}]: {file.name}")

    try:
        log(f"Reading parquet: {file}")

        df = pd.read_parquet(
            file,
            columns=["url", "label", "html"]
        )

        log(f"Rows loaded: {len(df)}")

        features = []

        for i, row in df.iterrows():

            if i % 100 == 0:
                log(
                    f"{file.name} | "
                    f"HTML processed: {i}/{len(df)}"
                )

            html = row["html"]
            url = row["url"]

            try:
                html_features = extract_html_features(
                    html,
                    url
                )

                html_features["url"] = url
                html_features["label"] = row["label"]

                features.append(html_features)

            except Exception as e:

                log(f"FAILED ROW: {i}")
                log(f"URL: {url}")
                log(f"ERROR: {repr(e)}")

                with open(log_file, "a", encoding="utf-8") as f:
                    traceback.print_exc(file=f)

        log(
            f"{file.name} | "
            f"Feature extraction finished: "
            f"{len(features)}/{len(df)} rows"
        )

        feature_df = pd.DataFrame(features)

        log(
            f"{file.name} | "
            f"Creating CSV with {len(feature_df)} rows"
        )

        feature_df.to_csv(
            output_file,
            index=False
        )

        if output_file.exists():

            log(
                f"SAVED: {output_file} | "
                f"SIZE: {output_file.stat().st_size} bytes"
            )

        else:

            log(
                f"ERROR: CSV WAS NOT CREATED: "
                f"{output_file}"
            )

        del df
        del features
        del feature_df

        gc.collect()

        log(
            f"FINISHED FILE [{file_number}/{len(files)}]: "
            f"{file.name}"
        )

    except Exception as e:

        log("=" * 70)
        log(f"FILE FAILED: {file.name}")
        log(f"ERROR: {repr(e)}")
        log("FULL TRACEBACK:")

        with open(log_file, "a", encoding="utf-8") as f:
            traceback.print_exc(file=f)

        log(f"SKIPPING FILE: {file.name}")

        gc.collect()


log("=" * 70)
log("HTML FEATURE EXTRACTION COMPLETE")
log("=" * 70)