from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.html_features import extract_html_features


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

parquet_folder = Path("data/raw/data")
html_folder = Path("data/processed/html_features")

problem_files = [
    "train-050.parquet",
    "train-054.parquet"
]


# --------------------------------------------------
# REPAIR
# --------------------------------------------------

print("=" * 70)
print("REPAIRING MISSING HTML ROWS")
print("=" * 70)


for filename in problem_files:

    parquet_file = parquet_folder / filename

    csv_file = html_folder / (
        parquet_file.stem + "_html_features.csv"
    )

    print()
    print("-" * 70)
    print("PROCESSING:", filename)
    print("-" * 70)

    original = pd.read_parquet(
        parquet_file,
        columns=["url", "label", "html"]
    )

    processed = pd.read_csv(csv_file)

    original_urls = set(
        original["url"].astype(str)
    )

    processed_urls = set(
        processed["url"].astype(str)
    )

    missing_urls = original_urls - processed_urls

    print("Missing rows:", len(missing_urls))

    for url in missing_urls:

        row = original[
            original["url"].astype(str) == url
        ].iloc[0]

        print()
        print("Repairing:")
        print("URL   :", row["url"])
        print("LABEL :", row["label"])

        features = extract_html_features(
            row["html"],
            row["url"]
        )

        features["url"] = row["url"]
        features["label"] = row["label"]

        new_row = pd.DataFrame([features])

        new_row = new_row[processed.columns]

        processed = pd.concat(
            [processed, new_row],
            ignore_index=True
        )

    processed.to_csv(
        csv_file,
        index=False
    )

    print()
    print("SAVED:", csv_file)
    print("New row count:", len(processed))


print()
print("=" * 70)
print("REPAIR COMPLETE")
print("=" * 70)
