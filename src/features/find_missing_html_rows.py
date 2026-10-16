from pathlib import Path
import pandas as pd


parquet_folder = Path("data/raw/data")
html_folder = Path("data/processed/html_features")

problem_files = [
    "train-050.parquet",
    "train-054.parquet"
]


print("=" * 70)
print("FINDING EXACT MISSING HTML ROWS")
print("=" * 70)


for filename in problem_files:

    parquet_file = parquet_folder / filename

    csv_file = html_folder / (
        parquet_file.stem + "_html_features.csv"
    )

    print()
    print("-" * 70)
    print("FILE:", filename)
    print("-" * 70)

    original = pd.read_parquet(
        parquet_file,
        columns=["url", "label"]
    )

    processed = pd.read_csv(
        csv_file,
        usecols=["url", "label"]
    )

    original_urls = set(original["url"].astype(str))
    processed_urls = set(processed["url"].astype(str))

    missing_urls = original_urls - processed_urls

    print("Original rows :", len(original))
    print("CSV rows      :", len(processed))
    print("Missing URLs  :", len(missing_urls))

    for url in missing_urls:

        row = original[
            original["url"].astype(str) == url
        ]

        print()
        print("MISSING ROW:")
        print("URL   :", url)
        print("LABEL :", row["label"].iloc[0])


print()
print("=" * 70)
print("DONE")
print("=" * 70)