from pathlib import Path
import pandas as pd


html_folder = Path("data/processed/html_features")
parquet_folder = Path("data/raw/data")

html_files = sorted(html_folder.glob("train-*_html_features.csv"))
parquet_files = sorted(parquet_folder.glob("train-*.parquet"))

print("=" * 70)
print("FINDING MISSING HTML ROWS")
print("=" * 70)

total_parquet = 0
total_csv = 0

for parquet_file in parquet_files:

    csv_file = html_folder / (
        parquet_file.stem + "_html_features.csv"
    )

    parquet_df = pd.read_parquet(
        parquet_file,
        columns=["url", "label"]
    )

    parquet_rows = len(parquet_df)
    total_parquet += parquet_rows

    if not csv_file.exists():
        print()
        print("MISSING CSV:", csv_file.name)
        continue

    csv_df = pd.read_csv(
        csv_file,
        usecols=["url", "label"]
    )

    csv_rows = len(csv_df)
    total_csv += csv_rows

    difference = parquet_rows - csv_rows

    if difference != 0:
        print()
        print("PROBLEM FOUND")
        print("File       :", parquet_file.name)
        print("Parquet    :", parquet_rows)
        print("HTML CSV   :", csv_rows)
        print("Difference :", difference)

print()
print("=" * 70)
print("TOTAL")
print("=" * 70)

print("Original Parquet rows :", total_parquet)
print("HTML CSV rows         :", total_csv)
print("Difference            :", total_parquet - total_csv)

print("=" * 70)