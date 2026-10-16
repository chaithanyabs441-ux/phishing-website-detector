from pathlib import Path
import sys
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from html_features import extract_html_features


# Load one parquet file
file_path = list(Path("data/raw/data").glob("train-*.parquet"))[0]

df = pd.read_parquet(
    file_path,
    columns=["url", "label", "html"]
)

print("ROWS LOADED:", len(df))
print()

# Test first 5 rows
for i in range(5):
    row = df.iloc[i]

    features = extract_html_features(
        row["html"],
        row["url"]
    )

    print("=" * 60)
    print("ROW:", i)
    print("LABEL:", row["label"])
    print("URL:", row["url"])
    print("HTML LENGTH:", len(row["html"]) if isinstance(row["html"], str) else 0)
    print()

    for name, value in features.items():
        print(f"{name}: {value}")