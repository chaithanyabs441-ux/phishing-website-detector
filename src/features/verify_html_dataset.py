from pathlib import Path
import pandas as pd


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

input_folder = Path("data/processed/html_features")

EXPECTED_FILES = 56
EXPECTED_ROWS = 498255


# --------------------------------------------------
# FIND CSV FILES
# --------------------------------------------------

files = sorted(input_folder.glob("train-*_html_features.csv"))

print("=" * 70)
print("HTML FEATURE DATASET VERIFICATION")
print("=" * 70)

print()
print("Expected CSV files :", EXPECTED_FILES)
print("Found CSV files    :", len(files))

if len(files) == EXPECTED_FILES:
    print("FILE COUNT: OK")
else:
    print("FILE COUNT: PROBLEM")


# --------------------------------------------------
# CHECK EACH FILE
# --------------------------------------------------

total_rows = 0
all_labels = set()

required_columns = {
    "url",
    "label",
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
}

print()
print("-" * 70)
print("CHECKING FILES")
print("-" * 70)

for number, file in enumerate(files, start=1):

    try:
        df = pd.read_csv(file)

        rows = len(df)
        total_rows += rows

        all_labels.update(df["label"].dropna().unique())

        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            print()
            print(f"[{number}/{len(files)}] PROBLEM: {file.name}")
            print("Missing columns:", missing_columns)
        else:
            print(
                f"[{number}/{len(files)}] OK: "
                f"{file.name} | rows={rows}"
            )

    except Exception as e:

        print()
        print(f"[{number}/{len(files)}] FAILED: {file.name}")
        print("ERROR:", e)


# --------------------------------------------------
# TOTAL ROW CHECK
# --------------------------------------------------

print()
print("=" * 70)
print("ROW COUNT CHECK")
print("=" * 70)

print("Expected rows :", EXPECTED_ROWS)
print("Actual rows   :", total_rows)

if total_rows == EXPECTED_ROWS:
    print("ROW COUNT: OK")
else:
    print("ROW COUNT: PROBLEM")


# --------------------------------------------------
# LABEL CHECK
# --------------------------------------------------

print()
print("=" * 70)
print("LABEL CHECK")
print("=" * 70)

print("Labels found:", sorted(all_labels))

if all_labels == {benign, phish}:
    print("LABELS: OK")
else:
    print("LABELS: PROBLEM")


# --------------------------------------------------
# CHECK COMPLETE DATASET
# --------------------------------------------------

print()
print("=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

missing_total = 0

for file in files:

    df = pd.read_csv(file)

    missing = df.isnull().sum().sum()

    if missing > 0:
        print(f"{file.name}: {missing} missing values")
        missing_total += missing


if missing_total == 0:
    print("MISSING VALUES: NONE")
else:
    print("TOTAL MISSING VALUES:", missing_total)


# --------------------------------------------------
# FINAL RESULT
# --------------------------------------------------

print()
print("=" * 70)
print("FINAL VERIFICATION RESULT")
print("=" * 70)

files_ok = len(files) == EXPECTED_FILES
rows_ok = total_rows == EXPECTED_ROWS
labels_ok = all_labels == {benign, phish}
missing_ok = missing_total == 0

if files_ok and rows_ok and labels_ok and missing_ok:

    print("EVERYTHING LOOKS GOOD")
    print()
    print("HTML DATASET IS READY FOR MODEL TRAINING.")

else:

    print("VERIFICATION FOUND A PROBLEM.")
    print("DO NOT TRAIN THE MODEL YET.")

print("=" * 70)