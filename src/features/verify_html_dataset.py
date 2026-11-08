from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_FOLDER = PROJECT_ROOT / "data" / "processed" / "html_features"
EXPECTED_FILES = 56
EXPECTED_ROWS = 498255
EXPECTED_LABELS = {"benign", "phish"}

REQUIRED_COLUMNS = {
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
    "has_external_form_action",
}


def verify_html_dataset(
    input_folder=DEFAULT_INPUT_FOLDER,
    expected_files=EXPECTED_FILES,
    expected_rows=EXPECTED_ROWS,
):
    """Validate processed HTML-feature CSVs and return whether all checks pass."""
    input_folder = Path(input_folder)
    files = sorted(input_folder.glob("train-*_html_features.csv"))
    total_rows = 0
    labels = set()
    missing_columns_by_file = {}
    missing_values = 0
    unreadable_files = []

    for file in files:
        try:
            frame = pd.read_csv(file)
        except Exception as error:
            unreadable_files.append((file.name, str(error)))
            continue

        total_rows += len(frame)
        if "label" in frame:
            labels.update(frame["label"].dropna().unique())
        missing_columns = REQUIRED_COLUMNS - set(frame.columns)
        if missing_columns:
            missing_columns_by_file[file.name] = sorted(missing_columns)
        missing_values += int(frame.isnull().sum().sum())

    files_ok = len(files) == expected_files
    rows_ok = total_rows == expected_rows
    labels_ok = labels == EXPECTED_LABELS
    columns_ok = not missing_columns_by_file
    readable_ok = not unreadable_files
    missing_ok = missing_values == 0

    print("=" * 70)
    print("HTML FEATURE DATASET VERIFICATION")
    print("=" * 70)
    print("Expected files:", expected_files)
    print("Found files   :", len(files))
    print("Files         :", "OK" if files_ok else "PROBLEM")
    print("Expected rows :", expected_rows)
    print("Actual rows   :", total_rows)
    print("Rows          :", "OK" if rows_ok else "PROBLEM")
    print("Expected labels:", sorted(EXPECTED_LABELS))
    print("Found labels   :", sorted(labels))
    print("Labels         :", "OK" if labels_ok else "PROBLEM")
    print("Columns        :", "OK" if columns_ok else "PROBLEM")
    print("Readable files :", "OK" if readable_ok else "PROBLEM")
    print("Missing values :", missing_values)
    print("Missing values :", "OK" if missing_ok else "PROBLEM")

    for filename, columns in missing_columns_by_file.items():
        print("Missing columns in {}: {}".format(filename, columns))
    for filename, error in unreadable_files:
        print("Could not read {}: {}".format(filename, error))

    passed = all((files_ok, rows_ok, labels_ok, columns_ok, readable_ok, missing_ok))
    print("Final result   :", "EVERYTHING LOOKS GOOD" if passed else "PROBLEM FOUND")
    print("=" * 70)
    return passed


def main():
    return verify_html_dataset()


if __name__ == "__main__":
    main()
