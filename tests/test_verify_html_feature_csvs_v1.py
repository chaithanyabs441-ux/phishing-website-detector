import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.verify_html_feature_csvs_v1 import REQUIRED_COLUMNS, verify_html_dataset


class VerifyHtmlDatasetTests(unittest.TestCase):
    def test_validates_a_tiny_synthetic_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            rows = []
            for label in ("benign", "phish"):
                row = {column: 0 for column in REQUIRED_COLUMNS}
                row["url"] = "https://example.test/"
                row["label"] = label
                rows.append(row)
            pd.DataFrame(rows).to_csv(
                folder / "train-000_html_features.csv",
                index=False,
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                passed = verify_html_dataset(
                    folder,
                    expected_files=1,
                    expected_rows=2,
                )

        self.assertTrue(passed)
        self.assertIn("Labels         : OK", output.getvalue())
        self.assertIn("EVERYTHING LOOKS GOOD", output.getvalue())


if __name__ == "__main__":
    unittest.main()
