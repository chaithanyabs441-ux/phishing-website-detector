import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.evaluation import audit_missing_html
from src.evaluation.build_html_features_v2_cache import build_cache
from src.features.html_features_v2 import FEATURE_SPEC_HASH_V2


class HtmlFeaturesV2CacheTests(unittest.TestCase):
    def test_missing_html_audit_counts_training_rows_by_label(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_dir = temporary_path / "raw"
            input_dir.mkdir()
            pd.DataFrame([
                {"label": "benign", "html": "<p>available</p>"},
                {"label": "benign", "html": "  "},
                {"label": "phish", "html": None},
            ]).to_parquet(input_dir / "train-000.parquet", index=False)
            output_path = temporary_path / "audit.json"

            with patch.object(audit_missing_html, "OUTPUT", output_path):
                result = audit_missing_html.audit(input_dir, batch_size=2)

            self.assertEqual(result["rows"], 3)
            self.assertEqual(result["missing_html_by_label"]["benign"]["missing_html"], 1)
            self.assertEqual(result["missing_html_by_label"]["phish"]["missing_html"], 1)
            self.assertTrue(output_path.is_file())

    def test_completed_shard_is_checksum_validated_and_skipped_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            input_dir = temporary_path / "raw"
            cache_root = temporary_path / "cache"
            input_dir.mkdir()
            pd.DataFrame([
                {
                    "sha256": "a" * 64,
                    "url": "https://example.com/",
                    "label": "benign",
                    "html": "<html><title>Example</title><body>Welcome</body></html>",
                },
                {
                    "sha256": "b" * 64,
                    "url": "https://example.net/login",
                    "label": "phish",
                    "html": "<form><input type='password'></form>",
                },
            ]).to_parquet(input_dir / "train-000.parquet", index=False)

            first = build_cache(
                input_dir, cache_root, n_jobs=1, timeout_seconds=None, row_batch_size=1
            )
            shard_output = (
                cache_root / FEATURE_SPEC_HASH_V2 / "train-000_features_v2.parquet"
            )
            original_hash = first["shards"]["train-000.parquet"]["output_sha256"]
            self.assertTrue(first["complete"])
            self.assertEqual(first["rows"], 2)

            resumed = build_cache(
                input_dir, cache_root, n_jobs=1, timeout_seconds=None, row_batch_size=1
            )
            progress = (
                cache_root / FEATURE_SPEC_HASH_V2 / "progress.log"
            ).read_text(encoding="utf-8")
            manifest = json.loads(
                (cache_root / FEATURE_SPEC_HASH_V2 / "MANIFEST.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertTrue(resumed["complete"])
            self.assertEqual(resumed["rows"], 2)
            self.assertEqual(manifest["shards"]["train-000.parquet"]["output_sha256"], original_hash)
            self.assertIn("skip_complete shard=train-000.parquet", progress)
            self.assertTrue(shard_output.is_file())

            shard_output.write_bytes(b"corrupt cache output")
            repaired = build_cache(
                input_dir, cache_root, n_jobs=1, timeout_seconds=None, row_batch_size=1
            )
            progress_after_rebuild = (
                cache_root / FEATURE_SPEC_HASH_V2 / "progress.log"
            ).read_text(encoding="utf-8")
            self.assertTrue(repaired["complete"])
            self.assertEqual(repaired["shards"]["train-000.parquet"]["output_sha256"], original_hash)
            self.assertEqual(progress_after_rebuild.count("completed=train-000.parquet"), 2)


if __name__ == "__main__":
    unittest.main()