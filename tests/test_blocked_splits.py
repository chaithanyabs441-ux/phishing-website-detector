import unittest

import pandas as pd

from src.evaluation.create_blocked_splits import build_assignments


class BlockedSplitTests(unittest.TestCase):
    @staticmethod
    def rows():
        records = []
        for i in range(10):
            day = i + 1
            records.append({
                "sha256": f"hash-{i}",
                "url": f"https://site{i}.testdomain{i}.com/p/{day}",
                "label": "phish" if i % 2 else "benign",
                "date": pd.Timestamp(f"2025-01-{day:02d}", tz="UTC"),
            })
        records.extend([
            {
                "sha256": "crossing-hash",
                "url": "https://cross.example.net/a",
                "label": "benign",
                "date": pd.Timestamp("2025-01-01", tz="UTC"),
            },
            {
                "sha256": "crossing-hash",
                "url": "https://cross.example.net/a",
                "label": "benign",
                "date": pd.Timestamp("2025-01-08", tz="UTC"),
            },
        ])
        return pd.DataFrame(records)

    def test_temporal_assignments_are_disjoint_ordered_and_purge_boundary_hashes(self):
        data = self.rows()
        temporal, _, audit = build_assignments(data, train_fraction=0.6, n_splits=3)
        labels_by_sha = temporal.groupby("sha256")["split"].nunique()
        self.assertTrue((labels_by_sha == 1).all())
        self.assertEqual(
            temporal.set_index("sha256").loc["crossing-hash", "split"],
            "purged_cross_boundary",
        )
        cutoff = pd.Timestamp(audit["cutoff_date_utc"])
        keyed = data.merge(temporal[["sha256", "split"]], on="sha256")
        self.assertLess(keyed.loc[keyed.split == "train", "date"].max(), cutoff)
        self.assertGreaterEqual(keyed.loc[keyed.split == "validation", "date"].min(), cutoff)

    def test_grouped_folds_keep_registered_domains_together(self):
        _, grouped, _ = build_assignments(self.rows(), train_fraction=0.6, n_splits=3)
        self.assertTrue((grouped.groupby("sha256")["fold"].nunique() == 1).all())
        self.assertTrue((grouped.groupby("registered_domain")["fold"].nunique() == 1).all())
        for fold in sorted(grouped["fold"].unique()):
            validation_domains = set(grouped.loc[grouped.fold == fold, "registered_domain"])
            training_domains = set(grouped.loc[grouped.fold != fold, "registered_domain"])
            self.assertFalse(validation_domains & training_domains)


if __name__ == "__main__":
    unittest.main()
