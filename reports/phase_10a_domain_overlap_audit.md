# Phase 10A — Train-domain overlap audit

This is a descriptive slice of the already-authorized Phase 7 predictions. The official test shards were read only for `sha256` and `url`; labels and calibrated scores came from `data/processed/phase7_test_predictions.parquet`. Training domains came from the frozen temporal training assignments joined to grouped-domain assignments. No fitting or threshold selection was performed.

- Test rows joined: 168,060
- Frozen training registered domains: 138,231
- Existing validation-selected BLOCK threshold: 0.941289490641

| Test domain group | Rows | Domains | AP | Recall at BLOCK | FPR at BLOCK |
|---|---:|---:|---:|---:|---:|
| Seen in training | 63,954 | 15,235 | 0.838875 | 0.702309 | 0.000604 |
| Unseen in training | 104,106 | 79,848 | 0.994796 | 0.835670 | 0.001960 |

These test-set slice metrics are descriptive only and will not be used for model, feature, or threshold selection.
