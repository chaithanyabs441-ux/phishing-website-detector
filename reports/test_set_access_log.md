# Official Test-Set Access Log

Official `test-*.parquet` files are locked except for the single finalist evaluation pre-registered in [`phase_7_preregistration.md`](phase_7_preregistration.md). No official test file has been read before this entry.

## Access 1 — CatBoost engineered finalist

- Status: **PARTIAL; EVALUATOR ABORTED DURING SHARD 12**
- Pre-registration commit: `86e804f` (`phase 7: preregister finalist and test protocol`)
- Pre-access log commit: `731e44527bcd82b5b7c34ff68f4206dbb95e422a`
- Evaluator code commit used: `6f691ef93cb9110617958d9b0f93b99aa41a2e5b` (`phase 7: final evaluator and validation thresholds`)
- Planned command:
  ```powershell
  .\.venv\Scripts\python.exe -m src.evaluation.evaluate_phase7_final --config configs/phase7_calibration.yaml
  ```
- Scope: read the official test shards once in a streaming pass; score only `catboostengineered.joblib`, apply the committed Platt calibrator and validation-selected thresholds, and use the same in-memory/hash-indexed predictions to map benchmark SHA-256 files. No retraining, tuning, threshold changes, or model selection from test labels.
- Execution: one invocation started after the pre-access log commit. It completed shards 1–11 (`76,880` rows) and opened shard 12, then aborted while extracting HTML features for a malformed IPv6 page URL (`ValueError: Invalid IPv6 URL` from `urllib.parse.urljoin`). The row count processed in shard 12 is unknown.
- Result: no final prediction parquet, final evaluation JSON, benchmark CSV, or benchmark summary was written. No locked-test metrics are available from this attempt.
- No retry or resume was performed. The parser failure was reproduced with a synthetic URL and the evaluator now falls back to the same `html_parse_error=1` feature row used by the training cache. Any further official-test access requires a new explicit authorization and a new pre-access log entry.

