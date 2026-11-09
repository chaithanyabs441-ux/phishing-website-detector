# Official Test-Set Access Log

Official `test-*.parquet` files are locked except for the finalist evaluations pre-registered in [`phase_7_preregistration.md`](phase_7_preregistration.md) and explicitly authorized below. Each access entry is committed before its command runs.

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

## Access 2 — CatBoost engineered finalist (authorized retry)

- Status: **PRE-ACCESS RECORD COMMITTED; NOT YET EXECUTED**
- Authorization: user explicitly approved a new full pass on 2026-11-09 after Access 1 failed.
- Pre-registration commit: `86e804f` (`phase 7: preregister finalist and test protocol`)
- Corrected evaluator commit: `65ccb2bf3494e50cda914d5e100b3dbf643109d2` (`phase 7: record partial test evaluation and fix parse failure`)
- Command:
  ```powershell
  .\.venv\Scripts\python.exe -m src.evaluation.evaluate_phase7_final --config configs/phase7_calibration.yaml
  ```
- Scope: one full streaming pass over official test shards, scoring only the pre-registered CatBoost engineered finalist with the committed Platt calibrator and validation thresholds, then mapping benchmark hashes from the same predictions. No fitting, tuning, threshold changes, or model selection from test labels.
- Expected outputs: `experiments/phase7_final_evaluation.json`, `experiments/phase7_benchmark_results.csv`, `experiments/phase7_benchmark_summary.csv`, and ignored `data/processed/phase7_test_predictions.parquet`. All paths were absent before this entry.
- Expected row total: 168,060; record the actual total and all output hashes after execution.

