# Official Test-Set Access Log

Official `test-*.parquet` files are locked except for the single finalist evaluation pre-registered in [`phase_7_preregistration.md`](phase_7_preregistration.md). No official test file has been read before this entry.

## Access 1 — CatBoost engineered finalist

- Status: **PRE-REGISTERED; NOT YET EXECUTED**
- Pre-registration commit: `86e804f` (`phase 7: preregister finalist and test protocol`)
- Evaluator code commit: `6f691ef93cb9110617958d9b0f93b99aa41a2e5b` (`phase 7: final evaluator and validation thresholds`)
- Planned command:
  ```powershell
  .\.venv\Scripts\python.exe -m src.evaluation.evaluate_phase7_final --config configs/phase7_calibration.yaml
  ```
- Scope: read the official test shards once in a streaming pass; score only `catboostengineered.joblib`, apply the committed Platt calibrator and validation-selected thresholds, and use the same in-memory/hash-indexed predictions to map benchmark SHA-256 files. No retraining, tuning, threshold changes, or model selection from test labels.
- Outputs: `experiments/phase7_final_evaluation.json`, `experiments/phase7_benchmark_results.csv`, `experiments/phase7_benchmark_summary.csv`, and ignored `data/processed/phase7_test_predictions.parquet`.
- Expected row total: 168,060; report actual loaded row count after execution.

This pre-execution record is committed before the command is run. Append the actual execution status, final code commit hash, command, row count, and output hashes after the single pass.
