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

- Status: **COMPLETED**
- Authorization: user explicitly approved a new full pass on 2026-11-09 after Access 1 failed.
- Pre-registration commit: `86e804f` (`phase 7: preregister finalist and test protocol`)
- Pre-access record commit: `e58ab583f59aa056a13919295b115f53e15e5e69` (`phase 7: authorize corrected finalist test pass`)
- Corrected evaluator commit used: `e58ab583f59aa056a13919295b115f53e15e5e69`
- Command:
  ```powershell
  .\.venv\Scripts\python.exe -m src.evaluation.evaluate_phase7_final --config configs/phase7_calibration.yaml
  ```
- Scope: one full streaming pass over official test shards, scoring only the pre-registered CatBoost engineered finalist with the committed Platt calibrator and validation thresholds, then mapping benchmark hashes from the same predictions. No fitting, tuning, threshold changes, or model selection from test labels.
- Execution: completed all 21 shards in one pass; 168,060 rows and 95,151 registered domains. Duplicate test SHA-256 rows: 0. Runtime: 6,835.74 seconds. Test AP: 0.990246; ROC-AUC: 0.988918.
- Benchmark audit: 975 files; 90,270 unique benign hashes correctly mapped; zero unmatched or wrong-label hashes; zero ambiguous test hashes.
- Output SHA-256 values:
  - `experiments/phase7_final_evaluation.json`: `5c9428d2370d9ee93e4658f540cb51e16c697da9d4d5eabc1e0de404566dd5bd`
  - `experiments/phase7_benchmark_results.csv`: `c37b9ff8b5dd0eaeccd6fe9dd06760a0718f9fda49874b96b40717eadeba3d47`
  - `experiments/phase7_benchmark_summary.csv`: `d5d9873e03cabeebb0199c024d1fff8ea095da9bd901a796fa17d89016e50c24`
  - `data/processed/phase7_test_predictions.parquet` (ignored): `142ce4c0b4c319e26ca697b4859cca75ad916f7493ed9f8ce0703fd3c3063080`

## Access 3 — Phase 10A descriptive train-domain overlap audit

- Status: **COMPLETED; DESCRIPTIVE ONLY**
- Authorization: user explicitly requested Phase 10A on 2026-11-12, including a descriptive train-domain overlap audit and seen/unseen metrics from the already-recorded predictions.
- Pre-access record commit: `2243c29afe23dc05b24aeb3514f927926b0031e6`.
- Command: `python -m src.evaluation.domain_overlap_audit`
- Scope executed: read only `sha256` and `url` columns, in batches, from the 21 official `test-*.parquet` shards (168,060 rows). Joined identities by SHA-256 to the existing `data/processed/phase7_test_predictions.parquet`; targets and calibrated scores came only from that prior authorized artifact. Training domains came only from frozen temporal training assignments joined to grouped-domain assignments. No fitting, tuning, threshold changes, benchmark evaluation, or model selection.
- Result: 63,954 test rows / 15,235 registered domains seen in training; AP 0.838875, recall at the existing validation-selected BLOCK threshold (0.941289490641) 0.702309, FPR 0.000604. 104,106 unseen-domain rows / 79,848 registered domains; AP 0.994796, BLOCK recall 0.835670, FPR 0.001960. Results and method are in `reports/phase_10a_domain_overlap_audit.md`. These figures are descriptive and did not change the model or thresholds.
- Stop condition: no other official-test access or Phase 10B/10C/10E evaluation was performed. Any further access must have a separate authorized pre-access record where required.

