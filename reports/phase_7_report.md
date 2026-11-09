# Phase 7 — Calibration, thresholds, and partial locked-test evaluation

## What I did

- Confirmed the pre-registered finalist remains CatBoost engineered (`models/phase4/catboostengineered.joblib`) and checked the existing validation calibration record, thresholds, evaluator, and benchmark hash files.
- Ran the committed finalist feature/model smoke check using training-derived feature categories and a synthetic URL/HTML row. The model and Platt calibrator produced finite, deterministic scores.
- Ran the final evaluator once after committing the pre-access record. It completed 11 official test shards (76,880 rows) and failed while extracting an HTML feature in shard 12 because a malformed IPv6 page URL caused `urljoin` to raise `ValueError: Invalid IPv6 URL`.
- Updated `src/evaluation/evaluate_phase7_final.py` to use the same parse-error feature fallback as the training cache for malformed HTML/page URL rows. Added a synthetic regression test in `tests/test_phase7_final.py`.
- Recorded the partial test access and failure in `reports/test_set_access_log.md`. No retry or resume was performed.

## Why

- The finalist and thresholds were selected using the committed pre-registration and validation-only results. The evaluator failure is an input-robustness defect: the Phase 3 cache already represents extractor failures as `html_parse_error=1` with other HTML features cleared, but the Phase 7 batch evaluator did not apply that fallback.
- The fix matches the training-cache representation so the serving feature schema stays consistent. The official test shards cannot be resumed or reread under the one-pass protocol; the code fix is validated only with synthetic input.

## Verification

- Focused pre-run checks:
  ```text
  .\.venv\Scripts\python.exe -m pytest -q tests/test_phase7_calibration.py tests/test_phase7_final.py
  8 passed
  .\.venv\Scripts\ruff.exe check src/evaluation/evaluate_phase7_calibration.py src/evaluation/evaluate_phase7_final.py src/evaluation/phase7_calibration.py tests/test_phase7_calibration.py tests/test_phase7_final.py
  All checks passed!
  ```
- Finalist smoke check (training data plus synthetic URL/HTML; no test shard):
  ```text
  train_rows=397480; feature_count=53; domain=example.org
  raw_probability=0.83662053; calibrated_probability=0.90492866; deterministic=True
  ```
- Benchmark input audit:
  ```text
  Files=976; HashLines=220181; DuplicateLines=0; FilesWithMalformedHash=0
  ```
  This includes 975 benchmark files and the shared `benign.txt` list.
- Post-fix regression suite and lint:
  ```text
  .\.venv\Scripts\python.exe -m pytest -q tests/test_phase7_calibration.py tests/test_phase7_final.py tests/test_html_features_v2.py
  13 passed, 13 dependency deprecation warnings
  .\.venv\Scripts\ruff.exe check src/evaluation/evaluate_phase7_final.py tests/test_phase7_final.py
  All checks passed!
  ```
- Locked-test command:
  ```text
  .\.venv\Scripts\python.exe -m src.evaluation.evaluate_phase7_final --config configs/phase7_calibration.yaml
  ```
  It failed in shard 12 with `ValueError: Invalid IPv6 URL`. Shards 1–11 completed with 76,880 rows. No final predictions or evaluation outputs were written.

## Validation results

The selected Platt calibrator was chosen by grouped out-of-fold log-loss on 100,775 temporal-validation rows across 46,561 registered domains. The saved record is `experiments/phase7_calibration.json`.

| Operating point | Probability threshold | Precision | Recall | FPR | Precision at 5% prevalence | Alerts / 1,000 at 5% prevalence |
|---|---:|---:|---:|---:|---:|---:|
| WARN | 0.06818 | 0.5897 | 0.9500 | 0.2598 | 0.1614 | 294.30 |
| BLOCK | 0.94129 | 0.9956 | 0.5706 | 0.0010 | 0.9679 | 29.48 |

The validation AP is 0.93851. The grouped OOF calibration metrics are Brier 0.05577, log-loss 0.19804, and ECE 0.02004. The full-validation refit metrics are Brier 0.05510, log-loss 0.19343, and ECE 0.01864. These are validation-only estimates, not final test results.

## Problems and surprises

- The official-test run exposed malformed page URL handling not covered by the pre-run synthetic smoke check. The stack trace identifies `urljoin` resolving an HTML resource against an invalid IPv6 URL.
- The run did not finish the test set, save its accumulated predictions, compute locked-test metrics, or evaluate benchmarks. The unknown partial row count in shard 12 is not inferred.
- The current one-pass authorization has been consumed. A retry would be a new official-test access event, not a continuation of the failed process.

## Risks and open questions

- The code fix is covered by synthetic regression tests but has not been validated against official test rows.
- Benchmark hashes have valid SHA-256 syntax and no duplicates within the input text files, but their mapping against test rows remains unevaluated.
- No locked-test model metrics or corrected benchmark scores can be reported from this partial run.

## Next phase

Phase 7 remains incomplete. The next step requires explicit authorization for a new single pass using the corrected evaluator. Before any further official-test read, commit the corrected evaluator and a new access-log entry. If authorized, that pass must read every official shard once, write its outputs atomically, and record the actual row count, hashes, and benchmark join audit. Do not rerun this failed command under the existing Access 1 entry.
