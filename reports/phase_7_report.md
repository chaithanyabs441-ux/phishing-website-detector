# Phase 7 — Calibration, threshold selection, and final evaluation

## What I did

- Kept the pre-registered finalist: CatBoost engineered features (`models/phase4/catboostengineered.joblib`).
- Selected Platt calibration and the WARN/BLOCK thresholds using grouped out-of-fold temporal-validation scores only.
- Committed a pre-access log for Access 1, which stopped in shard 12 on an invalid IPv6 URL. Matched the final evaluator's malformed-row fallback to the training cache and added a synthetic regression test.
- After explicit user authorization, committed a second pre-access log and ran one complete pass over all official test shards. The pass scored 168,060 rows and mapped the benchmark hashes from those same predictions.
- Saved the locked-test record, per-file benchmark CSV, benchmark summary CSV, and ignored test prediction Parquet. Details and hashes are in `reports/test_set_access_log.md` and the experiment record.

## Why

- The finalist follows the committed selection rule: it is the simplest single model within the top temporal-validation AP interval, while the OOF text hybrid and logistic stack did not show a reliable AP gain.
- Calibration and threshold choices used validation only. The official test results are a final estimate and did not alter the finalist, calibrator, or thresholds.
- The first evaluator attempt exposed a malformed-URL exception. The corrected batch path writes the same `html_parse_error=1` zero-feature row used by the training feature cache, preserving the trained feature representation.

## Verification

- Finalist smoke check (training-derived categories and synthetic URL/HTML):
  ```text
  train_rows=397480; feature_count=53; domain=example.org
  raw_probability=0.83662053; calibrated_probability=0.90492866; deterministic=True
  ```
- Benchmark text input audit:
  ```text
  Files=976; HashLines=220181; DuplicateLines=0; FilesWithMalformedHash=0
  ```
  This includes 975 benchmark files and `benign.txt`.
- Synthetic regression suite after the malformed-URL fallback:
  ```text
  .\.venv\Scripts\python.exe -m pytest -q tests/test_phase7_calibration.py tests/test_phase7_final.py tests/test_html_features_v2.py
  13 passed, 13 dependency deprecation warnings
  .\.venv\Scripts\ruff.exe check src/evaluation/evaluate_phase7_final.py tests/test_phase7_final.py
  All checks passed!
  ```
- Complete repository suite after the final evaluation:
  ```text
  .\.venv\Scripts\python.exe -m pytest -q
  60 passed, 14 warnings in 48.27s
  .\.venv\Scripts\ruff.exe check src/evaluation/evaluate_phase7_final.py tests/test_phase7_final.py src/evaluation/phase7_calibration.py tests/test_phase7_calibration.py
  All checks passed!
  ```
- Authorized final evaluation command (Access 2):
  ```text
  .\.venv\Scripts\python.exe -m src.evaluation.evaluate_phase7_final --config configs/phase7_calibration.yaml
  test rows=168060; AP=0.990246; ROC-AUC=0.988918
  benchmark files=975; unmatched phish hashes=0
  elapsed=6835.74s
  ```

## Validation calibration and thresholds

Platt was selected by grouped out-of-fold log-loss on the 100,775-row temporal validation set across 46,561 registered domains. The selected OOF calibration had Brier 0.05577, log-loss 0.19804, and ECE 0.02004. The calibrator refit on all validation scores had Brier 0.05510, log-loss 0.19343, and ECE 0.01864. Validation AP was 0.93851.

| Operating point | Probability threshold | Precision | Recall | FPR | Precision at 5% prevalence | Alerts / 1,000 at 5% prevalence |
|---|---:|---:|---:|---:|---:|---:|
| WARN | 0.06818 | 0.5897 | 0.9500 | 0.2598 | 0.1614 | 294.30 |
| BLOCK | 0.94129 | 0.9956 | 0.5706 | 0.0010 | 0.9679 | 29.48 |

## Locked-test results

All 21 test shards yielded 168,060 rows and 95,151 registered domains. The observed phishing prevalence was 45.698%. There were no duplicate test SHA-256 rows. The table below reports the pre-registered 0.5 probability operating point; intervals are 95% registered-domain bootstrap intervals (200 replicates).

| Metric | Estimate | 95% CI |
|---|---:|---:|
| Average precision | 0.99025 | [0.98783, 0.99271] |
| ROC-AUC | 0.98892 | [0.98527, 0.99226] |
| Accuracy | 0.96123 | [0.95688, 0.96606] |
| Precision | 0.95454 | [0.95200, 0.95723] |
| Recall | 0.96094 | [0.95369, 0.96886] |
| F1 | 0.95773 | [0.95302, 0.96208] |
| FPR | 0.03852 | [0.03413, 0.04225] |

| Probability threshold | TP | FP | TN | FN |
|---|---:|---:|---:|---:|
| 0.5 reference | 73,800 | 3,515 | 87,745 | 3,000 |
| WARN 0.06818 | 75,800 | 28,373 | 62,887 | 1,000 |
| BLOCK 0.94129 | 63,602 | 98 | 91,162 | 13,198 |

Precision and alerts per 1,000 at the requested prevalences are reweighted from test class-conditional rates. Intervals are 95% registered-domain bootstrap intervals.

| Assumed phishing prevalence | Precision (95% CI) | Alerts / 1,000 (95% CI) |
|---:|---:|---:|
| 0.05% | 0.01233 [0.01125, 0.01392] | 38.98 [34.59, 42.70] |
| 0.5% | 0.11140 [0.10261, 0.12418] | 43.13 [38.77, 46.84] |
| 1% | 0.20128 [0.18688, 0.22179] | 47.74 [43.39, 51.44] |
| 5% | 0.56768 [0.54494, 0.59759] | 84.64 [80.38, 88.29] |
| 45% | 0.95330 [0.94902, 0.95848] | 453.61 [449.17, 458.25] |

## Corrected PhreshPhish benchmark results

Each SHA-256 mapped uniquely to a test row. The audit found 90,270 unique benign hashes, all correctly mapped; all benchmark phishing hashes also mapped with no label mismatches. There were zero ambiguous test hashes, unmatched hashes, or wrong-label joins.

| Configured base rate | Files | AP mean (std) | P@R90 mean (std) |
|---:|---:|---:|---:|
| 0.05% | 586 | 0.6373 (0.0631) | 0.0603 (0.0716) |
| 0.1% | 296 | 0.7047 (0.0435) | 0.1023 (0.0870) |
| 0.5% | 59 | 0.8256 (0.0161) | 0.2694 (0.0910) |
| 1% | 29 | 0.8618 (0.0107) | 0.4161 (0.0921) |
| 5% | 5 | 0.9203 (0.0024) | 0.7775 (0.0283) |

These results replace the legacy hash-as-URL benchmark scores documented in `docs/benchmark_results_INVALID.md`.

## Problems and surprises

- Access 1 stopped in shard 12 on malformed IPv6 input after 76,880 rows in the prior shards. It produced no results. That access and its failure remain recorded; the corrected evaluator then completed Access 2.
- At the validation-selected BLOCK threshold, test recall was 63,602 / 76,800 = 82.8%, higher than the validation estimate of 57.1%. WARN had 1,000 false negatives and 28,373 false positives at its validation recall target.
- The observed test AP is below the earlier historical random-split URL+HTML SVM AP (0.9942) but above the temporal-validation CatBoost AP (0.9385). These figures refer to different models and evaluation periods; the test result is not evidence that model selection generalizes without drift.

## Risks and open questions

- Thresholds were selected on validation only. Test outcomes should not be used to tune them or change the finalist.
- At a 0.05% phishing prevalence, the selected model's reweighted 0.5-threshold precision is about 1.23%; operational alert volume and decision costs still depend on the real deployment prevalence.
- This concludes Phase 7 evaluation. The existing FastAPI application still serves the three LinearSVC paths; serving CatBoost, calibration, and the new thresholds belongs to Phase 8 and has not been started.

## Next phase

Phase 8 proposal: add the pre-registered CatBoost model as a carefully validated serving option, retain the current SVM fallback, use calibrated WARN/BLOCK thresholds from `configs/thresholds.yaml`, and apply model artifact integrity checks. Stop here for approval before Phase 8.
