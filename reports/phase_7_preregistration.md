# Phase 7 — Pre-registered selection and test protocol

This selection rule is committed before any official `test-*.parquet` file is opened in Phase 7.

## Finalist rule

Among Phase 2–6 candidates, identify the highest temporal-validation AP and its registered-domain 95% bootstrap interval. Select the simplest single model whose temporal-validation AP point estimate lies inside that interval. Do not select an ensemble unless it improves on the best single model under a paired registered-domain bootstrap.

## Selected finalist

- **CatBoost engineered features** (`models/phase4/catboostengineered.joblib`) is the sole finalist.
- The highest observed Phase 4/5 single-model point AP is CatBoost + OOF text score (0.941135; 95% CI 0.916644–0.959462). CatBoost engineered-only AP is 0.940751, inside that interval. The Phase 5 paired AP difference for adding the OOF text score is −0.000821 (95% CI −0.005129–0.003541), so the additional text model is not selected.
- The Phase 6 logistic stack AP is 0.938526 with a paired difference versus CatBoost of −0.002225 (95% CI −0.005978–0.000325), so the stack is not selected. The SVM URL+HTML baseline AP is 0.905383, below the selected interval’s lower bound; its historical official-test result will be copied from Phase 0 for context, not re-evaluated.

This rule uses only already-recorded temporal validation results. No test-set metrics influenced the finalist choice.

## Calibration and threshold protocol

- Obtain the selected CatBoost finalist’s validation scores from the Phase 4 record/artifact.
- Compare Platt/logistic and isotonic calibration with registered-domain GroupKFold on the temporal validation rows. Select by mean held-out log-loss, tie-break by Brier score; report ECE and reliability summaries. Refit the selected calibrator on all validation rows for final test scoring.
- Set the `BLOCK` threshold to the lowest calibrated probability threshold whose validation FPR is at most 0.1%. Set the `WARN` threshold to the lowest threshold achieving at least 95% validation recall. Report the resulting validation precision/recall/FPR and prevalence-reweighted alerts; do not describe `SAFE-LOOKING` as certainty.
- Report R10 metrics and 95% registered-domain bootstrap intervals at 0.05%, 0.5%, 1%, 5%, and 45% prevalence. Threshold and calibration selection use validation only.

## Benchmark and locked-test access

- Redo the hash-list benchmark by joining each benchmark SHA-256 to the dataset `sha256` column, never by scoring the hash string as a URL. Verify row labels and report unmatched/duplicate hashes. Evaluate only the selected CatBoost finalist, with AP and P@R90 per benchmark file/base rate and aggregate mean/std.
- After calibration, threshold selection, and benchmark code are fixed, run one final command that reads official test shards once, scores the selected CatBoost model once, and derives both the locked-test metrics and any benchmark rows that map to that test pass. Do not tune, select, or change thresholds from test results.
- The command and the committed code hash will be entered in `reports/test_set_access_log.md` before reading test rows. No SVM rerun is planned.

## Stop conditions

If benchmark hashes cannot be joined uniquely or the selected model artifact/feature schema fails a deterministic smoke check, stop before official-test access and document the blocker. No other model becomes a finalist based on these checks.
