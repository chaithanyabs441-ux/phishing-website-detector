# Filename consistency refactor

Date: 14 November 2026. Scope: the approved filename audit and rename plan.

## Changes

Applied 40 canonical filename changes: 27 Python implementation modules, 10 test modules, two YAML configurations, and one JavaScript module. Python names use snake_case. Phase identifiers remain in experiment-specific names and all existing research records.

Python implementations retain their former module/CLI paths as compatibility aliases. Previously existing aliases now target the descriptive canonical modules. Function names, private helper names, constants, assertions, fixtures, algorithms, feature order, calibration, thresholds, and output/provenance filenames were preserved.

| Previous path | Canonical path |
|---|---|
| `src/data/inspect_dataset.py` | `src/data/download_phreshphish_requests.py` |
| `src/data/check_local_data.py` | `src/data/inspect_official_test_shards.py` |
| `src/data/profile_dataset.py` | `src/data/profile_training_and_official_test_shards.py` |
| `src/data/find_html_row_difference.py` | `src/data/compare_html_feature_row_counts_v1.py` |
| `src/data/find_missing_html_rows.py` | `src/data/find_missing_html_feature_urls_v1.py` |
| `src/training/train_url_html2.py` | `src/training/train_url_html_svm.py` |
| `src/evaluation/feature_importance.py` | `src/evaluation/inspect_url_svm_coefficients.py` |
| `src/evaluation/threshold_analysis.py` | `src/evaluation/analyze_url_engineered_svm_test_thresholds.py` |
| `src/evaluation/analyze_errors.py` | `src/evaluation/analyze_url_tfidf_svm_test_errors.py` |
| `src/training/train_url_combined.py` | `src/training/train_url_engineered_svm.py` |
| `src/evaluation/evaluate_url_combined.py` | `src/evaluation/evaluate_url_engineered_svm_official.py` |
| `src/evaluation/evaluate_url_baseline.py` | `src/evaluation/evaluate_url_tfidf_svm_official.py` |
| `src/data/repair_missing_html_rows.py` | `src/data/repair_html_feature_rows_v1.py` |
| `src/data/process_html_dataset.py` | `src/data/build_html_feature_csvs_v1.py` |
| `src/data/verify_html_dataset.py` | `src/data/verify_html_feature_csvs_v1.py` |
| `src/data/phase4_dataset.py` | `src/data/engineered_feature_dataset.py` |
| `src/training/phase5_text.py` | `src/training/text_models.py` |
| `src/evaluation/phase7_calibration.py` | `src/evaluation/calibration_validation.py` |
| `src/training/phase10c_train_serving_models.py` | `src/training/train_serving_catboost_phase10c.py` |
| `src/evaluation/locked_test/phase7.py` | `src/evaluation/locked_test/evaluate_finalist_phase7.py` |
| `src/evaluation/locked_test/phase10.py` | `src/evaluation/locked_test/evaluate_serving_routes_phase10.py` |
| `src/evaluation/locked_test/svm_html.py` | `src/evaluation/locked_test/evaluate_url_html_svm.py` |
| `src/training/train_url_baseline.py` | `src/training/train_url_tfidf_svm.py` |
| `src/evaluation/find_validation_threshold.py` | `src/evaluation/select_url_engineered_svm_validation_threshold.py` |
| `src/evaluation/bootstrap_exp2_exp3.py` | `src/evaluation/bootstrap_url_svm_comparison_exp2_exp3.py` |
| `src/evaluation/evaluate_benchmarks.py` | `src/evaluation/evaluate_legacy_svm_benchmarks.py` |
| `src/training/smoke.py` | `src/training/training_smoke.py` |
| `tests/test_phase4_data.py` | `tests/test_engineered_feature_dataset.py` |
| `tests/test_phase5_text.py` | `tests/test_text_models.py` |
| `tests/test_phase7_calibration.py` | `tests/test_calibration_validation.py` |
| `tests/test_phase7_final.py` | `tests/test_finalist_evaluation_phase7.py` |
| `tests/test_phase10_final_evaluation.py` | `tests/test_serving_routes_evaluation_phase10.py` |
| `tests/test_phase10_threshold_config.py` | `tests/test_serving_threshold_config.py` |
| `tests/test_verify_html_dataset.py` | `tests/test_verify_html_feature_csvs_v1.py` |
| `tests/test_real_html.py` | `tests/test_html_feature_inputs.py` |
| `tests/test_threshold.py` | `tests/test_scan_verdicts.py` |
| `tests/test_smoke_train.py` | `tests/test_training_smoke.py` |
| `configs/phase5_text.yaml` | `configs/model_comparison_phase5.yaml` |
| `configs/model_config.yaml` | `configs/serving.yaml` |
| `webapp/static/visuals.js` | `webapp/static/hero_background.js` |

## Active references and configuration

Updated imports, relevant test imports, compatibility targets, YAML path readers, the CI synthetic-training command, the HTML asset reference, current README/model-card/architecture references, and six narrow data-module .gitignore exceptions. PowerShell launchers, Docker entry points, dependencies, and the webapp.app:app ASGI path are unchanged.

`configs/serving.yaml` replaces `configs/model_config.yaml`. `configs/model_comparison_phase5.yaml` replaces `configs/phase5_text.yaml`. Both renamed YAML files retain their exact original bytes. Current callers and commands use the new paths; historical commands in preserved records refer to their original revision/configuration filenames.

Three already-ignored local utilities remain ignored: download_phreshphish_requests.py, inspect_official_test_shards.py, and profile_training_and_official_test_shards.py. This rename does not change which local utilities are published to Git.

## Preservation

- SHA-256 checks confirmed 112 protected files unchanged, including every registered model artifact, the model manifest, feature extractor/schema implementations, thresholds and other unchanged configs, reports/access logs, experiment records, archives, and patch records.
- Original Python implementations and tests have identical ASTs after excluding imports and normalizing only the two approved YAML filename changes. No assertions or test fixtures were changed.
- The renamed JavaScript module is byte-identical. The only template edit changes its static asset filename; the existing versioned URL mechanism is retained.
- Existing manifest trained_with entries, provenance-writing strings, historical report commands and preregistered report placeholders were not rewritten.
- Both src.evaluation.phase5_text.NBSVMClassifier and src.training.phase5_text.NBSVMClassifier resolve to the canonical src.training.text_models.NBSVMClassifier without loading a pickle/artifact.
- No official test dataset was opened or evaluated. No historical reports, model artifacts, datasets or duplicate candidates were renamed or deleted.

## Verification

- Baseline full pytest under an official-test read guard: 92 passed, 1 failed, 13 warnings (30.56 seconds).
- After renames, full pytest under the same guard: 92 passed, 1 failed, 13 warnings (25.54 seconds). No blocked official-test read attempts in either run.
- The unchanged failure is tests/test_sanity_urls.py::test_major_site_sanity_urls_stay_below_phishing_alert_gate: BLOCK=1, WARN=2; its requirement remains BLOCK=0 and WARN<=2. No test was weakened or hidden.
- Ruff 0.14.3 passed across src, webapp, scripts and tests.
- 51 import-alias identity checks passed with dataset and model loading prohibited. Both historical NB-SVM class lookups and feature-schema object identity passed.
- The historical downloader that creates its output directory on import and the historical bootstrap script that executes on import were verified statically only. Their import-time behavior was not changed in this filename-only migration. The unchanged serving-smoke import initializes the app and is covered by app startup checks instead of the no-model-loading import check.
- Both the new synthetic-training CLI and the historical direct src/models/smoke_train.py command passed: 8 fit rows, 4 validation rows, 523 features, AP 1.0000.
- Restarted the existing local app with the renamed serving configuration: home and health returned HTTP 200. Prediction, probability, verdict, evidence and feature-summary fields for synthetic URL-only and URL+HTML requests matched the pre-restart process.
- Validation was run on Windows/Python 3.9. Linux execution was not available locally; path depth, pathlib usage and the existing Ubuntu CI entry points were preserved.
- No commit or push was made.

The previous structural and frontend checkpoints remain historical records. Duplicate/obsolete-file candidates from the audit remain untouched.
