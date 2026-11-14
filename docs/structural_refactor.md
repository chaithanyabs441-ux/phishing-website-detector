# Structural refactor checkpoint

Date: 13 November 2026. Scope: the approved incremental structural refactor. Optional comment and test cleanup are excluded.

## Increment 1: shared schema and dependency boundaries

- Moved categorical names and ordered model columns into `src/features/feature_schema.py`; retained old exports through the Phase 4 data compatibility path.
- Removed the serving registry's import of the evaluation/data-loader module.
- Added `requirements-test.txt`, including the existing research pins required by the complete synthetic ML/cache test suite, and wired CI to it.
- Aligned `tasks.ps1 test` with CI's pytest command.

## Increment 2: HTTP adapter and serving service

- Extracted scan orchestration into `src/serving/scan_service.py`.
- Extracted verdict/evidence presentation and URL validation into their own serving modules.
- Extracted runtime settings readers and JSON request schemas.
- Retained `webapp.app:app`, all routes, form parameters, response fields, compatibility helper names, and request-time dependency substitutions.
- Shared calibration application now lives in `src/models/calibration.py`; fitting and validation selection remain in evaluation.

The focused threshold, registry, web security/upload, and dataset-assembly checks passed: **20 passed**.

## Increment 3: explicit data and training modules

Canonical implementations moved from feature/evaluation script folders into `src/data/` and `src/training/`. Old paths remain compatibility modules; the file depth and project-root resolution are preserved.

### Training moves

| Previous implementation | Canonical implementation |
|---|---|
| `src/evaluation/phase4_boosting.py` | `src/training/phase4_boosting.py` |
| `src/evaluation/phase5_text.py` | `src/training/phase5_text.py` |
| `src/evaluation/phase5_trees.py` | `src/training/phase5_trees.py` |
| `src/evaluation/phase6_stacking.py` | `src/training/phase6_stacking.py` |
| `src/evaluation/phase10b_categorical_ablation.py` | `src/training/phase10b_categorical_ablation.py` |
| `src/evaluation/phase10c_train_serving_models.py` | `src/training/phase10c_train_serving_models.py` |
| `src/models/train_url_baseline.py` | `src/training/train_url_baseline.py` |
| `src/models/train_url_combined.py` | `src/training/train_url_combined.py` |
| `src/models/train_url_html2.py` | `src/training/train_url_html2.py` |
| `src/models/smoke_train.py` | `src/training/smoke.py` |

### Data moves

| Previous implementation | Canonical implementation |
|---|---|
| `src/evaluation/phase4_data.py` | `src/data/phase4_dataset.py` |
| `src/evaluation/create_blocked_splits.py` | `src/data/splits.py` |
| `src/evaluation/build_html_features_v2_cache.py` | `src/data/html_feature_cache.py` |
| `src/evaluation/audit_missing_html.py` | `src/data/audit_missing_html.py` |
| `src/evaluation/download_benchmarks.py` | `src/data/download_benchmarks.py` |
| `src/evaluation/inspect_benchmarks.py` | `src/data/inspect_benchmarks.py` |
| `src/features/process_html_dataset.py` | `src/data/process_html_dataset.py` |
| `src/features/find_html_row_difference.py` | `src/data/find_html_row_difference.py` |
| `src/features/find_missing_html_rows.py` | `src/data/find_missing_html_rows.py` |
| `src/features/repair_missing_html_rows.py` | `src/data/repair_missing_html_rows.py` |
| `src/features/verify_html_dataset.py` | `src/data/verify_html_dataset.py` |

Executable data/training jobs now require explicit `main()`/`__main__` execution. Downloads, scans, repairs, model fitting, directory creation, and job logging in the affected legacy scripts no longer run on import. Existing operation order and calculations are preserved.

## Increment 4: evaluation, security, and diagnostic boundaries

- Extracted the existing metric bundle/domain bootstrap into `src/evaluation/metrics.py`, retaining old exports and unchanged calculations.
- Moved Phase 7, Phase 10, and the historical SVM official-test evaluator implementations into `src/evaluation/locked_test/`; old commands remain adapters.
- Moved request limits and safe fetch implementations into `src/security/`, retaining the old serving imports.
- Moved the serving smoke implementation to `scripts/serving/serving_smoke.py`, retaining the old model-package command.

## Preservation and checks

- No model or calibrator artifact, manifest, feature extractor definition, config value, threshold, frozen split assignment, evaluation result, report, or access log is changed.
- No test case is removed or rewritten.
- Module alias checks include private names, project-root resolution, schema-list identity, and the historical NB-SVM pickle class path.
- Implementation AST/byte comparisons were used for the mechanical data/training/evaluator moves, allowing only import/entry-point/root-bootstrap changes.
- Full-suite checks block reads of official `test-*.parquet` shards under `data/raw` and disable live Safe Browsing configuration.
- Ruff **0.14.3**: all checks passed across `src`, `webapp`, `scripts`, and `tests`.
- Full pytest: **88 passed, 1 failed, 14 warnings** in 26.44 seconds. The sole failure is the previously documented major-site sanity gate: **1 BLOCK, 2 WARN**; the zero-BLOCK requirement remains unmet. No test was weakened, removed, or marked as expected failure.
- No official-test access guard was triggered during the full-suite run.
- Both `python -m src.training.smoke` and the historical direct command `python src/models/smoke_train.py` returned the same tiny synthetic fit summary: 8 fit rows, 4 validation rows, 523 features, AP 1.0000.
- The local no-index requirements dry run succeeded for `requirements-test.txt`; a fresh remote package installation was not performed.
- `git diff --check` passed. Changes to `models/`, `configs/`, `reports/`, frozen `data/splits/`, and `tests/` were checked and absent.

## Deliberately retained items

Existing model/artifact paths, configuration paths, historical source entry points, reports, and access logs remain stable. The tracked `webapp.zip`, older root Word report, historical patch/archive records, and the two pre-existing untracked skill directories are retained. Their external use or provenance requires separate review before removal. The package namespace remains `src.*` for this checkpoint; changing it is a separate migration from the responsibility boundaries established here.
