# Phase 3 — HTML feature pipeline v2

## What I did

- Added `src/features/html_features_v2.py` with deterministic, offline HTML features and cleaned text. The extractor caps input at 500,000 characters, never executes JavaScript, and uses the shared offline registered-domain helper from `src/features/url_features.py`.
- Added `src/evaluation/build_html_features_v2_cache.py` to process only `data/raw/data/train-*.parquet`. It writes one Parquet output per source shard under `data/processed/features_v2/<feature-spec-sha256>/`, retaining each row's `sha256` and storing the feature-spec hash.
- Added resumable shard-level output with temporary files, atomic finalization, per-shard SHA-256 checksums, a progress log, and a 1,800-second worker timeout. Completed outputs are skipped only when their checksums match the manifest.
- Added `src/evaluation/audit_missing_html.py`, `configs/html_features_v2.yaml`, and the tracked 5,000-row sample experiment record at `experiments/phase3_html_sample.json`.
- Added extractor fixtures and cache resume/audit tests in `tests/test_html_features_v2.py` and `tests/test_html_features_v2_cache.py`. Updated `.gitignore` narrowly so the Phase 3 sample JSON is tracked while generated cache files remain excluded.

## Why

The existing `data/processed/html_features/` contains the older feature schema and cannot supply Phase 3's richer HTML signals or cleaned text for Phase 5. The v2 schema is versioned independently so existing SVM artifacts and feature behavior are unchanged. Parquet shard outputs reduce repeated parsing while the checksum manifest makes interrupted runs resumable and auditable. Two workers were used to limit memory pressure on the 12-logical-CPU machine.

## Results

| Cache | Rows | Source shards | Elapsed | Throughput | Parse errors | Missing HTML |
|---|---:|---:|---:|---:|---:|---:|
| 5k smoke sample | 5,000 | 2 | 93.09 s | 53.71 rows/s | 0 | 0 |
| Full training cache | 498,255 | 56 | 9,596.88 s (2 h 39 m 57 s) | 51.92 rows/s | 5 | 0 |

The full output row count equals the manifest count. All 56 output files matched their recorded SHA-256 and row counts. Every output is keyed by its `sha256` column and contains the v2 feature-spec hash `f5434603820e65699462bc276cc34b2c6ba4ad7ed72f9f061a1cc0d13d79e6d3`.

| Label | Rows | Missing HTML | Missing rate |
|---|---:|---:|---:|
| benign | 276,729 | 0 | 0.00% |
| phish | 221,526 | 0 | 0.00% |

Features include form/input counts and form-action domain mismatch; distinct external, script, CSS, and image domains; external-link/resource ratios; favicon mismatch; title and visible-text lengths; title/domain token mismatch; hidden iframes; meta refresh and JavaScript redirect indicators; mouseover/context-menu indicators; data URI and empty/hash-link counts; token-based visible login keywords; explicit missing/truncated/parse-error indicators; and title-plus-visible-text capped at 2,048 characters.

No model metrics or confidence intervals apply in this feature/cache phase. No official test files were accessed.

## Verification

- Full cache command:
  ```text
  .\.venv\Scripts\python.exe -u -m src.evaluation.build_html_features_v2_cache --n-jobs 2 --timeout-seconds 1800 --row-batch-size 500
  ```
  Final manifest: `complete=True`, `shards=56`, `rows=498255`, `parse_errors=5`, `official_test_accessed=False`.
- Full-cache integrity check (all shard checksums and Parquet row counts):
  ```text
  complete=True shards=56 manifest_rows=498255 output_rows=498255 parse_errors=5
  checksum_or_row_mismatches=0 missing_html_by_label={"benign": {"missing_html": 0, "missing_rate": 0.0, "rows": 276729}, "phish": {"missing_html": 0, "missing_rate": 0.0, "rows": 221526}}
  elapsed_seconds=9596.88 official_test_accessed=False
  ```
- Test suite:
  ```text
  .\.venv\Scripts\python.exe -m pytest -q
  28 passed in 33.83s
  ```
- Lint and syntax:
  ```text
  .\.venv\Scripts\ruff.exe check src/features/html_features_v2.py src/evaluation/build_html_features_v2_cache.py src/evaluation/audit_missing_html.py tests/test_html_features_v2.py tests/test_html_features_v2_cache.py
  All checks passed!
  .\.venv\Scripts\python.exe -m py_compile src/features/html_features_v2.py src/evaluation/build_html_features_v2_cache.py src/evaluation/audit_missing_html.py
  [exit 0]
  ```
- Resume regression verified that a valid completed shard is skipped and a checksum-corrupt shard is rebuilt. Fixture tests cover a phishing login clone, benign blog, missing HTML, input cap, and single/batch parity.

## Problems and surprises

- The full cache took about 2 hours 40 minutes, substantially longer than a straight-line estimate from the initial sample. It stayed within the approved two-worker run and completed all training shards.
- Five rows were marked `html_parse_error=1`; replaying only those rows reproduced `ValueError: Invalid IPv6 URL` for each. The errors were contained and no row URLs or HTML were logged. These failures arise from malformed page URLs during URL resolution; affected rows retain the explicit error indicator rather than silently appearing as successful parses.
- Parsing is bounded by the 500,000-character cap and each shard worker has a 30-minute timeout. There is no separate per-document wall-clock timeout; this remains a limitation if parser behavior on adversarial HTML becomes a serving concern.
- Missing HTML was zero for both labels in this training snapshot, so no class-conditional missingness signal was observed.

## Risks and open questions

- The title/domain mismatch is a simple token heuristic based on the registered domain's first label; it is not brand resolution and may flag legitimate titles.
- A timed-out shard is retried at shard granularity, so completed shards are reused but work inside an interrupted shard may be repeated.
- The cache is intentionally ignored by Git because it is a large generated artifact. Its manifest and checksums remain local; the feature spec and sample experiment record are version-controlled.

## Next phase

Phase 4: tune LightGBM, XGBoost, and CatBoost on training-only blocked splits, with engineered URL v2 and HTML v2 features, plus the required no-text and out-of-fold-text-score variants. Before any tuning or refit expected to exceed 60 minutes, obtain explicit runtime approval as required by the compute budget. Keep the official test set locked.
