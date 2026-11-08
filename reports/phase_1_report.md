# Phase 1 — Shared features and model artifacts

## Goal

Make existing feature extraction consistent across training, evaluation, and serving; repair the HTML dataset verifier; add a separate offline URL feature schema v2; and document the model artifacts. No model was retrained.

## Changes

- Fixed `src/features/verify_html_dataset.py`: `benign` and `phish` are now string labels, verification is callable with a temporary folder and expected counts, and the command has a `main` guard. Added a small synthetic CSV test.
- Kept the existing 13-column `extract_url_features` behavior as v1 and added `extract_url_features_v1` as its named entry point. Added deterministic `extract_url_features_v2`, using the bundled Public Suffix List without network access. V2 adds registered domain/suffix, suffix category, tokenized suspicious-word count, entropy, digit and letter ratios, longest token, path depth, IDN/punycode flag, percent-encoded byte count, and a known-shortener flag. V2 handles user information, ports, IP addresses, malformed and empty URLs separately from v1.
- Replaced duplicated URL extraction in `evaluate_benchmarks.py` and duplicated HTML extraction in `repair_missing_html_rows.py` with imports from `src/features/`. Updated trainers and evaluation/processing scripts to use the same package imports, including root-path setup for direct script execution.
- Added parity and v2 edge-case tests. The v1 model-compatible extractor remains unchanged; new feature columns are not fed to existing models.
- Added `configs/model_config.yaml`. The web app now reads artifact filenames, model thresholds, and default/fallback model choices from it.
- Found that the URL+engineered classifier's saved scaler differs from the URL+HTML pipeline scaler. The old `url_feature_scaler.joblib` matches the combined classifier; the app had loaded `url_engineered_scaler.joblib` for that classifier. Copied the correct scaler to `models/combined_url_scaler.joblib`, updated the trainer/evaluators/app to use the clear name, and retained `url_engineered_scaler.joblib` for the URL+HTML pipeline. The app now loads the two scaler artifacts separately.
- Added `models/MANIFEST.json` with SHA-256, training script/split, feature version, and scikit-learn version (`1.3.2`) for each of the 10 shipped artifacts. Added Git ignore allow-list entries for the new scaler and manifest.

## Verification

- Full test suite:

  ```text
  .\.venv\Scripts\python.exe -m pytest -q
  20 passed in 28.91s
  ```

- Focused feature and verifier suite:

  ```text
  .\.venv\Scripts\python.exe -m unittest tests.test_feature_parity tests.test_url_features_v2 tests.test_verify_html_dataset -v
  Ran 8 tests in 0.052s
  OK
  ```

- Ruff on changed Python files:

  ```text
  .\.venv\Scripts\ruff.exe check [changed Python files]
  All checks passed!
  ```

- Imported the web app and loaded all three model pipelines from configured artifact paths. Thresholds loaded as `url_tfidf=1.5`, `url_engineered=0.0`, and `url_html=0.0`. Sample engineered-URL and URL+HTML predictions completed (`-0.850375` and `-0.748619` decision scores for `https://example.com/login` with a synthetic password form).
- Checked all 10 manifest hashes against the files: `hash mismatches: []`.
- Feature tests include standard v1 URL parity and HTML parity fixtures, multi-label suffixes, URL userinfo/ports, IPs, IDNs, shorteners, encoded bytes, malformed/empty/long inputs, and repeat-call determinism.
- Did not run training, benchmark evaluation, or official test evaluation. Official `test-*.parquet` files were not opened or changed in this phase.

## Known difference and risk

Before removing the benchmark evaluator's private URL feature function, it was compared with the serving v1 extractor over 498,255 training URLs. There were 389 mismatches, concentrated in URLs with user information or ports. The benchmark copy was not fully identical to the serving function; Phase 1 makes evaluation use the serving/model-compatible v1 path. Standard URL fixtures pass parity, but benchmark scores for those unusual URL forms may change. This all-training comparison is not an official test evaluation.

The scaler fix corrects the app path to use the artifact saved by the combined-model trainer; models were not retrained. V2 is an additional extractor only; existing classifiers still expect v1. The manifest records the repository scripts and split settings; it does not prove historical provenance beyond those files and the inspected artifacts.

## Next

Phase 1 is complete. Wait for explicit direction before starting the next phase.
