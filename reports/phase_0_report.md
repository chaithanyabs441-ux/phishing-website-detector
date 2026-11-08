# Phase 0 — Safety net and reproducibility

## What I did

- Confirmed the repository was already a Git repository (`main`, initial HEAD `0cad8c7`); did not reinitialize it.
- Kept the existing Python 3.9 runtime pins in `requirements.txt` and added pinned classical-ML and feature/config dependencies: LightGBM, XGBoost, CatBoost, Optuna, `tldextract`, and PyYAML. Added Python 3.9-compatible `anyio` and `greenlet` pins after dependency resolution exposed the need for them.
- Added `requirements-dev.txt` with pinned pytest and Ruff.
- Added `/experiments/` to `.gitignore`. Confirmed `.env` is ignored and untracked; `.env.example` already existed with the placeholder variable.
- Moved the two regression modules from `patch/tests/` to `tests/`, added `tests/__init__.py`, and left the patch documentation and rollback backups in place.
- Added `tasks.ps1` with `test`, `lint`, and `run-app` tasks.
- Updated the README test command to use the new test task after moving the regression tests.
- Removed the generated Hugging Face cache directory `data/benchmarks/.cache/` earlier in this task. Raw data, benchmark data, models, logs, and saved results were not changed.
- Detected 12 logical CPUs.

## Why

- Reused the existing Git history and dependency pins rather than replacing project setup that was already present in the working tree.
- Selected classical-model packages that have compatible artifacts for the existing Python 3.9 Windows environment. No neural-network package was added.
- Moved regression tests into a normal `tests/` directory so the project has a stable test command while retaining `patch/` as change history.
- Added only task commands needed now. Model-development workflows and experiment configuration are left for their later phases.

## Verification

- Initial regression baseline, before moving tests:

  ```text
  .\.venv\Scripts\python.exe -m unittest patch.tests.test_threshold patch.tests.test_google_safe_browsing -v
  Ran 12 tests in 0.120s
  OK
  ```

- Installed dependencies:

  ```text
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
  Successfully installed ... lightgbm-4.6.0 xgboost-2.1.4 catboost-1.2.10
  optuna-5.0.0 tldextract-5.1.2 pytest-8.4.2 ruff-0.14.3
  ```

  The first install attempt failed because the old pip 20.2.3 resolver selected incompatible dependencies; upgrading the venv pip to 25.3 and pinning `anyio==4.10.0` and `greenlet==3.1.1` resolved installation. `greenlet` otherwise attempted a source build and failed because this machine's compiler setup lacks `math.h`.

- Import/version check:

  ```text
  .\.venv\Scripts\python.exe -c "import lightgbm,xgboost,catboost,optuna,tldextract,yaml,pytest,ruff; print('all classical ML and dev imports OK')"
  all classical ML and dev imports OK
  ```

  Installed versions: Python 3.9.0; FastAPI 0.128.8; Uvicorn 0.39.0; scikit-learn 1.3.2; NumPy 1.26.4; pandas 2.3.3; PyArrow 21.0.0; LightGBM 4.6.0; XGBoost 2.1.4; CatBoost 1.2.10; Optuna 5.0.0; tldextract 5.1.2; PyYAML 6.0.3; pytest 8.4.2; Ruff 0.14.3.

- Post-move task test:

  ```text
  powershell -NoProfile -ExecutionPolicy Bypass -File .\tasks.ps1 test
  Ran 12 tests in 0.031s
  OK
  ```

- Local app smoke check started Uvicorn on `127.0.0.1:8011`, requested `/`, then stopped that process:

  ```text
  HTTP status: 200
  Title rendered: True
  ```

- Ruff check:

  ```text
  .\.venv\Scripts\python.exe -m ruff check src webapp tests
  Found 16 errors.
  ```

  Existing findings include undefined `benign`/`phish` names in `src/features/verify_html_dataset.py`, unused imports/variables, and imports below executable statements. The verifier bug is scheduled for Phase 1; no unrelated source cleanup was made in Phase 0.

- `pip check` reports two pre-existing platform warnings: `multiprocess 0.70.19 is not supported on this platform` and `xxhash 4.0.1 is not supported on this platform`. Neither is a direct project requirement.
- Checked requirements for neural-network packages: none found.
- `.env` is ignored by Git; `.env.example` is not ignored. The `.env` value was never printed or copied.
- Official test data was not evaluated or recomputed in this phase. Baseline metrics below were copied from saved evaluation artifacts only.

## Baseline results

| Model and source | Rows | Accuracy | Precision | Recall | F1 | AP | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| URL TF-IDF, `results/same_test_model_comparison.csv` | 5,000 | 0.9624 | 0.9789 | 0.9358 | 0.9569 | 0.9925 | 0.0162 | 0.0642 |
| URL + engineered, `results/same_test_model_comparison.csv` | 5,000 | 0.9646 | 0.9794 | 0.9403 | 0.9595 | 0.9932 | 0.0159 | 0.0597 |
| URL + HTML, `logs/evaluate_url_html_test_log.txt` | 168,060 | 0.9683 | 0.9817 | 0.9482 | Not recorded | 0.9942 | Not separately recorded | Not separately recorded |

The 5,000-row comparison used the original zero threshold. It does not measure the currently configured URL-only threshold of 1.5. The URL + HTML figures are copied from the saved official-test log, including its confusion matrix; metrics not recorded there were not recomputed. No confidence intervals were present in these saved baseline outputs.

## Problems / surprises

- The prompt described no Git repository, an empty `requirements.txt`, and no `.env.example`; the current repository has Git history, a populated but uncommitted requirements file, and an existing untracked `.env.example`.
- The raw dataset is present: 56 train shards and 21 test shards. Its sampled schema is `sha256, url, label, target, date, lang, lang_score, html`.
- A sample benchmark entry is a 64-character hexadecimal string. Its mapping to the dataset `sha256` column has not been tested yet.
- Existing staged and unstaged user changes were present before Phase 0, including a staged `Untitled-1.ps1` → `run.ps1` rename. They were not included in the Phase 0 changes.
- The system PowerShell policy blocks direct execution of `.ps1` scripts; the test task works when invoked with `-ExecutionPolicy Bypass` for that process.

## Risks and open questions

- The API-key variable name exists in `.env`, but its value was deliberately not inspected. Rotate the key as the prompt requests, especially if it has ever been exposed.
- Benchmark hash mapping, the claimed recall at threshold 1.5, and leakage rates remain unverified. The official test set remains locked until Phase 7.
- Ruff currently fails on 16 existing findings; Phase 1 will fix the specified verifier defect and related feature-path issues.
- The repository's pre-existing staged/unstaged changes must remain separate from phase commits.

## Next phase

Phase 1: fix HTML dataset verification, centralize shared v1 feature extraction with parity tests, add a separate deterministic v2 URL feature set, standardize artifact metadata, and test edge cases. Do not retrain models.

## Commit scope

The phase commit includes only Phase 0 files. Pre-existing staged/unstaged application and README changes are kept out of it.
