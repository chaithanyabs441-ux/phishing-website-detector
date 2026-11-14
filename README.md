# PhishReaper

PhishReaper is a classical machine-learning phishing URL detector. The deployed path chooses a calibrated CatBoost URL-only model when no HTML is supplied and a calibrated URL+HTML model when HTML is supplied. A failure uses the verified URL TF-IDF SVM for that request; the margin is uncalibrated and the primary model remains available for the next request. An HTML file upload is optional; server-side fetching is disabled by default.

Use the output for analyst triage. `BLOCK`, `WARN`, and `NO STRONG SIGNAL` are model signals, not proof about a site. The test set is an offline evaluation asset and must not be used to tune a deployment.

## Architecture

The web adapter is in `webapp/`; scan orchestration and the artifact registry are in `src/serving/`. Pure feature extraction is in `src/features/`, dataset preparation in `src/data/`, fitting/tuning in `src/training/`, and validation tools in `src/evaluation/`. Official-test evaluators are isolated under `src/evaluation/locked_test/`. Old module commands remain compatibility entry points during the migration.

See [docs/architecture.md](docs/architecture.md) for dependency boundaries and [docs/structural_refactor.md](docs/structural_refactor.md) for the incremental move map and verification record.

## Requirements and local setup

- Python 3.9 (the committed model artifacts and pins target this version).
- Windows PowerShell commands below use the project virtual environment.
- Model inference needs the committed artifacts under `models/`; training and evaluation additionally need the local dataset and generated caches.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

This installs the pinned app and model-inference dependencies. Install optional development and research tools only when needed:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-research.txt -r requirements-data.txt
```

The research requirements include the runtime dependencies plus PyArrow, LightGBM, XGBoost, and Optuna. Dataset download support is separate. The trained model files are committed under `models/`; do not install the research or dataset packages just to run the app.

Run the app locally:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tasks.ps1 run-app
```

On macOS or Linux, create the Python 3.9 environment, install the same runtime file, then launch Uvicorn:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn webapp.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The HTML form accepts a URL and optional `.html`/`.htm` file up to 10 MB. The JSON endpoint is `POST /api/v1/scan` with `{"url":"https://example.org","html":""}`. `/health` and `/version` return service status and model/feature version.

For the full test suite and lint, install the test environment first:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check src webapp scripts tests
```

Run an offline real-model smoke check (Safe Browsing is mocked and no target pages are fetched):

```powershell
.\.venv\Scripts\python.exe -m scripts.serving.serving_smoke
```

## Data

Training, validation, and test parquet files are not included in Git. Install the pinned optional download dependency, then fetch only the PhreshPhish parquet shards:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
.\.venv\Scripts\python.exe -m src.data.download_dataset
```

This downloads to `data/raw/`. The dataset source is `phreshphish/phreshphish` on Hugging Face. The dataset has 56 training shards (498,255 rows) and 21 official test shards (168,060 rows) in the snapshot evaluated here. Check the source dataset's current license and terms before redistribution or commercial use.

The benchmark hash lists are separate from dataset rows. To fetch them, run `.\.venv\Scripts\python.exe -m src.data.download_benchmarks`; this downloads the benchmark directory only. Benchmark entries are SHA-256 IDs, not URLs, and must be joined on the dataset `sha256` column.

## Reproducible pipeline

All feature code is shared from `src/features/`. Configurations are under `configs/`; experiment records and phase reports are under `experiments/` and `reports/`. The fixed split manifests are stored under `data/splits/` and excluded from the raw data.

The end-to-end research sequence is:

1. Download the train shards, then build and audit frozen train-only splits:
   `.\.venv\Scripts\python.exe -m src.data.splits`
2. Run the existing SVM baselines on the temporal holdout:
   `.\.venv\Scripts\python.exe -m src.evaluation.evaluate_svm_blocked`
   `.\.venv\Scripts\python.exe -m src.evaluation.evaluate_svm_grouped_cv`
3. Build the richer training HTML cache (the full recorded run took 2 h 40 m on this machine, so the compute-budget approval rule applies before rerunning it):
   `.\.venv\Scripts\python.exe -u -m src.data.html_feature_cache --n-jobs 2 --timeout-seconds 1800 --row-batch-size 500`
4. Tune and compare classical tree models with the committed config:
   `.\.venv\Scripts\python.exe -m src.training.phase4_boosting --config configs/phase4_gbm.yaml`
5. Compare classical text models and hybrids:
   `.\.venv\Scripts\python.exe -m src.evaluation.evaluate_phase5_text --config configs/model_comparison_phase5.yaml`
6. Generate grouped OOF stack comparisons:
   `.\.venv\Scripts\python.exe -m src.training.phase6_stacking --config configs/phase6_stacking.yaml`
7. Fit calibration and choose thresholds using temporal validation only:
   `.\.venv\Scripts\python.exe -m src.evaluation.evaluate_phase7_calibration --config configs/phase7_calibration.yaml`

The evaluations above need local data and caches. Set seeds and resource limits in the YAML configs; preserve each emitted experiment JSON/CSV with its split/config hashes. Do not rerun the locked-test evaluator as part of routine development. `reports/test_set_access_log.md` records the authorized Phase 7 pass and the descriptive Phase 10A identity audit.

## Serving configuration

`configs/serving.yaml` selects the URL-only and URL+HTML model artifacts; `configs/thresholds.yaml` contains separate validation-selected calibrated cutoffs. Empty HTML selects URL-only CatBoost; nonempty supplied HTML selects URL+HTML CatBoost. Both routes have their own grouped-OOF-selected calibrator. `models/MANIFEST.json` records artifact hashes. On startup the loader verifies hashes and project-contained paths before loading. The URL TF-IDF SVM is the fallback for a missing or failing primary route.

The validation cutoffs use grouped-OOF calibrated validation scores. BLOCK is set at validation FPR no higher than 0.1%; WARN is set at validation FPR no higher than 2%.

| Route | Calibration | BLOCK cutoff (validation FPR / recall) | WARN cutoff (validation FPR / recall) |
|---|---|---:|---:|
| URL-only | Isotonic | 0.98860 (0.000968 / 0.39520) | 0.38889 (0.019450 / 0.77077) |
| URL + HTML | Platt | 0.94129 (0.000995 / 0.57062) | 0.58389 (0.019989 / 0.81068) |

The threshold provenance is `validation temporal split only; grouped OOF-calibrated scores`; there is no manual override. Model and threshold methodology, artifacts, and validation metrics are in [reports/phase_10c_serving_models.md](reports/phase_10c_serving_models.md).

At an illustrative 0.05% phishing prevalence, reweighting each route's validation FPR and recall gives estimated precision/alerts per 1,000 scans of 17.0% / 1.16 for URL-only BLOCK, 22.3% / 1.28 for URL+HTML BLOCK, about 1.9% / 19.8 for URL-only WARN, and 2.0% / 20.4 for URL+HTML WARN. These are prevalence-shift estimates, not measured production results; they assume the validation class-conditional rates transfer to the deployment.

The local `configs/top_sites.txt` list is a small bootstrap context list, not a current Tranco snapshot. To refresh it, download a Tranco list manually, review its format, and replace the file with one registered domain per line. The app does not fetch that list. `configs/shared_hosting_domains.txt` contains platforms where URL-only evidence is inconclusive.

HTML fetching is controlled by `serving.enable_html_fetch` and is `false` by default. If enabled, the fetcher applies DNS/IP, scheme, port, redirect, response-size, timeout, and content-type checks. Keep it disabled unless deployment egress controls also address DNS rebinding. Uploaded HTML is parsed locally and JavaScript is never executed. Uploads are limited to 10 MB and parsing is limited to 10 million characters.

Google Safe Browsing is optional. Set `GOOGLE_SAFE_BROWSING_API_KEY` in the environment or copy `.env.example` to `.env`. When enabled, the complete submitted URL, including path and query, is sent to Google's Safe Browsing API for lookup; submitted HTML is not sent. A threat match overrides the model verdict. No match does not mean safe. Do not put secrets in Git or logs.

## Containers

Docker Compose runs only the web app and mounts local model artifacts read-only:

```powershell
docker compose up --build
```

The service binds to `127.0.0.1:8000`. The data directory is not copied into the image. See `Dockerfile` and `docker-compose.yml` before changing the network exposure or mounting secrets.

Safe Browsing is disabled in Compose by default. To enable it for a one-off Compose run, first supply `GOOGLE_SAFE_BROWSING_API_KEY` through your secret manager or PowerShell process environment, then use `docker compose run --service-ports -e GOOGLE_SAFE_BROWSING_API_KEY phishreaper`. Compose files deliberately do not interpolate the project `.env` key into rendered configuration.

## Evaluation protocol and results

The split protocol uses temporal validation plus five-fold registered-domain grouped CV/OOF for tuning and calibration. Phase 10B compared the current full categorical feature set (A) with reduced categorical variants. B and C were rejected by the preregistered validation rule: paired AP-difference 95% intervals versus A were [-0.03303, -0.01398] and [-0.04964, -0.02481], respectively, both below the -0.01 noninferiority margin.

Phase 10C validation results for the current two routes are:

| Route | Validation AP / ROC-AUC (raw scores) | Grouped-OOF calibrated AP / ROC-AUC |
|---|---:|---:|
| URL-only | 0.923877 / 0.955195 | 0.914668 / 0.946892 |
| URL + HTML | 0.940751 / 0.963142 | 0.938509 / 0.960363 |

No official test pass has been run for the new URL-only model or the Phase 10 serving package. The metrics below are the prior authorized Phase 7 pass for the earlier single URL+HTML CatBoost finalist. They are historical results and are not presented as fresh evaluation of both current serving routes.

That Phase 7 pass measured test AP 0.99025 [0.98783, 0.99271] and ROC-AUC 0.98892 [0.98527, 0.99226]. At a 0.5 probability cut, precision was 0.95454, recall 0.96094, F1 0.95773, FPR 0.03852, and accuracy 0.96123 (95% registered-domain bootstrap intervals). At an assumed 0.05% phishing prevalence, the reweighted estimate was 1.23% precision and 38.98 alerts per 1,000.

The corrected Phase 7 hash-mapped benchmark summary is also for that prior single-model pass:

| Configured base rate | Files | AP mean (std) | P@R90 mean (std) |
|---:|---:|---:|---:|
| 0.05% | 586 | 0.6373 (0.0631) | 0.0603 (0.0716) |
| 0.1% | 296 | 0.7047 (0.0435) | 0.1023 (0.0870) |
| 0.5% | 59 | 0.8256 (0.0161) | 0.2694 (0.0910) |
| 1% | 29 | 0.8618 (0.0107) | 0.4161 (0.0921) |
| 5% | 5 | 0.9203 (0.0024) | 0.7775 (0.0283) |

The Phase 10A descriptive overlap audit reused the prior Phase 7 predictions and read only `sha256` and `url` from official shards. At the prior BLOCK threshold of 0.941289, seen-in-training test domains had AP 0.838875, BLOCK recall 0.702309, and FPR 0.000604 (63,954 rows / 15,235 domains); unseen domains had AP 0.994796, recall 0.835670, and FPR 0.001960 (104,106 rows / 79,848 domains). These are descriptive slices of that prior test pass, not model-selection results. The corrected benchmark, locked-test protocol, and overlap audit are documented in [reports/phase_7_report.md](reports/phase_7_report.md), [reports/test_set_access_log.md](reports/test_set_access_log.md), and [reports/phase_10a_domain_overlap_audit.md](reports/phase_10a_domain_overlap_audit.md).

## Limitations

- The data is a historical collection and the test set has 45.7% phishing prevalence; operational prevalence and URL distributions will differ.
- A fixed 50-URL major-site regression gate currently reports 1 BLOCK and 2 WARN, so it fails the zero-BLOCK requirement. The list is not used for fitting or threshold selection. See [reports/phase_10e_sanity_url_gate.md](reports/phase_10e_sanity_url_gate.md).
- The Phase 10A seen/unseen test slice results are descriptive and show materially different metrics between groups. They must not be used to select features or thresholds.
- Shared hosting, famous brands, URL-only irreducible cases, and distribution shift remain difficult. Page HTML is optional and user-supplied; the server does not render pages.
- Local allowlists are context only and do not alter the model probability. Domain lists need maintenance.
- In-process rate limiting is per worker. Production deployment needs shared rate limiting, authentication/access controls, monitoring, and egress policy.
- No neural-network model, pretrained embedding, or browser automation is used.

See [MODEL_CARD.md](MODEL_CARD.md) for intended use, detailed metrics, and ethical and privacy notes. Historical one-off files are preserved in `archive/phase9/`.
