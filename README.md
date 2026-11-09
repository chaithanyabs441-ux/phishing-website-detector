# PhishReaper

PhishReaper is a classical machine-learning phishing URL detector. The deployed path uses URL and HTML structure features with a CatBoost classifier, then applies a Platt calibrator. If the primary model cannot load or predict, the app falls back to the verified URL TF-IDF SVM and clearly identifies that its margin is uncalibrated. An HTML file upload is optional; server-side fetching is disabled by default.

Use the output for analyst triage. `BLOCK`, `WARN`, and `SAFE-LOOKING` are model signals, not proof about a site. The test set is an offline evaluation asset and should not be used to tune a deployment.

## Requirements and local setup

- Python 3.9 (the committed model artifacts and pins target this version).
- Windows PowerShell commands below use the project virtual environment.
- Model inference needs the committed artifacts under `models/`; training and evaluation additionally need the local dataset and generated caches.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
```

Run the app locally:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tasks.ps1 run-app
```

Open `http://127.0.0.1:8000`. The HTML form accepts a URL and optional `.html`/`.htm` file up to 10 MB. The JSON endpoint is `POST /api/v1/scan` with `{"url":"https://example.org","html":""}`. `/health` and `/version` return service status and model/feature version.

For tests and lint:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check src webapp tests
```

Run an offline real-model smoke check (Safe Browsing is mocked and no target pages are fetched):

```powershell
.\.venv\Scripts\python.exe -m src.models.serving_smoke
```

## Data

Training, validation, and test parquet files are not included in Git. Install the pinned optional download dependency, then fetch only the PhreshPhish parquet shards:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
.\.venv\Scripts\python.exe -m src.data.download_dataset
```

This downloads to `data/raw/`. The dataset source is `phreshphish/phreshphish` on Hugging Face. The dataset has 56 training shards (498,255 rows) and 21 official test shards (168,060 rows) in the snapshot evaluated here. Check the source dataset's current license and terms before redistribution or commercial use.

The benchmark hash lists are separate from dataset rows. To fetch them, run `.\.venv\Scripts\python.exe -m src.evaluation.download_benchmarks`; this downloads the benchmark directory only. Benchmark entries are SHA-256 IDs, not URLs, and must be joined on the dataset `sha256` column.

## Reproducible pipeline

All feature code is shared from `src/features/`. Configurations are under `configs/`; experiment records and phase reports are under `experiments/` and `reports/`. The fixed split manifests are stored under `data/splits/` and excluded from the raw data.

The end-to-end research sequence is:

1. Download the train shards, then build and audit frozen train-only splits:
   `.\.venv\Scripts\python.exe -m src.evaluation.create_blocked_splits`
2. Run the existing SVM baselines on the temporal holdout:
   `.\.venv\Scripts\python.exe -m src.evaluation.evaluate_svm_blocked`
   `.\.venv\Scripts\python.exe -m src.evaluation.evaluate_svm_grouped_cv`
3. Build the richer training HTML cache (the full recorded run took 2 h 40 m on this machine, so the compute-budget approval rule applies before rerunning it):
   `.\.venv\Scripts\python.exe -u -m src.evaluation.build_html_features_v2_cache --n-jobs 2 --timeout-seconds 1800 --row-batch-size 500`
4. Tune and compare classical tree models with the committed config:
   `.\.venv\Scripts\python.exe -m src.evaluation.phase4_boosting --config configs/phase4_gbm.yaml`
5. Compare classical text models and hybrids:
   `.\.venv\Scripts\python.exe -m src.evaluation.evaluate_phase5_text --config configs/phase5_text.yaml`
6. Generate grouped OOF stack comparisons:
   `.\.venv\Scripts\python.exe -m src.evaluation.phase6_stacking --config configs/phase6_stacking.yaml`
7. Fit calibration and choose thresholds using temporal validation only:
   `.\.venv\Scripts\python.exe -m src.evaluation.evaluate_phase7_calibration --config configs/phase7_calibration.yaml`

The evaluations above need local data and caches. Set seeds and resource limits in the YAML configs; preserve each emitted experiment JSON/CSV with its split/config hashes. Do not rerun the locked-test evaluator as part of routine development. Phase 7 records the one completed authorized pass and its access log at `reports/test_set_access_log.md`.

## Serving configuration

`configs/model_config.yaml` selects the model artifacts and feature inputs; `configs/thresholds.yaml` contains the validation-selected calibrated cutoffs. `models/MANIFEST.json` records artifact hashes. On startup the loader verifies hashes and project-contained paths before loading. The CatBoost finalist and calibrator are the primary path; the URL TF-IDF SVM is the fallback.

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

The split protocol is temporal validation over later collection dates plus five-fold registered-domain grouped CV/OOF for tuning and stack inputs. Calibration and thresholds were fitted from grouped OOF scores on temporal validation. The final official test pass was performed once for the selected CatBoost model. It achieved AP 0.99025 [0.98783, 0.99271] and ROC-AUC 0.98892 [0.98527, 0.99226] at the test score ranking level; at a 0.5 probability cut, precision was 0.95454, recall 0.96094, F1 0.95773, FPR 0.03852, and accuracy 0.96123 (95% registered-domain bootstrap intervals are in the model card).

At a 0.05% assumed phishing prevalence, that same 0.5 cutoff yields estimated precision 1.23% and 38.98 alerts per 1,000. These estimates show why evaluation prevalence and operating costs matter. Full test and benchmark results are in `reports/phase_7_report.md`; the unified temporal validation comparison and published-claim discussion are in `reports/phase_9_report.md`.

## Limitations

- The data is a historical collection and the test set has 45.7% phishing prevalence; operational prevalence and URL distributions will differ.
- WARN is deliberately recall-oriented and had 25.98% validation FPR at its selected threshold. It can produce many false alarms.
- Shared hosting, famous brands, URL-only irreducible cases, and distribution shift remain difficult. Page HTML is optional and user-supplied; the server does not render pages.
- Local allowlists are context only and do not alter the model probability. Domain lists need maintenance.
- In-process rate limiting is per worker. Production deployment needs shared rate limiting, authentication/access controls, monitoring, and egress policy.
- No neural-network model, pretrained embedding, or browser automation is used.

See [MODEL_CARD.md](MODEL_CARD.md) for intended use, detailed metrics, and ethical and privacy notes. Historical one-off files are preserved in `archive/phase9/`.
