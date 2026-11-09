# Phase 9 — Documentation, CI, and handoff

## What I did

- Replaced the outdated setup and serving description in `README.md`; documented dataset download, split protocol, reproducible pipeline commands, app/API operation, Safe Browsing privacy, model configuration, Docker, and limitations.
- Added `MODEL_CARD.md` with intended use, data, calibration, locked-test metrics and CIs, threshold tradeoffs, reweighted precision/alert counts, known failure classes, privacy, and ethical guidance.
- Added app-only `Dockerfile`, `docker-compose.yml`, `.dockerignore`, and pinned `requirements-app.txt`; model files are mounted read-only and data is excluded.
- Added GitHub Actions CI for Ruff, the unit suite, and a deterministic tiny-sample classical text-model fit. Added optional pinned Hugging Face download requirements and `src/data/download_dataset.py`.
- Preserved the Phase 0 stray files `parquet_locations.txt`, `project_structure.txt`, and `tree.py` under `archive/phase9/`. `Untitled-1.ps1` is not present in the current repository; `run.ps1` is the current launcher and was left in place.
- Fixed a serving calibration mismatch discovered while checking the model card. Phase 7 trains and applies Platt calibration to `logit(raw_probability)`, but the Phase 8 serving code had supplied raw probability. Serving now uses the same logit transform, with a parity regression test. The prior Phase 8 smoke probability table is explicitly marked invalid in its report.

## Why

- A handoff should match the committed serving path and the measured split/test protocol, rather than preserve README text that described the pre-Phase 8 SVM selector.
- Compose mounts `models/` read-only because the fitted artifacts are deployment inputs and the app has no training role. The image excludes datasets, experiment caches, secrets, and the model directory.
- The small CI training fixture checks vectorizer fitting, classical model fitting, and scoring without needing the large research dataset.
- The calibration mismatch was a correctness issue that changed verdict probabilities. It is fixed in this final phase, and the documentation distinguishes the earlier smoke output from the corrected serving behavior.

## Unified temporal-validation comparison

The table below combines the recorded temporal holdout results from Phases 2–6. Every row uses the same 100,775-row later-date validation partition and a score threshold of 0 (probability threshold 0.5 for probabilistic models). Confidence intervals resample registered domains (95%, 200 replicates). The artifacts and training schemes differ by phase; this is a common evaluation set and metric convention, not a claim that all tuning budgets or model fitting procedures were identical. Duplicate base-model reruns are represented once; all distinct tested model variants are included.

| Model | AP [95% CI] | ROC-AUC [95% CI] | Precision [95% CI] | Recall [95% CI] | F1 [95% CI] | FPR [95% CI] | Accuracy [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| SVM URL TF-IDF | 0.8945 [0.8632, 0.9210] | 0.9156 [0.8776, 0.9414] | 0.9616 [0.9573, 0.9655] | 0.6943 [0.6402, 0.7543] | 0.8064 [0.7675, 0.8447] | 0.0109 [0.0098, 0.0123] | 0.9059 [0.8823, 0.9258] |
| SVM URL + engineered | 0.8947 [0.8607, 0.9246] | 0.9100 [0.8725, 0.9405] | 0.9641 [0.9605, 0.9685] | 0.7315 [0.6805, 0.7919] | 0.8319 [0.7968, 0.8692] | 0.0107 [0.0094, 0.0122] | 0.9166 [0.8967, 0.9376] |
| SVM URL + HTML | 0.9054 [0.8800, 0.9293] | 0.9250 [0.8968, 0.9502] | 0.9664 [0.9624, 0.9702] | 0.7342 [0.6885, 0.7872] | 0.8344 [0.8045, 0.8683] | 0.0100 [0.0087, 0.0114] | 0.9178 [0.8991, 0.9357] |
| LightGBM engineered | 0.9192 [0.8929, 0.9405] | 0.9448 [0.9197, 0.9631] | 0.9530 [0.9485, 0.9578] | 0.7668 [0.7092, 0.8187] | 0.8498 [0.8137, 0.8817] | 0.0149 [0.0131, 0.0167] | 0.9235 [0.9012, 0.9394] |
| LightGBM + OOF text | 0.9183 [0.8889, 0.9399] | 0.9417 [0.9139, 0.9610] | 0.9702 [0.9659, 0.9737] | 0.7113 [0.6537, 0.7736] | 0.8208 [0.7817, 0.8610] | 0.0086 [0.0074, 0.0101] | 0.9124 [0.8885, 0.9324] |
| XGBoost engineered | 0.9210 [0.8918, 0.9434] | 0.9419 [0.9129, 0.9626] | 0.9624 [0.9580, 0.9663] | 0.7812 [0.7276, 0.8332] | 0.8624 [0.8281, 0.8939] | 0.0120 [0.0104, 0.0136] | 0.9296 [0.9083, 0.9450] |
| XGBoost + OOF text | 0.9129 [0.8802, 0.9356] | 0.9379 [0.9076, 0.9590] | 0.9598 [0.9546, 0.9635] | 0.7460 [0.6882, 0.8023] | 0.8395 [0.8007, 0.8734] | 0.0123 [0.0107, 0.0141] | 0.9195 [0.8957, 0.9367] |
| CatBoost engineered | 0.9408 [0.9167, 0.9606] | 0.9631 [0.9418, 0.9782] | 0.9791 [0.9764, 0.9818] | 0.7758 [0.7187, 0.8293] | 0.8657 [0.8291, 0.8983] | 0.0065 [0.0056, 0.0075] | 0.9321 [0.9103, 0.9478] |
| CatBoost + OOF text | 0.9411 [0.9166, 0.9595] | 0.9624 [0.9382, 0.9787] | 0.9824 [0.9790, 0.9851] | 0.7478 [0.6906, 0.8039] | 0.8492 [0.8106, 0.8845] | 0.0053 [0.0044, 0.0064] | 0.9251 [0.9034, 0.9441] |
| Logistic engineered | 0.8632 [0.8304, 0.8970] | 0.8893 [0.8539, 0.9185] | 0.8719 [0.8596, 0.8846] | 0.7459 [0.6853, 0.8004] | 0.8040 [0.7684, 0.8366] | 0.0431 [0.0379, 0.0474] | 0.8974 [0.8767, 0.9145] |
| Char SVC 2?5 | 0.8945 [0.8632, 0.9210] | 0.9156 [0.8776, 0.9414] | 0.9616 [0.9573, 0.9655] | 0.6943 [0.6402, 0.7543] | 0.8064 [0.7675, 0.8447] | 0.0109 [0.0098, 0.0123] | 0.9059 [0.8823, 0.9258] |
| Char SVC 3?6 | 0.8938 [0.8645, 0.9237] | 0.9155 [0.8826, 0.9427] | 0.9602 [0.9563, 0.9645] | 0.6856 [0.6370, 0.7474] | 0.7999 [0.7655, 0.8407] | 0.0112 [0.0098, 0.0126] | 0.9032 [0.8822, 0.9247] |
| Char LogisticRegression liblinear | 0.8911 [0.8651, 0.9176] | 0.9188 [0.8904, 0.9451] | 0.9468 [0.9414, 0.9518] | 0.7148 [0.6713, 0.7606] | 0.8146 [0.7857, 0.8437] | 0.0158 [0.0138, 0.0174] | 0.9082 [0.8873, 0.9258] |
| Char LogisticRegression saga | 0.8911 [0.8627, 0.9186] | 0.9188 [0.8867, 0.9451] | 0.9468 [0.9412, 0.9515] | 0.7147 [0.6684, 0.7696] | 0.8145 [0.7832, 0.8504] | 0.0158 [0.0143, 0.0175] | 0.9082 [0.8866, 0.9274] |
| Hash SGD modified-huber | 0.8815 [0.8472, 0.9131] | 0.9116 [0.8772, 0.9426] | 0.9324 [0.9265, 0.9394] | 0.7150 [0.6656, 0.7731] | 0.8093 [0.7754, 0.8457] | 0.0204 [0.0184, 0.0225] | 0.9049 [0.8867, 0.9248] |
| Hash SGD log-loss | 0.8371 [0.8051, 0.8646] | 0.8965 [0.8540, 0.9245] | 0.8568 [0.8442, 0.8701] | 0.6747 [0.6211, 0.7320] | 0.7549 [0.7181, 0.7921] | 0.0443 [0.0399, 0.0490] | 0.8764 [0.8514, 0.8954] |
| Char ComplementNB | 0.8606 [0.8296, 0.8910] | 0.9007 [0.8692, 0.9336] | 0.9354 [0.9249, 0.9435] | 0.6184 [0.5717, 0.6779] | 0.7446 [0.7083, 0.7866] | 0.0168 [0.0144, 0.0192] | 0.8803 [0.8588, 0.9044] |
| Char MultinomialNB | 0.8606 [0.8318, 0.8908] | 0.9007 [0.8663, 0.9296] | 0.9363 [0.9276, 0.9428] | 0.6161 [0.5623, 0.6693] | 0.7432 [0.7019, 0.7804] | 0.0165 [0.0146, 0.0192] | 0.8799 [0.8546, 0.8997] |
| Char NB-SVM | 0.8880 [0.8522, 0.9166] | 0.9027 [0.8607, 0.9351] | 0.9583 [0.9541, 0.9629] | 0.6998 [0.6302, 0.7483] | 0.8089 [0.7599, 0.8410] | 0.0120 [0.0106, 0.0134] | 0.9067 [0.8796, 0.9240] |
| HTML word SVC | 0.8834 [0.8644, 0.9037] | 0.9327 [0.9191, 0.9444] | 0.8610 [0.8186, 0.8972] | 0.6792 [0.6387, 0.7287] | 0.7594 [0.7257, 0.7931] | 0.0431 [0.0309, 0.0587] | 0.8785 [0.8589, 0.8995] |
| Char SVC min_df=1 | 0.8944 [0.8639, 0.9214] | 0.9154 [0.8819, 0.9442] | 0.9614 [0.9572, 0.9651] | 0.6946 [0.6358, 0.7427] | 0.8065 [0.7653, 0.8385] | 0.0109 [0.0096, 0.0122] | 0.9060 [0.8850, 0.9230] |
| Char SVC min_df=3 | 0.8946 [0.8658, 0.9228] | 0.9157 [0.8825, 0.9421] | 0.9615 [0.9575, 0.9661] | 0.6942 [0.6455, 0.7531] | 0.8063 [0.7722, 0.8460] | 0.0109 [0.0098, 0.0121] | 0.9059 [0.8833, 0.9252] |
| CatBoost + selected OOF text hybrid | 0.9399 [0.9145, 0.9595] | 0.9612 [0.9347, 0.9785] | 0.9826 [0.9797, 0.9852] | 0.7532 [0.6956, 0.8080] | 0.8527 [0.8143, 0.8873] | 0.0052 [0.0045, 0.0062] | 0.9266 [0.9049, 0.9449] |
| ExtraTrees engineered | 0.8932 [0.8644, 0.9178] | 0.9303 [0.9031, 0.9510] | 0.9355 [0.9274, 0.9418] | 0.7088 [0.6468, 0.7622] | 0.8065 [0.7629, 0.8417] | 0.0192 [0.0173, 0.0216] | 0.9040 [0.8805, 0.9213] |
| HistGradientBoosting engineered | 0.9034 [0.8714, 0.9293] | 0.9235 [0.8886, 0.9489] | 0.9489 [0.9435, 0.9542] | 0.7503 [0.6958, 0.8077] | 0.8380 [0.8014, 0.8735] | 0.0159 [0.0142, 0.0178] | 0.9182 [0.8970, 0.9371] |
| Mean-probability ensemble | 0.9218 [0.8951, 0.9478] | 0.9383 [0.9119, 0.9623] | 0.9780 [0.9752, 0.9808] | 0.7801 [0.7256, 0.8490] | 0.8679 [0.8335, 0.9090] | 0.0069 [0.0060, 0.0079] | 0.9330 [0.9139, 0.9515] |
| Rank-average ensemble | 0.9305 [0.9046, 0.9543] | 0.9508 [0.9234, 0.9712] | 0.5775 [0.5551, 0.6037] | 0.9359 [0.9015, 0.9657] | 0.7143 [0.6947, 0.7338] | 0.2691 [0.2450, 0.2923] | 0.7888 [0.7719, 0.8054] |
| Logistic OOF stack | 0.9385 [0.9174, 0.9618] | 0.9595 [0.9380, 0.9782] | 0.9475 [0.9412, 0.9530] | 0.8127 [0.7577, 0.8724] | 0.8750 [0.8405, 0.9097] | 0.0177 [0.0158, 0.0195] | 0.9345 [0.9164, 0.9539] |
| LightGBM OOF stack | 0.9166 [0.8897, 0.9437] | 0.9337 [0.9042, 0.9597] | 0.9590 [0.9544, 0.9630] | 0.7946 [0.7413, 0.8545] | 0.8691 [0.8370, 0.9055] | 0.0133 [0.0118, 0.0151] | 0.9325 [0.9113, 0.9515] |

The selected model's separate, one-pass locked-test estimates and corrected benchmark results are in `reports/phase_7_report.md`; they are not mixed into this temporal-validation comparison.

## Published-claim comparison

- The reported 99.45% CatBoost accuracy claim was **not reproduced** on this project's blocked temporal validation: the CatBoost engineered model had accuracy 0.9321 [0.9138, 0.9487]. Its separate official-test accuracy at a 0.5 cut was 0.96123 [0.95688, 0.96606], below 0.9945. The datasets, labels, splits, and operating conditions may differ, so this is not a direct replication of the paper.
- The reported 97.34% stacking accuracy claim was **not reproduced**: the logistic OOF stack had temporal-validation accuracy 0.9345 [0.9164, 0.9539], and it did not improve AP over the best single CatBoost model. The stack was not selected for the one-time test evaluation. Paper protocol differences prevent an apples-to-apples claim.
- The CNN+LightGBM result was deliberately not tested because PhishReaper is classical-ML-only.

## Verification

- Serving/calibrator parity and tiny training tests:
  ```text
  .\.venv\Scripts\python.exe -m pytest -q tests/test_model_registry.py tests/test_smoke_train.py
  4 passed
  ```
- Tiny smoke fit:
  ```text
  .\.venv\Scripts\python.exe -m src.models.smoke_train
  smoke_fit_rows=8 validation_rows=4 features=523 AP=1.0000
  ```
- Corrected five-URL serving smoke check (real CatBoost model; Safe Browsing mocked; no target page fetching):
  ```text
  .\.venv\Scripts\python.exe -m src.models.serving_smoke
  health: {'status': 'ok', 'backend': 'catboostengineered'}
  https://www.google.com => BLOCK (99.8%)
  https://example.com/login => BLOCK (97.5%)
  https://orca-app-2ulpy.ondigitalocean.app => NEEDS PAGE CONTENT (99.9%)
  https://paypal.com/signin => WARN (74.2%)
  https://github.io/account => NEEDS PAGE CONTENT (99.9%)
  pasted HTML => 200 BLOCK (99.0%)
  ```
- Full project lint:
  ```text
  .\.venv\Scripts\ruff.exe check src webapp tests
  All checks passed!
  ```
- Full test suite:
  ```text
  .\.venv\Scripts\python.exe -m pytest -q
  79 passed, 13 warnings in 29.57s
  ```
- Compose configuration after removing `.env` interpolation:
  ```text
  docker compose config
  Valid configuration; model directory mounted read-only; service bound to 127.0.0.1:8000.
  ```
- `docker compose build` could not run because the Docker Desktop Linux engine pipe was unavailable. `docker --version` returned client 29.9.2. GitHub Actions was not run remotely; its lint, test, and tiny-fit commands were run locally.
- Syntax check:
  ```text
  .\.venv\Scripts\python.exe -m py_compile src/data/download_dataset.py src/models/smoke_train.py src/serving/model_registry.py
  exit 0
  ```

## Problems and surprises

- The Phase 8 smoke exercise reported 86% probabilities for Google and example URLs. A parity check revealed that Platt calibration received untransformed raw probabilities in serving. The corrected transform yields higher probabilities on those samples, including BLOCK for Google and example.com/login. This is a serious false-positive warning and is documented rather than hidden. The local popular-domain list is context-only and does not suppress model scores.
- Dataset download requires the optional `requirements-data.txt` package and external network access; CI does not download the dataset or touch locked test shards.
- The app-only container cannot train models. It needs the repository's local model artifacts mounted at `/app/models`.
- An initial `docker compose config` resolved the local Safe Browsing key from `.env` into command output. Compose was changed to omit that environment mapping entirely; the key was not copied into project files. Rotate the exposed key before further use.
- Before adding `serving_smoke.py`, one Phase 9 smoke pass called the app with its normal local `.env` loading behavior. The configured Safe Browsing key meant those URLs may have been submitted to Google. This pass is superseded by the offline mocked-reputation check above.

## Risks and open questions

- The selected model still blocks some familiar benign domains in the corrected smoke set. Do not expose this prototype as an automatic enforcement control. A new deployment-representative validation study and possibly a serving-feature parity investigation are needed before operational use; no thresholds were changed in Phase 9.
- The Docker image and Compose definition are local configuration artifacts; Docker verification depends on a Docker engine being available.
- The Docker daemon was unavailable, so the image could not be built or launched. Compose parsing was checked after removing secret interpolation.
- Google Safe Browsing receives complete URLs when configured. URL paths and query strings can contain secrets.
- Rotate the Safe Browsing API key: Compose rendered the local key into command output, and an earlier app smoke pass may have sent the sample URLs to Google. The key value is not included in the report or repository.
- `archive/phase9/` retains the three machine-specific stray files. The absent `Untitled-1.ps1` could not be archived.

## Next

Phase 9 completes the requested project plan. No additional model or threshold change is made here. The final handoff is ready for review.

## Docker follow-up after Phase 9

The Docker daemon became reachable after the initial report. `docker compose config` succeeds without interpolating `.env`, but `docker compose build` still cannot resolve `python:3.9-slim`: the daemon receives an HTTP response to its HTTPS request to Docker Hub. A PowerShell HEAD request to the same registry endpoint returns HTTP 401, the expected unauthenticated registry challenge, so this host can reach the registry while the engine's registry path cannot. The image has not been built or launched; resolve the Docker Desktop/engine registry or proxy configuration and rerun the build.
