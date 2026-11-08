# Phase 4 — Classical gradient-boosted models

## What I did

- Added `src/evaluation/phase4_data.py` to assemble URL v2 and HTML v2 numeric features by SHA-256, with one-to-one coverage checks against the frozen temporal and grouped assignments. The loader reads only `train-*.parquet`; `cleaned_text` is excluded from tree inputs.
- Added `src/evaluation/phase4_boosting.py` and `configs/phase4_gbm.yaml` for CPU LightGBM, XGBoost, CatBoost, a grouped-C logistic reference, and their no-text / `+oof_text_score` variants.
- Optuna studies use seeded TPE, MedianPruner, grouped domain holdout inside temporal training only, a 100k/25k stratified cap, four workers, and a 900-second per-model timeout. SQLite studies and trial CSVs are saved under `experiments/phase4_studies/` and `experiments/`.
- Character TF-IDF + LinearSVC margins are produced by five-fold `GroupKFold` OOF predictions for training rows; each row receives exactly one score. The validation score comes from a model fit only on temporal training rows.
- Final boosted models are refit on all 397,480 temporal training rows with early stopping on the 100,775-row temporal validation partition. Logistic regression is refit on all temporal training rows with its grouped-C choice. The run records model metrics, domain-bootstrap intervals, model size, inference time, and native plus permutation importances.
- Added `tests/test_phase4_data.py` and `tests/test_phase4_boosting.py`. The runner checkpoint is commit `3172538`; the experiment JSON records that code revision.

## Why

Tuning only within the temporal training partition prevents the later validation window from influencing hyperparameter selection. Group separation in the tuning split and all OOF folds prevents shared registered domains from leaking across fit and score rows. Trees receive engineered numeric and categorical URL/HTML features, not sparse TF-IDF matrices; the text model contributes only an explicitly labeled OOF score. The original SVMs remain unchanged as the reference and fallback.

## Results

The full run took **3,027.12 seconds (50 minutes 27 seconds)** on a 12-logical-CPU machine, using four estimator workers. The study caps completed 60 LightGBM trials, 27 XGBoost trials, and 15 CatBoost trials (including each study's one-trial timing pilot). XGBoost and CatBoost hit their 900-second study timeouts before reaching 60 trials; this is reported rather than extrapolated. The grouped AP objective values were 0.98718, 0.98917, and 0.98939, respectively. The four-value logistic C grid selected `C=10` with grouped AP 0.97238.

The five-fold text OOF pass scored all 397,480 training rows exactly once and took 326.77 seconds. No test-set rows were opened. The tuning AP values come from the grouped tuning subset and are not comparable to the later temporal holdout AP values below.

### Temporal validation metrics

Point precision/recall/F1/FPR/accuracy use a fixed score threshold of 0 (probability 0.5 for the boosted models). Intervals are 95% percentile intervals from 200 registered-domain bootstrap replicates. The existing SVM rows are copied from the Phase 2 experiment record; no SVM was retrained in this phase.

| Model | AP [95% CI] | ROC-AUC [95% CI] | Precision [95% CI] | Recall [95% CI] | F1 [95% CI] | FPR [95% CI] | Accuracy [95% CI] |
|---|---|---|---|---|---|---|---|
| SVM URL TF-IDF | 0.8945 [0.8632, 0.9210] | 0.9156 [0.8776, 0.9414] | 0.9616 [0.9573, 0.9655] | 0.6943 [0.6402, 0.7543] | 0.8064 [0.7675, 0.8447] | 0.0109 [0.0098, 0.0123] | 0.9059 [0.8823, 0.9258] |
| SVM URL + engineered | 0.8947 [0.8607, 0.9246] | 0.9100 [0.8725, 0.9405] | 0.9641 [0.9605, 0.9685] | 0.7315 [0.6805, 0.7919] | 0.8319 [0.7968, 0.8692] | 0.0107 [0.0094, 0.0122] | 0.9166 [0.8967, 0.9376] |
| SVM URL + HTML | 0.9054 [0.8800, 0.9293] | 0.9250 [0.8968, 0.9502] | 0.9664 [0.9624, 0.9702] | 0.7342 [0.6885, 0.7872] | 0.8344 [0.8045, 0.8683] | 0.0100 [0.0087, 0.0114] | 0.9178 [0.8991, 0.9357] |
| LightGBM engineered | 0.9192 [0.8929, 0.9405] | 0.9448 [0.9197, 0.9631] | 0.9530 [0.9485, 0.9578] | 0.7668 [0.7092, 0.8187] | 0.8498 [0.8137, 0.8817] | 0.0149 [0.0131, 0.0167] | 0.9235 [0.9012, 0.9394] |
| LightGBM + OOF text score | 0.9183 [0.8889, 0.9399] | 0.9417 [0.9139, 0.9610] | 0.9702 [0.9659, 0.9737] | 0.7113 [0.6537, 0.7736] | 0.8208 [0.7817, 0.8610] | 0.0086 [0.0074, 0.0101] | 0.9124 [0.8885, 0.9324] |
| XGBoost engineered | 0.9210 [0.8918, 0.9434] | 0.9419 [0.9129, 0.9626] | 0.9624 [0.9580, 0.9663] | 0.7812 [0.7276, 0.8332] | 0.8624 [0.8281, 0.8939] | 0.0120 [0.0104, 0.0136] | 0.9296 [0.9083, 0.9450] |
| XGBoost + OOF text score | 0.9129 [0.8802, 0.9356] | 0.9379 [0.9076, 0.9590] | 0.9598 [0.9546, 0.9635] | 0.7460 [0.6882, 0.8023] | 0.8395 [0.8007, 0.8734] | 0.0123 [0.0107, 0.0141] | 0.9195 [0.8957, 0.9367] |
| CatBoost engineered | 0.9408 [0.9167, 0.9606] | 0.9631 [0.9418, 0.9782] | 0.9791 [0.9764, 0.9818] | 0.7758 [0.7187, 0.8293] | 0.8657 [0.8291, 0.8983] | 0.0065 [0.0056, 0.0075] | 0.9321 [0.9103, 0.9478] |
| CatBoost + OOF text score | 0.9411 [0.9166, 0.9595] | 0.9624 [0.9382, 0.9787] | 0.9824 [0.9790, 0.9851] | 0.7478 [0.6906, 0.8039] | 0.8492 [0.8106, 0.8845] | 0.0053 [0.0044, 0.0064] | 0.9251 [0.9034, 0.9441] |
| Logistic engineered | 0.8632 [0.8304, 0.8970] | 0.8893 [0.8539, 0.9185] | 0.8719 [0.8596, 0.8846] | 0.7459 [0.6853, 0.8004] | 0.8040 [0.7684, 0.8366] | 0.0431 [0.0379, 0.0474] | 0.8974 [0.8767, 0.9145] |

### Precision at assumed prevalence

Each cell is precision with its 95% registered-domain bootstrap CI. SVM prevalence results remain fully tabulated in [reports/phase_2_report.md](reports/phase_2_report.md); the table here covers Phase 4 models.

| Model | 0.05% | 0.5% | 1% | 5% | 45% |
|---|---|---|---|---|---|
| LightGBM engineered | 0.0252 [0.0219, 0.0285] | 0.2059 [0.1839, 0.2278] | 0.3426 [0.3118, 0.3723] | 0.7309 [0.7025, 0.7555] | 0.9769 [0.9735, 0.9796] |
| LightGBM + OOF text score | 0.0400 [0.0336, 0.0463] | 0.2940 [0.2588, 0.3278] | 0.4551 [0.4124, 0.4950] | 0.8129 [0.7853, 0.8362] | 0.9847 [0.9827, 0.9876] |
| XGBoost engineered | 0.0317 [0.0273, 0.0359] | 0.2462 [0.2200, 0.2722] | 0.3960 [0.3618, 0.4292] | 0.7735 [0.7471, 0.7966] | 0.9815 [0.9787, 0.9838] |
| XGBoost + OOF text score | 0.0294 [0.0251, 0.0339] | 0.2336 [0.2052, 0.2607] | 0.3800 [0.3416, 0.4148] | 0.7619 [0.7300, 0.7869] | 0.9802 [0.9768, 0.9829] |
| CatBoost engineered | 0.0558 [0.0480, 0.0653] | 0.3750 [0.3364, 0.4122] | 0.5474 [0.5047, 0.5850] | 0.8633 [0.8415, 0.8802] | 0.9899 [0.9880, 0.9913] |
| CatBoost + OOF text score | 0.0660 [0.0549, 0.0795] | 0.4163 [0.3686, 0.4645] | 0.5894 [0.5399, 0.6355] | 0.8817 [0.8594, 0.9008] | 0.9913 [0.9896, 0.9930] |
| Logistic engineered | 0.0087 [0.0076, 0.0098] | 0.0800 [0.0713, 0.0907] | 0.1494 [0.1338, 0.1670] | 0.4772 [0.4459, 0.5108] | 0.9340 [0.9260, 0.9420] |

### Alerts per 1,000 at assumed prevalence

Each cell is alerts per 1,000 with its 95% registered-domain bootstrap CI. These use class-conditional TPR/FPR reweighted to the assumed prevalence.

| Model | 0.05% | 0.5% | 1% | 5% | 45% |
|---|---|---|---|---|---|
| LightGBM engineered | 15.24 [13.42, 17.06] | 18.62 [16.77, 20.58] | 22.38 [20.39, 24.41] | 52.46 [49.20, 55.75] | 353.25 [327.67, 377.39] |
| LightGBM + OOF text score | 8.95 [7.75, 10.43] | 12.11 [10.88, 13.74] | 15.63 [14.26, 17.28] | 43.73 [40.76, 47.14] | 324.83 [298.49, 352.89] |
| XGBoost engineered | 12.40 [10.84, 13.95] | 15.86 [14.40, 17.47] | 19.70 [18.04, 21.42] | 50.47 [47.26, 53.88] | 358.15 [333.39, 382.01] |
| XGBoost + OOF text score | 12.66 [11.11, 14.43] | 15.96 [14.38, 17.83] | 19.63 [17.96, 21.63] | 48.98 [45.82, 52.30] | 342.48 [316.46, 367.96] |
| CatBoost engineered | 6.88 [5.96, 7.89] | 10.34 [9.30, 11.44] | 14.19 [13.00, 15.48] | 44.96 [42.08, 47.98] | 352.67 [326.67, 377.04] |
| CatBoost + OOF text score | 5.64 [4.74, 6.74] | 8.98 [8.13, 10.09] | 12.69 [11.63, 13.91] | 42.39 [39.54, 45.45] | 339.41 [313.96, 364.80] |
| Logistic engineered | 43.41 [38.25, 47.80] | 46.57 [41.36, 51.06] | 50.09 [45.00, 54.43] | 78.20 [72.60, 83.74] | 359.36 [330.58, 383.27] |

The CatBoost variants have the strongest point AP on the temporal holdout. However, the CatBoost AP CIs overlap the SVM URL+HTML interval (0.8800–0.9293), as do the other GBM intervals. Therefore this run does **not** establish that a GBM beats the SVM beyond the reported domain-bootstrap intervals. We did not compute a paired bootstrap of model AP differences. The OOF text score reduced AP for LightGBM and XGBoost and changed CatBoost AP by only +0.0004; the overlapping intervals do not support a reliable AP gain from the score in this phase.

Boosted models used categorical suffix/domain features natively. Native importance is LightGBM gain, XGBoost gain, and CatBoost `PredictionValuesChange` (CatBoost does not expose the same gain definition); all models also have permutation importance measured on a seeded 5,000-row validation sample with three repeats. Complete top-30 importance lists, timing, sizes, parameters, and all metric intervals are in `experiments/phase4_gbm_results.json`.

## Verification

- Data assembly preflight:
  ```text
  train_source_rows=498255 temporal_ids=498255 cache_rows=498255 distinct_cache_ids=498255 split_ids_missing_from_cache=0 logical_cpus=12
  temporal_train=397480 validation=100775 purged=0
  ```
- Runtime pilot, committed config:
  ```text
  .\.venv\Scripts\python.exe -m src.evaluation.phase4_boosting --pilot
  lightgbm pilot: AP=0.984637; trial_seconds=3.6
  xgboost pilot: AP=0.986589; trial_seconds=22.0
  catboost pilot: AP=0.987969; trial_seconds=97.2
  Pilot wall seconds: 164.98
  ```
  Pilot AP values are a runtime check only, not model selection.
- Full experiment:
  ```text
  .\.venv\Scripts\python.exe -u -m src.evaluation.phase4_boosting
  lightgbm: best grouped AP=0.987178; elapsed=213.6s
  xgboost: best grouped AP=0.989173; elapsed=910.8s
  catboost: best grouped AP=0.989393; elapsed=904.4s
  Wrote experiment record: experiments/phase4_gbm_results.json
  Total wall seconds: 3027.12
  ```
- Full test suite and static checks before run:
  ```text
  .\.venv\Scripts\python.exe -m pytest -q
  35 passed in 33.03s
  .\.venv\Scripts\ruff.exe check src/evaluation/phase4_data.py src/evaluation/phase4_boosting.py tests/test_phase4_data.py tests/test_phase4_boosting.py
  All checks passed!
  .\.venv\Scripts\python.exe -m py_compile src/evaluation/phase4_data.py src/evaluation/phase4_boosting.py
  [exit 0]
  ```
- All seven model artifact files exist and their actual byte sizes match the experiment record. The test suite also checks synthetic determinism, serialization round trips, single/batch prediction parity, SHA joins, domain-disjoint tuning, and exactly-once OOF alignment.

## Problems and surprises

- The configured 60-trial limit was reached only for LightGBM. XGBoost and CatBoost stopped at their hard timeouts after 27 and 15 total trials, respectively. Their studies remain saved in SQLite and can resume, but this phase reports the observed budget-limited results.
- Grouped tuning AP is higher than later temporal validation AP for all boosters. The held-out later-time window remains the more conservative estimate for comparison with the Phase 2 temporal SVMs.
- The OOF score decreased AP for two of three GBMs and did not materially change CatBoost AP. It does reduce false-positive rate at the fixed 0.5 threshold for some variants, with a corresponding recall tradeoff.
- XGBoost artifacts are large (340,287,766 and 227,092,645 bytes). Generated model weights remain ignored by Git; the experiment record reports their local paths and sizes. The existing SVM files were not changed.

## Risks and open questions

- Marginal CIs overlap; paired bootstrap differences are needed before claiming statistically supported AP improvements over the SVM. The model-selection comparison does not justify replacing the fallback.
- Feature importance depends on correlated URL/HTML features and is not causal. CatBoost importance uses a different native definition than LightGBM/XGBoost gain.
- Refits early-stopped on the same temporal validation partition used for final metric reporting, as required by the phase plan. This makes the reported validation metrics model-selection estimates, not an untouched final test estimate.
- The tuning studies use up to 100,000 fit and 25,000 domain-held-out rows; estimates may shift on a different group sample. No official test files were accessed.

## Next phase

Phase 5: compare classical URL character and word-level text models, HashingVectorizer, linear/SGD/naive-Bayes alternatives, and light tree references using the same blocked protocol. Continue to generate strictly grouped OOF scores and keep the official test set locked. No further Phase 4 model selection is proposed from these results alone.
