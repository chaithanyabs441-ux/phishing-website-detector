# Phase 2 — Blocked splits and leakage audit

## What I did

- Added `src/evaluation/create_blocked_splits.py`. It reads **only** `data/raw/data/train-*.parquet`, extracts registered domains through the shared offline PSL helper, and writes frozen temporal and grouped assignments by SHA-256.
- Added `data/splits/temporal_assignments.csv.gz`, `data/splits/grouped_domain_folds.csv.gz`, `data/splits/leakage_audit.json`, and `data/splits/MANIFEST.json`. The two compressed assignment files use fixed gzip metadata; rerunning the split generator produced identical SHA-256 hashes.
- Added `configs/splits.yaml` and `configs/svm_blocked_baseline.yaml` to record split and model settings.
- Added `src/evaluation/evaluate_svm_blocked.py` and `src/evaluation/evaluate_svm_grouped_cv.py`. Both refit the three existing SVM designs in memory, using training folds only for vectorizers and scalers. Predictions from the grouped run are out-of-fold: each row is scored once by a model whose training data excludes its registered domain. No saved model was changed or replaced.
- Added `tests/test_blocked_splits.py` for temporal ordering, SHA assignment, boundary purging, deterministic domain assignment, and zero domain overlap in grouped folds.
- Added small `.gitignore` exceptions for `data/splits/` and the two experiment JSON records, while raw/processed data and other experiment outputs remain ignored.

## Why

The random row split can put the same domain, URL patterns, and adjacent collection dates on both sides. The temporal holdout measures performance on later collection dates; GroupKFold measures performance on domains absent from the corresponding training fold. The temporal split purges any SHA-256 that would cross the cutoff, so one ID cannot appear in both sides.

I did not deduplicate training rows. The audit found zero exact duplicate URLs and zero URL label conflicts, while 46,764 rows share a normalized host/path template. That template count is a useful near-duplicate warning, but it is too broad to justify silently deleting data. Grouped validation directly tests generalization to unseen domains.

The instructions also requested train/test overlap measurement. I deferred it because the standing test-lock rule forbids reading official `test-*.parquet` before Phase 7. No official test file was opened, listed, or modified.

## Split and audit results

Training source: 56 `train-*.parquet` shards, 498,255 rows, 498,255 unique SHA-256 values, dates 2024-07-02 through 2025-09-08 UTC. The temporal cutoff is 2025-07-10 UTC.

The machine has 12 logical CPUs (recorded in Phase 0). These LinearSVC fits ran sequentially; the vectorizer/scalers were fit per training partition, with no cross-fold parallel jobs.

| Temporal partition | Rows | Benign | Phish | Date range |
|---|---:|---:|---:|---|
| Train | 397,480 | 204,388 | 193,092 | 2024-07-02 to 2025-07-09 |
| Validation | 100,775 | 72,341 | 28,434 | 2025-07-10 to 2025-09-08 |

Phish prevalence changes from 48.6% in temporal training to 28.2% in temporal validation. The later data therefore has a different class balance. There are 14,182 registered domains shared across the temporal train and validation periods; this is why the separate grouped result is also reported.

| Grouped validation fold | Rows | Benign | Phish | Registered domains | Train/validation domain overlap |
|---:|---:|---:|---:|---:|---:|
| 0 | 99,651 | 48,778 | 50,873 | 34,120 | 0 |
| 1 | 99,651 | 51,024 | 48,627 | 34,120 | 0 |
| 2 | 99,651 | 55,467 | 44,184 | 34,123 | 0 |
| 3 | 99,651 | 67,027 | 32,624 | 34,124 | 0 |
| 4 | 99,651 | 54,433 | 45,218 | 34,123 | 0 |

Each grouped fold spans the full date range. The fold validation sizes are even, but their label balance differs because the split is grouped by domain, not stratified by row.

Other train-only audit counts: **0** exact duplicate URL rows; **0** URLs with conflicting labels; **46,764** rows beyond the first occurrence of a normalized host+path template (**9.39%**); **0** SHA/domain inconsistencies. The near-duplicate template rate is a proxy and can include ordinary repeated page patterns.

## Model evaluation

All point metrics below use a fixed decision threshold of **0.0**. No threshold was selected or tuned. Precision/alerts at the requested prevalences are calculated by reweighting the measured class-conditional true-positive and false-positive rates. Confidence intervals are 95% percentile intervals from 200 bootstrap replicates resampling registered domains with replacement.

### Temporal holdout

| Model | AP (95% CI) | ROC-AUC (95% CI) | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | FPR (95% CI) | Accuracy (95% CI) |
|---|---|---|---|---|---|---|---|
| URL TF-IDF | 0.8945 [0.8632, 0.9210] | 0.9156 [0.8776, 0.9414] | 0.9616 [0.9573, 0.9655] | 0.6943 [0.6402, 0.7543] | 0.8064 [0.7675, 0.8447] | 0.0109 [0.0098, 0.0123] | 0.9059 [0.8823, 0.9258] |
| URL + engineered | 0.8947 [0.8607, 0.9246] | 0.9100 [0.8725, 0.9405] | 0.9641 [0.9605, 0.9685] | 0.7315 [0.6805, 0.7919] | 0.8319 [0.7968, 0.8692] | 0.0107 [0.0094, 0.0122] | 0.9166 [0.8967, 0.9376] |
| URL + HTML | 0.9054 [0.8800, 0.9293] | 0.9250 [0.8968, 0.9502] | 0.9664 [0.9624, 0.9702] | 0.7342 [0.6885, 0.7872] | 0.8344 [0.8045, 0.8683] | 0.0100 [0.0087, 0.0114] | 0.9178 [0.8991, 0.9357] |

| Model | Phish prevalence | Precision (95% CI) | Alerts / 1,000 (95% CI) |
|---|---:|---|---|
| URL TF-IDF | 0.05% | 0.0309 [0.0267, 0.0350] | 11.25 [10.16, 12.66] |
| URL TF-IDF | 0.5% | 0.2424 [0.2161, 0.2671] | 14.32 [13.12, 15.73] |
| URL TF-IDF | 1% | 0.3914 [0.3565, 0.4229] | 17.74 [16.38, 19.26] |
| URL TF-IDF | 5% | 0.7701 [0.7427, 0.7924] | 45.08 [42.05, 48.54] |
| URL TF-IDF | 45% | 0.9812 [0.9782, 0.9834] | 318.44 [293.62, 345.77] |
| URL + engineered | 0.05% | 0.0330 [0.0286, 0.0377] | 11.07 [9.81, 12.53] |
| URL + engineered | 0.5% | 0.2555 [0.2282, 0.2824] | 14.32 [13.10, 15.72] |
| URL + engineered | 1% | 0.4082 [0.3728, 0.4417] | 17.92 [16.56, 19.37] |
| URL + engineered | 5% | 0.7823 [0.7559, 0.8048] | 46.75 [43.62, 49.65] |
| URL + engineered | 45% | 0.9824 [0.9796, 0.9846] | 335.08 [311.89, 362.17] |
| URL + HTML | 0.05% | 0.0354 [0.0300, 0.0411] | 10.38 [9.10, 11.74] |
| URL + HTML | 0.5% | 0.2691 [0.2368, 0.3007] | 13.64 [12.40, 14.96] |
| URL + HTML | 1% | 0.4253 [0.3841, 0.4637] | 17.26 [15.98, 18.69] |
| URL + HTML | 5% | 0.7940 [0.7647, 0.8183] | 46.23 [43.71, 49.26] |
| URL + HTML | 45% | 0.9836 [0.9806, 0.9859] | 335.88 [315.92, 359.56] |

### Registered-domain grouped 5-fold OOF

| Model | AP (95% CI) | ROC-AUC (95% CI) | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | FPR (95% CI) | Accuracy (95% CI) |
|---|---|---|---|---|---|---|---|
| URL TF-IDF | 0.9825 [0.9679, 0.9917] | 0.9851 [0.9730, 0.9932] | 0.9589 [0.9223, 0.9783] | 0.8878 [0.8162, 0.9401] | 0.9220 [0.8817, 0.9539] | 0.0304 [0.0167, 0.0503] | 0.9332 [0.8976, 0.9600] |
| URL + engineered | 0.9845 [0.9660, 0.9941] | 0.9856 [0.9693, 0.9948] | 0.9739 [0.9605, 0.9820] | 0.9040 [0.8275, 0.9494] | 0.9376 [0.8942, 0.9635] | 0.0194 [0.0147, 0.0265] | 0.9465 [0.9139, 0.9676] |
| URL + HTML | 0.9827 [0.9657, 0.9947] | 0.9824 [0.9612, 0.9957] | 0.9755 [0.9643, 0.9823] | 0.9095 [0.8491, 0.9554] | 0.9413 [0.9082, 0.9661] | 0.0183 [0.0152, 0.0235] | 0.9496 [0.9199, 0.9706] |

| Model | Phish prevalence | Precision (95% CI) | Alerts / 1,000 (95% CI) |
|---|---:|---|---|
| URL TF-IDF | 0.05% | 0.0144 [0.0088, 0.0256] | 30.86 [17.12, 50.72] |
| URL TF-IDF | 0.5% | 0.1279 [0.0816, 0.2087] | 34.72 [21.11, 54.26] |
| URL TF-IDF | 1% | 0.2276 [0.1515, 0.3464] | 39.00 [25.58, 58.21] |
| URL TF-IDF | 5% | 0.6056 [0.4820, 0.7342] | 73.30 [60.23, 92.20] |
| URL TF-IDF | 45% | 0.9598 [0.9353, 0.9772] | 416.23 [381.79, 442.31] |
| URL + engineered | 0.05% | 0.0228 [0.0168, 0.0304] | 19.84 [15.13, 26.94] |
| URL + engineered | 0.5% | 0.1897 [0.1463, 0.2394] | 23.82 [19.25, 30.97] |
| URL + engineered | 1% | 0.3200 [0.2562, 0.3875] | 28.25 [23.81, 35.50] |
| URL + engineered | 5% | 0.7103 [0.6422, 0.7672] | 63.63 [58.39, 71.09] |
| URL + engineered | 45% | 0.9744 [0.9654, 0.9809] | 417.47 [385.28, 438.59] |
| URL + HTML | 0.05% | 0.0242 [0.0189, 0.0295] | 18.76 [15.63, 23.91] |
| URL + HTML | 0.5% | 0.1997 [0.1622, 0.2340] | 22.77 [19.61, 27.90] |
| URL + HTML | 1% | 0.3340 [0.2802, 0.3805] | 27.23 [24.05, 32.33] |
| URL + HTML | 5% | 0.7232 [0.6698, 0.7619] | 62.88 [58.88, 68.66] |
| URL + HTML | 45% | 0.9760 [0.9693, 0.9803] | 419.36 [393.33, 439.51] |

The random-validation figures previously supplied in the project prompt (accuracy 0.9856 / AP 0.9981) were not independently verified here. Grouped CV AP is around 0.983, while the later temporal window is around 0.895–0.905 AP and has a lower phishing prevalence. This shows that the estimate depends strongly on split protocol and time period. It is consistent with random-split optimism and temporal drift; these results do not isolate the contribution of each effect. The models' AP confidence intervals overlap substantially, so Phase 2 does not establish a reliable AP winner among the three.

## Verification

- Split generation, run twice with deterministic gzip output:

  ```text
  .\.venv\Scripts\python.exe -m src.evaluation.create_blocked_splits
  Training rows: 498255; unique SHA-256: 498255
  Temporal cutoff UTC: 2025-07-10T00:00:00+00:00
  Temporal train/validation/purged: 397480/100775/0 rows
  Temporal registered-domain overlap: 14182
  Exact duplicate URL rows: 0; label-conflict URLs: 0
  Near-duplicate host/path template rows: 46764
  Wrote splits: temporal_assignments.csv.gz, grouped_domain_folds.csv.gz
  Elapsed seconds: 22.35
  ```

  The two compressed split files had identical SHA-256 hashes on the repeat run. All split and audit hashes in `data/splits/MANIFEST.json` verified.

- Automated suite:

  ```text
  .\.venv\Scripts\python.exe -m pytest -q
  22 passed in 28.19s
  ```

- Changed-file lint and syntax checks:

  ```text
  .\.venv\Scripts\ruff.exe check src/evaluation/create_blocked_splits.py src/evaluation/evaluate_svm_blocked.py src/evaluation/evaluate_svm_grouped_cv.py src/features/url_features.py tests/test_blocked_splits.py
  All checks passed!

  .\.venv\Scripts\python.exe -m py_compile src/evaluation/create_blocked_splits.py src/evaluation/evaluate_svm_blocked.py src/evaluation/evaluate_svm_grouped_cv.py src/features/url_features.py
  [exit 0]
  ```

- Temporal refit and grouped OOF commands:

  ```text
  .\.venv\Scripts\python.exe -m src.evaluation.evaluate_svm_blocked --bootstrap-replicates 200 --seed 42
  url_tfidf: AP=0.894493, ROC-AUC=0.915554, precision=0.961570, recall=0.694310, F1=0.806372, FPR=0.010907
  url_engineered: AP=0.894726, ROC-AUC=0.909989, precision=0.964079, recall=0.731519, F1=0.831850, FPR=0.010713
  url_html: AP=0.905383, ROC-AUC=0.925022, precision=0.966435, recall=0.734156, F1=0.834433, FPR=0.010022
  Vectorizer fit/transform seconds: 62.53
  Total wall seconds: 568.30

  .\.venv\Scripts\python.exe -u -m src.evaluation.evaluate_svm_grouped_cv --bootstrap-replicates 200 --seed 42
  url_tfidf grouped OOF: AP=0.982513, ROC-AUC=0.985088, precision=0.958939, recall=0.887769, F1=0.921983, FPR=0.030430
  url_engineered grouped OOF: AP=0.984530, ROC-AUC=0.985552, precision=0.973890, recall=0.904002, F1=0.937646, FPR=0.019402
  url_html grouped OOF: AP=0.982713, ROC-AUC=0.982372, precision=0.975459, recall=0.909519, F1=0.941335, FPR=0.018318
  Convergence warnings: 10
  Grouped CV wall seconds: 3147.75
  ```

  The grouped run produced 498,255 OOF scores per model. No trained classifier or scaler was saved. The grouped run was restarted after an initial attempt was stopped at fold 3 when its observed pace exceeded the original 60-minute estimate; approval was then received to finish all five folds. The completed run used per-fold checkpoints and took 52 minutes 27.75 seconds.

- Verified every shipped model artifact against the Phase 1 model manifest after evaluation: all hashes remain valid and unchanged. Both experiment records declare `official_test_accessed: false` and their split-file hashes match the current manifest.

## Problems and surprises

- Temporal validation is substantially harder than random/grouped folds: URL+HTML AP is 0.905 temporally versus 0.983 grouped OOF. The phishing proportion drops to 28.2% in the later period from 48.6% in the earlier training period.
- The URL+engineered and URL+HTML LinearSVC fits hit their configured 5,000-iteration cap in every grouped fold (10 convergence warnings total) and in the temporal run (2 warnings). Results are the observed fixed-config scores; I did not tune on validation. Treat combined-model point estimates as less settled until convergence settings are addressed using training-only evidence.
- The first grouped run was stopped after fold 3 because runtime projected beyond the phase's 60-minute budget. After explicit approval, the full run completed with checkpointing. The interrupted run wrote no experiment results; the completed rerun is the one recorded above.
- The official test overlap audit remains deferred to preserve the Phase 7-only test lock.

## Risks and open questions

- The app's URL-only threshold of 1.5 was not evaluated or selected here. All threshold-dependent figures use 0.0; threshold selection remains Phase 7 work.
- Grouped-fold estimates are strong but the fold labels and row balance vary by domain. The temporal holdout captures a major distribution shift; neither estimate alone should be presented as a universal production rate.
- The grouped SVM fit warnings merit a training-only convergence investigation before later model comparisons. No hyperparameters were changed in this phase.
- Exact URL deduplication is not indicated by this audit; whether to deduplicate the near-template groups should be tested as a separate training-only experiment, not applied silently.
- `target` was not used in this phase. Its potential use in later experiments needs a leakage review.

## Next phase

Phase 3: build a versioned, bounded HTML v2 feature cache from stored training HTML, including missing-HTML indicators and the requested structural/text features. Continue to use training shards only; official test access remains locked until Phase 7.
