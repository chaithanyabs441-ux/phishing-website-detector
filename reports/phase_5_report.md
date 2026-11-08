# Phase 5 — Classical text models and hybrids

## What I did

- Added `src/evaluation/phase5_text.py` with classical URL character TF-IDF and hashing models, word-level TF-IDF on Phase 3 cleaned HTML text, LinearSVC, LogisticRegression (liblinear/saga), SGD (modified-huber/log-loss), Complement/Multinomial NB, and NB-SVM.
- Added `src/evaluation/evaluate_phase5_text.py` for grouped runtime pilots, temporal validation comparisons, domain-grouped bootstrap intervals, and the CatBoost + OOF text-score hybrid.
- Added `src/evaluation/phase5_trees.py` for Optuna-tuned ExtraTrees and HistGradientBoosting references on numeric engineered features. Registered-domain identity and string categories are excluded from these tree inputs.
- Added `configs/phase5_text.yaml`, `tests/test_phase5_text.py`, and `tests/test_phase5_trees.py`. Tests cover train-only vocabulary fitting, exactly-once grouped OOF alignment, domain separation, label-shuffle chance behavior, model serialization/single-vs-batch parity, shared sparse-matrix reuse, paired bootstrap, and deterministic tree predictions.
- Experiment records, trial tables, and resumable SQLite studies are in `experiments/phase5_*.json`, `experiments/phase5_*_trials.csv`, and `experiments/phase5_studies/`. Model artifacts remain local under `models/phase5/` and are ignored by Git.

## Why

All model selection and validation use the frozen Phase 2 temporal split, while OOF scores use registered-domain `GroupKFold` within temporal training only. Each OOF row receives exactly one score from a model and vectorizer that did not fit on that row or its domain. Tree models use engineered numeric features; sparse TF-IDF is used only by classical text models and enters the CatBoost hybrid as one OOF score. Sparse matrices are shared between compatible classifiers to reduce repeated transforms and peak memory.

## Results

The Phase 5 text comparison took **1,299.73 seconds (21m 40s)**; the two tree references took **285.28 seconds (4m 45s)** including tuning, refits, and evaluation. Both used the 397,480-row temporal training split and 100,775-row later validation split. All intervals below are 95% percentile bootstrap intervals resampling registered domains (200 requested replicates). The decision threshold is 0 on model scores (0.5 probability for probabilistic models). Accuracy is included for comparison, not as the selection objective.

| Model | AP [95% CI] | ROC-AUC [95% CI] | Precision [95% CI] | Recall [95% CI] | F1 [95% CI] | FPR [95% CI] | Accuracy [95% CI] |
|---|---|---|---|---|---|---|---|
| SVM URL + HTML reference | 0.9054 [0.8800, 0.9293] | 0.9250 [0.8968, 0.9502] | 0.9664 [0.9624, 0.9702] | 0.7342 [0.6885, 0.7872] | 0.8344 [0.8045, 0.8683] | 0.0100 [0.0087, 0.0114] | 0.9178 [0.8991, 0.9357] |
| CatBoost engineered reference | 0.9408 [0.9167, 0.9606] | 0.9631 [0.9418, 0.9782] | 0.9791 [0.9764, 0.9818] | 0.7758 [0.7187, 0.8293] | 0.8657 [0.8291, 0.8983] | 0.0065 [0.0056, 0.0075] | 0.9321 [0.9103, 0.9478] |
| Char SVC 2–5 | 0.8945 [0.8632, 0.9210] | 0.9156 [0.8776, 0.9414] | 0.9616 [0.9573, 0.9655] | 0.6943 [0.6402, 0.7543] | 0.8064 [0.7675, 0.8447] | 0.0109 [0.0098, 0.0123] | 0.9059 [0.8823, 0.9258] |
| Char SVC 3–6 | 0.8938 [0.8645, 0.9237] | 0.9155 [0.8826, 0.9427] | 0.9602 [0.9563, 0.9645] | 0.6856 [0.6370, 0.7474] | 0.7999 [0.7655, 0.8407] | 0.0112 [0.0098, 0.0126] | 0.9032 [0.8822, 0.9247] |
| LogisticRegression liblinear | 0.8911 [0.8651, 0.9176] | 0.9188 [0.8904, 0.9451] | 0.9468 [0.9414, 0.9518] | 0.7148 [0.6713, 0.7606] | 0.8146 [0.7857, 0.8437] | 0.0158 [0.0138, 0.0174] | 0.9082 [0.8873, 0.9258] |
| LogisticRegression saga | 0.8911 [0.8627, 0.9186] | 0.9188 [0.8867, 0.9451] | 0.9468 [0.9412, 0.9515] | 0.7147 [0.6684, 0.7696] | 0.8145 [0.7832, 0.8504] | 0.0158 [0.0143, 0.0175] | 0.9082 [0.8866, 0.9274] |
| SGD modified-huber | 0.8815 [0.8472, 0.9131] | 0.9116 [0.8772, 0.9426] | 0.9324 [0.9265, 0.9394] | 0.7150 [0.6656, 0.7731] | 0.8093 [0.7754, 0.8457] | 0.0204 [0.0184, 0.0225] | 0.9049 [0.8867, 0.9248] |
| SGD log-loss | 0.8371 [0.8051, 0.8646] | 0.8965 [0.8540, 0.9245] | 0.8568 [0.8442, 0.8701] | 0.6747 [0.6211, 0.7320] | 0.7549 [0.7181, 0.7921] | 0.0443 [0.0399, 0.0490] | 0.8764 [0.8514, 0.8954] |
| ComplementNB | 0.8606 [0.8296, 0.8910] | 0.9007 [0.8692, 0.9336] | 0.9354 [0.9249, 0.9435] | 0.6184 [0.5717, 0.6779] | 0.7446 [0.7083, 0.7866] | 0.0168 [0.0144, 0.0192] | 0.8803 [0.8588, 0.9044] |
| MultinomialNB | 0.8606 [0.8318, 0.8908] | 0.9007 [0.8663, 0.9296] | 0.9363 [0.9276, 0.9428] | 0.6161 [0.5623, 0.6693] | 0.7432 [0.7019, 0.7804] | 0.0165 [0.0146, 0.0192] | 0.8799 [0.8546, 0.8997] |
| NB-SVM | 0.8880 [0.8522, 0.9166] | 0.9027 [0.8607, 0.9351] | 0.9583 [0.9541, 0.9629] | 0.6998 [0.6302, 0.7483] | 0.8089 [0.7599, 0.8410] | 0.0120 [0.0106, 0.0134] | 0.9067 [0.8796, 0.9240] |
| HTML word SVC | 0.8834 [0.8644, 0.9037] | 0.9327 [0.9191, 0.9444] | 0.8610 [0.8186, 0.8972] | 0.6792 [0.6387, 0.7287] | 0.7594 [0.7257, 0.7931] | 0.0431 [0.0309, 0.0587] | 0.8785 [0.8589, 0.8995] |
| Char SVC min_df=1 | 0.8944 [0.8639, 0.9214] | 0.9154 [0.8819, 0.9442] | 0.9614 [0.9572, 0.9651] | 0.6946 [0.6358, 0.7427] | 0.8065 [0.7653, 0.8385] | 0.0109 [0.0096, 0.0122] | 0.9060 [0.8850, 0.9230] |
| Char SVC min_df=3 | 0.8946 [0.8658, 0.9228] | 0.9157 [0.8825, 0.9421] | 0.9615 [0.9575, 0.9661] | 0.6942 [0.6455, 0.7531] | 0.8063 [0.7722, 0.8460] | 0.0109 [0.0098, 0.0121] | 0.9059 [0.8833, 0.9252] |
| ExtraTrees | 0.8932 [0.8644, 0.9178] | 0.9303 [0.9031, 0.9510] | 0.9355 [0.9274, 0.9418] | 0.7088 [0.6468, 0.7622] | 0.8065 [0.7629, 0.8417] | 0.0192 [0.0173, 0.0216] | 0.9040 [0.8805, 0.9213] |
| HistGradientBoosting | 0.9034 [0.8714, 0.9293] | 0.9235 [0.8886, 0.9489] | 0.9489 [0.9435, 0.9542] | 0.7503 [0.6958, 0.8077] | 0.8380 [0.8014, 0.8735] | 0.0159 [0.0142, 0.0178] | 0.9182 [0.8970, 0.9371] |
| CatBoost + best text OOF score | 0.9399 [0.9145, 0.9595] | 0.9612 [0.9347, 0.9785] | 0.9826 [0.9797, 0.9852] | 0.7532 [0.6956, 0.8080] | 0.8527 [0.8143, 0.8873] | 0.0052 [0.0045, 0.0062] | 0.9266 [0.9049, 0.9449] |

### Precision and alerts by prevalence

Each cell contains `precision [95% CI] / alerts per 1,000 [95% CI]`, reweighted from validation class-conditional rates. Full equivalent values for the Phase 2 SVM and Phase 4 CatBoost baselines are in their respective phase reports.

| Model | 0.05% | 0.5% | 1% | 5% | 45% |
|---|---|---|---|---|---|
| Char SVC 2–5 | 0.0309 [0.0267,0.0350] / 11.25 [10.16,12.66] | 0.2424 [0.2161,0.2671] / 14.32 [13.12,15.73] | 0.3914 [0.3565,0.4229] / 17.74 [16.38,19.26] | 0.7701 [0.7427,0.7924] / 45.08 [42.05,48.54] | 0.9812 [0.9782,0.9834] / 318.44 [293.62,345.77] |
| Char SVC 3–6 | 0.0298 [0.0257,0.0342] / 11.52 [10.18,12.92] | 0.2355 [0.2095,0.2625] / 14.55 [13.27,15.86] | 0.3824 [0.3476,0.4171] / 17.93 [16.59,19.29] | 0.7634 [0.7352,0.7885] / 44.90 [41.90,47.90] | 0.9805 [0.9774,0.9830] / 314.65 [292.54,342.29] |
| Logistic liblinear | 0.0221 [0.0191,0.0256] / 16.15 [14.11,17.77] | 0.1852 [0.1637,0.2089] / 19.30 [17.45,20.89] | 0.3137 [0.2824,0.3468] / 22.79 [20.95,24.41] | 0.7042 [0.6722,0.7345] / 50.75 [48.28,53.80] | 0.9737 [0.9696,0.9773] / 330.37 [311.41,351.40] |
| Logistic saga | 0.0221 [0.0194,0.0248] / 16.15 [14.60,17.89] | 0.1852 [0.1661,0.2037] / 19.29 [17.66,21.06] | 0.3136 [0.2858,0.3396] / 22.79 [21.04,24.59] | 0.7042 [0.6759,0.7282] / 50.75 [47.90,54.42] | 0.9737 [0.9701,0.9766] / 330.31 [309.36,355.14] |
| SGD modified-huber | 0.0172 [0.0153,0.0202] / 20.74 [18.72,22.81] | 0.1498 [0.1350,0.1716] / 23.86 [21.72,25.99] | 0.2616 [0.2388,0.2940] / 27.34 [25.07,29.53] | 0.6486 [0.6204,0.6845] / 55.12 [51.83,58.28] | 0.9663 [0.9621,0.9712] / 332.94 [311.28,359.67] |
| SGD log-loss | 0.0076 [0.0066,0.0085] / 44.63 [40.20,49.26] | 0.0711 [0.0621,0.0794] / 47.47 [42.95,51.98] | 0.1333 [0.1175,0.1478] / 50.62 [46.02,55.34] | 0.4448 [0.4096,0.4747] / 75.83 [70.79,81.46] | 0.9257 [0.9152,0.9335] / 327.97 [303.96,353.64] |
| ComplementNB | 0.0181 [0.0157,0.0213] / 17.10 [14.73,19.52] | 0.1561 [0.1379,0.1792] / 19.80 [17.40,22.20] | 0.2711 [0.2433,0.3050] / 22.81 [20.39,25.19] | 0.6596 [0.6262,0.6958] / 46.88 [43.46,50.37] | 0.9679 [0.9630,0.9726] / 287.52 [266.61,314.66] |
| MultinomialNB | 0.0184 [0.0155,0.0210] / 16.78 [14.94,19.53] | 0.1582 [0.1368,0.1771] / 19.48 [17.62,22.34] | 0.2741 [0.2416,0.3020] / 22.47 [20.42,25.33] | 0.6631 [0.6241,0.6927] / 46.46 [43.05,50.17] | 0.9683 [0.9627,0.9723] / 286.30 [262.69,310.94] |
| NB-SVM | 0.0284 [0.0249,0.0324] / 12.30 [10.94,13.75] | 0.2273 [0.2042,0.2519] / 15.40 [14.15,16.80] | 0.3715 [0.3403,0.4036] / 18.84 [17.60,20.32] | 0.7549 [0.7289,0.7791] / 46.35 [43.61,48.86] | 0.9795 [0.9766,0.9821] / 321.47 [290.22,343.57] |
| HTML word SVC | 0.0078 [0.0058,0.0108] / 43.43 [31.20,59.02] | 0.0734 [0.0550,0.0986] / 46.30 [34.10,61.75] | 0.1373 [0.1047,0.1802] / 49.48 [37.31,64.78] | 0.4533 [0.3786,0.5338] / 74.92 [62.48,89.04] | 0.9280 [0.9045,0.9468] / 329.36 [310.87,355.07] |
| Char SVC min_df=1 | 0.0308 [0.0268,0.0354] / 11.29 [9.91,12.54] | 0.2418 [0.2167,0.2695] / 14.37 [12.89,15.66] | 0.3906 [0.3573,0.4258] / 17.78 [16.15,19.14] | 0.7695 [0.7434,0.7944] / 45.13 [41.54,47.77] | 0.9811 [0.9783,0.9836] / 318.60 [292.34,339.88] |
| Char SVC min_df=3 | 0.0308 [0.0269,0.0355] / 11.28 [10.09,12.41] | 0.2419 [0.2175,0.2698] / 14.35 [13.05,15.56] | 0.3907 [0.3585,0.4262] / 17.77 [16.39,19.19] | 0.7697 [0.7443,0.7947] / 45.10 [42.07,48.00] | 0.9811 [0.9784,0.9836] / 318.42 [296.22,344.54] |
| ExtraTrees | 0.0181 [0.0156,0.0203] / 19.55 [17.67,21.92] | 0.1565 [0.1372,0.1721] / 22.65 [20.67,24.91] | 0.2716 [0.2421,0.2947] / 26.10 [23.93,28.37] | 0.6602 [0.6247,0.6853] / 53.68 [50.32,57.51] | 0.9680 [0.9628,0.9713] / 329.50 [302.90,353.84] |
| HistGradientBoosting | 0.0231 [0.0201,0.0265] / 16.24 [14.59,18.18] | 0.1920 [0.1711,0.2144] / 19.54 [17.88,21.44] | 0.3232 [0.2933,0.3543] / 23.21 [21.41,25.10] | 0.7133 [0.6838,0.7409] / 52.59 [49.01,55.83] | 0.9748 [0.9711,0.9780] / 346.38 [321.86,371.92] |
| CatBoost + OOF text score | 0.0671 [0.0556,0.0799] / 5.61 [4.86,6.62] | 0.4194 [0.3714,0.4661] / 8.98 [8.09,10.03] | 0.5922 [0.5429,0.6370] / 12.72 [11.71,13.88] | 0.8833 [0.8609,0.9014] / 42.64 [39.70,45.61] | 0.9916 [0.9897,0.9930] / 341.81 [315.97,366.54] |

The full results JSONs also include each model's threshold metrics, all prevalence intervals, artifact path/size, transform and fit time, and exact seed/split/config hashes: [experiments/phase5_text_results.json](experiments/phase5_text_results.json) and [experiments/phase5_tree_results.json](experiments/phase5_tree_results.json).

The selected text representation by temporal AP was character 2–5 TF-IDF with `min_df=3` (0.89461). It is effectively tied with the existing URL-char SVM baseline (0.89449; intervals overlap). The HTML word SVC had higher ROC-AUC (0.9327) but much lower precision and materially higher FPR at the fixed threshold. Hashing SGD was faster and produced smaller artifacts, but its sparse matrix still occupied about 930 MiB at full training size versus about 815 MiB for char 2–5 TF-IDF; in this configuration it did not reduce matrix memory.

The best text model generated exactly one grouped OOF score for each of 397,480 temporal training rows (`oof_assignment_min=max=1`, five grouped folds; 326.77 seconds). The CatBoost hybrid achieved AP 0.93993 [0.9145, 0.9595]. Its paired domain-bootstrap AP difference versus Phase 4 CatBoost engineered-only was −0.00082, 95% CI [−0.00513, 0.00354]; there is no evidence the OOF text score improved AP. At the fixed threshold it lowered FPR, with the expected recall tradeoff.

ExtraTrees tuning used 10 trials (139.52 s) and selected 200 trees, `max_features=sqrt`, `min_samples_leaf=4`, `max_depth=24`; grouped AP was 0.98443. HistGradientBoosting used 10 trials (36.30 s), selected `learning_rate=0.03119`, 294 iterations, 55 leaves, `min_samples_leaf=37`, and `l2_regularization=0.000811`; grouped AP was 0.98640. These tuning scores are on the grouped 100k/25k training subset, not the later temporal holdout.

Full-training costs: char 2–5 SVC fit (vectorizer + classifier) 55.19 s, validation transform 10.30 s, classifier scoring 0.0017 s per 1,000 already-vectorized rows, matrix about 815 MiB, artifact 41.2 MiB. HTML word SVC fit 142.19 s, transform 28.83 s, artifact 294 MiB. Hash SGD log-loss fit 21.00 s, matrix about 930 MiB, artifact 8 MiB. ExtraTrees fit 36.56 s, score 0.0310 s per 1,000 rows, artifact 271.6 MiB; HistGradientBoosting fit 14.59 s, score 0.0100 s per 1,000 rows, artifact 1.81 MiB. Text inference time is reported as sparse-matrix scoring; vectorizer-transform cost is separately recorded and should be included in end-to-end serving estimates.

## Verification

- Cleaned-text join preflight:
  ```text
  usable_rows=498255 train=397480 validation=100775
  cleaned_text_nonempty=483632 nonempty_rate=0.970652 mean_chars=1296.82
  official_test_accessed=False elapsed_seconds=55.92
  ```
- Runtime pilot (12 text candidates, grouped 25k/6k sample): 76.20 seconds total across two batches; largest measured per-model fit was 9.20 seconds for HTML word SVC. Tree pilot (fresh one-trial studies) measured 2.0 seconds for ExtraTrees and 5.2 seconds for HistGradientBoosting. Pilot scores were not used for final model selection.
- Full text comparison:
  ```text
  .\.venv\Scripts\python.exe -m src.evaluation.evaluate_phase5_text
  char_svc_2_5: AP=0.894493; ROC-AUC=0.915554; fit_s=55.2
  html_word_svc: AP=0.883424; ROC-AUC=0.932668; fit_s=142.2
  Total elapsed seconds: 1299.73
  Wrote experiment record: experiments/phase5_text_results.json
  ```
- Tree comparison, corrected and rerun:
  ```text
  .\.venv\Scripts\python.exe -u -m src.evaluation.phase5_trees
  extra_trees: best grouped AP=0.984427; elapsed=142.3s
  hist_gradient_boosting: best grouped AP=0.986405; elapsed=35.9s
  extra_trees: temporal AP=0.893175
  hist_gradient_boosting: temporal AP=0.903388
  ```
- Full suite and static checks:
  ```text
  .\.venv\Scripts\python.exe -m pytest -q
  45 passed in 37.44s
  .\.venv\Scripts\ruff.exe check src/evaluation/phase5_text.py src/evaluation/evaluate_phase5_text.py src/evaluation/phase5_trees.py tests/test_phase5_text.py tests/test_phase5_trees.py
  All checks passed!
  ```
  Both experiment JSON files parse, OOF alignment is exactly once per row, and all 15 saved Phase 5 model artifacts exist with recorded sizes.

## Problems and surprises

- An initial tree result passed probabilities to a metric helper with threshold zero, causing every row to be predicted positive. AP/ROC-AUC were unaffected, but threshold-dependent metrics were invalid. This was caught before reporting, fixed by converting probabilities to log-odds, tested, and the tree evaluation was rerun. The current result JSON contains the corrected metrics.
- The tree run timer was initially shadowed by a per-model timer, understating total elapsed time. The timer variable was corrected and the run rerun; the current record includes tuning, data load, refits, and evaluation.
- Hashing was faster and reduced artifact size but did not reduce sparse-matrix memory for the chosen 2^20 feature width.
- OOF text scores did not improve CatBoost AP within paired domain-bootstrap uncertainty. Complexity does not earn its place based on AP in this phase.

## Risks and open questions

- The Phase 5 text models are not calibrated; score thresholds are fixed at zero for comparability, and threshold selection remains Phase 7 work.
- Several model families have overlapping AP intervals. Do not select solely on tiny point-score differences.
- Model artifacts are ignored local outputs; SQLite studies and CSV/JSON experiment records are intended to be tracked. The existing SVM artifacts remain unchanged.
- Official test data was not accessed.

## Next phase

Phase 6: generate grouped OOF predictions for LightGBM, XGBoost, CatBoost, and the best Phase 5 text candidate; compare regularized logistic stacking, shallow LightGBM meta-learning, simple averaging, and rank averaging. Keep the official test set locked.
