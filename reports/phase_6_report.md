# Phase 6 — Grouped OOF stacking

## What I did

- Added `src/evaluation/phase6_stacking.py` and `configs/phase6_stacking.yaml` for grouped OOF base predictions from LightGBM, XGBoost, CatBoost, and the Phase 5 selected char SVM (`char_svc_2_5_min_df3`).
- Built five `GroupKFold` partitions over the 397,480-row temporal training split. Every base OOF model is trained without its score fold's registered domains; every training row receives exactly one OOF prediction.
- Trained a regularized logistic meta-learner on logits of base probabilities and a shallow LightGBM meta model. Compared these with mean-probability and rank-average ensembles on the later temporal validation split.
- Persisted the aligned base OOF matrix and temporal validation predictions as Parquet under `data/processed/` with row IDs, fold assignments, and SHA-256 hashes in `experiments/phase6_stacking_results.json`.
- Added `tests/test_phase6_stacking.py` for fold/domain disjointness, exactly-once OOF alignment, finite meta logits, paired bootstrap, XGBoost fixed-iteration behavior, and label-shuffle diagnostics.
- Added a nested group-disjoint meta-label shuffle experiment at `experiments/phase6_label_shuffle.json`. It uses a 25k/6k sample, refits all base OOF scores within the outer fit domains only, and evaluates on unseen outer domains.

## Why

The meta model must be trained on OOF base scores; in-sample base predictions would leak fit labels. Temporal validation predictions are separate from base/meta training. The diagnostic was strengthened after an initial shuffle check used a random row split over a shared OOF table; base-model folds for meta-training rows could have included the diagnostic holdout domains. The nested diagnostic excludes those domains from every base fit and from OOF vectorizer fitting, so it is the authoritative leakage check. The initial one-shot shuffle statistic is retained in the main result JSON for traceability but is not relied on as the leakage proof.

## Results

The full OOF stack run took **904.27 seconds (15m 4s)** on a 12-logical-CPU machine. All 397,480 temporal training rows had one OOF score per base; all five folds contained 79,496 score rows. The model comparison uses the 100,775-row later temporal validation split and 200 registered-domain bootstrap replicates.

| Model | AP [95% CI] | ROC-AUC [95% CI] | Precision [95% CI] | Recall [95% CI] | F1 [95% CI] | FPR [95% CI] | Accuracy [95% CI] |
|---|---|---|---|---|---|---|---|
| LightGBM | 0.9192 [0.8929, 0.9405] | 0.9448 [0.9197, 0.9631] | 0.9530 [0.9485, 0.9578] | 0.7668 [0.7092, 0.8187] | 0.8498 [0.8137, 0.8817] | 0.0149 [0.0131, 0.0167] | 0.9235 [0.9012, 0.9394] |
| XGBoost | 0.9210 [0.8920, 0.9439] | 0.9419 [0.9122, 0.9628] | 0.9624 [0.9577, 0.9669] | 0.7812 [0.7226, 0.8304] | 0.8624 [0.8248, 0.8925] | 0.0120 [0.0105, 0.0136] | 0.9296 [0.9101, 0.9481] |
| CatBoost | 0.9408 [0.9187, 0.9580] | 0.9631 [0.9424, 0.9782] | 0.9791 [0.9762, 0.9821] | 0.7758 [0.7304, 0.8255] | 0.8657 [0.8360, 0.8960] | 0.0065 [0.0056, 0.0074] | 0.9321 [0.9138, 0.9487] |
| Char SVC min_df=3 | 0.8946 [0.8658, 0.9234] | 0.9157 [0.8821, 0.9446] | 0.9615 [0.9573, 0.9651] | 0.6942 [0.6456, 0.7539] | 0.8063 [0.7723, 0.8452] | 0.0109 [0.0098, 0.0122] | 0.9059 [0.8846, 0.9274] |
| Mean probability | 0.9218 [0.8951, 0.9478] | 0.9383 [0.9119, 0.9623] | 0.9780 [0.9752, 0.9808] | 0.7801 [0.7256, 0.8490] | 0.8679 [0.8335, 0.9090] | 0.0069 [0.0060, 0.0079] | 0.9330 [0.9139, 0.9515] |
| Rank average | 0.9305 [0.9046, 0.9543] | 0.9508 [0.9234, 0.9712] | 0.5775 [0.5551, 0.6037] | 0.9359 [0.9015, 0.9657] | 0.7143 [0.6947, 0.7338] | 0.2691 [0.2450, 0.2923] | 0.7888 [0.7719, 0.8054] |
| Logistic meta | 0.9385 [0.9174, 0.9618] | 0.9595 [0.9380, 0.9782] | 0.9475 [0.9412, 0.9530] | 0.8127 [0.7577, 0.8724] | 0.8750 [0.8405, 0.9097] | 0.0177 [0.0158, 0.0195] | 0.9345 [0.9164, 0.9539] |
| LightGBM meta | 0.9166 [0.8897, 0.9437] | 0.9337 [0.9042, 0.9597] | 0.9590 [0.9544, 0.9630] | 0.7946 [0.7413, 0.8545] | 0.8691 [0.8370, 0.9055] | 0.0133 [0.0118, 0.0151] | 0.9325 [0.9113, 0.9515] |

### Precision and alerts by prevalence

Each cell is `precision [95% CI] / alerts per 1,000 [95% CI]`, reweighted from validation class-conditional rates. This includes all base and ensemble methods.

| Model | 0.05% | 0.5% | 1% | 5% | 45% |
|---|---|---|---|---|---|
| LightGBM | 0.0252 [0.0219,0.0285] / 15.24 [13.42,17.06] | 0.2059 [0.1839,0.2278] / 18.62 [16.77,20.58] | 0.3426 [0.3118,0.3723] / 22.38 [20.39,24.41] | 0.7309 [0.7025,0.7555] / 52.46 [49.20,55.75] | 0.9769 [0.9735,0.9796] / 353.25 [327.67,377.39] |
| XGBoost | 0.0315 [0.0274,0.0360] / 12.40 [10.91,14.01] | 0.2463 [0.2204,0.2726] / 15.86 [14.47,17.45] | 0.3965 [0.3624,0.4297] / 19.70 [18.15,21.19] | 0.7739 [0.7476,0.7970] / 50.47 [47.19,53.31] | 0.9816 [0.9787,0.9839] / 358.15 [331.74,380.14] |
| CatBoost | 0.0564 [0.0486,0.0648] / 6.88 [5.97,7.73] | 0.3750 [0.3392,0.4106] / 10.34 [9.47,11.22] | 0.5467 [0.5078,0.5833] / 14.19 [13.19,15.16] | 0.8627 [0.8432,0.8794] / 44.96 [42.29,47.59] | 0.9899 [0.9882,0.9913] / 352.67 [332.04,375.05] |
| Char SVC min_df=3 | 0.0308 [0.0270,0.0355] / 11.28 [10.11,12.51] | 0.2419 [0.2178,0.2697] / 14.35 [13.18,15.58] | 0.3907 [0.3588,0.4260] / 17.77 [16.52,19.08] | 0.7697 [0.7447,0.7945] / 45.10 [42.47,48.27] | 0.9811 [0.9784,0.9836] / 318.42 [297.22,345.17] |
| Mean probability | 0.0535 [0.0466,0.0623] / 7.28 [6.42,8.26] | 0.3624 [0.3294,0.4001] / 10.76 [9.74,11.73] | 0.5332 [0.4969,0.5728] / 14.63 [13.44,15.66] | 0.8562 [0.8373,0.8748] / 45.56 [42.68,48.73] | 0.9893 [0.9877,0.9909] / 354.83 [330.22,385.83] |
| Rank average | 0.0017 [0.0016,0.0019] / 269.41 [245.38,292.64] | 0.0172 [0.0158,0.0191] / 272.41 [248.54,295.61] | 0.0339 [0.0312,0.0376] / 275.74 [252.05,298.91] | 0.1547 [0.1436,0.1690] / 302.41 [278.73,325.35] | 0.7400 [0.7227,0.7597] / 569.14 [549.26,590.82] |
| Logistic meta | 0.0225 [0.0199,0.0264] / 18.09 [16.15,19.91] | 0.1875 [0.1695,0.2138] / 21.67 [19.68,23.56] | 0.3169 [0.2909,0.3534] / 25.64 [23.67,27.63] | 0.7074 [0.6813,0.7401] / 57.45 [54.22,60.94] | 0.9741 [0.9708,0.9779] / 375.46 [351.12,402.29] |
| LightGBM meta | 0.0289 [0.0253,0.0329] / 13.73 [12.22,15.50] | 0.2304 [0.2067,0.2548] / 17.25 [15.70,19.11] | 0.3757 [0.3437,0.4074] / 21.15 [19.49,23.17] | 0.7582 [0.7318,0.7817] / 52.40 [49.60,56.14] | 0.9799 [0.9770,0.9824] / 364.90 [340.87,392.34] |

The best single base model is CatBoost (AP 0.94075); the best ensemble is logistic meta (AP 0.93853), a point difference of −0.00223. Paired domain bootstrap for logistic meta versus CatBoost gives 95% CI [−0.00598, 0.00032], which includes zero and does not establish a gain. The recommendation is **drop the stack**: it adds deployment and memory cost without a validation AP improvement beyond uncertainty. Mean probability is also below CatBoost; rank averaging has a high false-positive rate (26.9%) at the fixed threshold.

Base score correlations are high: LightGBM/XGBoost 0.988, XGBoost/CatBoost 0.985, LightGBM/CatBoost 0.973; each tree’s correlation with the text base is about 0.904–0.906. The ensemble’s total base artifact size is 423,442,232 bytes (~404 MiB); the base inferences sum to 0.2884 seconds per 1,000 scans, including the text vectorizer transform. Logistic adds about 0.00005 seconds/1k and 907 bytes; LightGBM meta adds 0.00041 seconds/1k and 96,502 bytes.

### Nested label-shuffle diagnostic

The diagnostic used 25,000 fit and 6,000 validation rows sampled by registered domain from the temporal training partition, with 13,191 fit domains, 3,592 validation domains, and zero overlap. All four bases generated exactly-once grouped OOF scores inside the outer fit subset; validation base models were fit only on that subset. Unshuffled logistic meta AP was 0.9993. Across five independent meta-training label shuffles, mean AP was 0.2477 (SD 0.0548), below the validation prevalence 0.3733. This demonstrates that the nested diagnostic meta learner loses predictive signal when its labels are shuffled. The smaller diagnostic is intentionally a bounded leakage check, not a second full performance estimate.

## Verification

- Full stacking experiment:
  ```text
  .\.venv\Scripts\python.exe -u -m src.evaluation.phase6_stacking
  lightgbm OOF fold 5/5 complete; rows=79496
  xgboost OOF fold 5/5 complete; rows=79496
  catboost OOF fold 5/5 complete; rows=79496
  char_svc_2_5_min_df3 OOF fold 5/5 complete; rows=79496
  best single=catboost; best ensemble=logistic_meta; AP gain=-0.002225
  Wrote experiment record: experiments/phase6_stacking_results.json; elapsed=904.27s
  ```
- Nested grouped label-shuffle check:
  ```text
  .\.venv\Scripts\python.exe -m src.evaluation.phase6_stacking --nested-label-shuffle-only
  nested shuffle AP=0.247663 +/-0.054838; chance=0.373333; unshuffled=0.999312; domain_overlap=0
  Wrote nested diagnostic: experiments/phase6_label_shuffle.json
  ```
- Final suite and static checks:
  ```text
  .\.venv\Scripts\python.exe -m pytest -q
  51 passed in 35.91s
  .\.venv\Scripts\ruff.exe check src/evaluation/phase6_stacking.py tests/test_phase6_stacking.py
  All checks passed!
  ```
- OOF Parquet integrity: 397,480 rows, 27,333,467 bytes; validation prediction Parquet: 100,775 rows, 7,043,346 bytes. Both paths and SHA-256 values are recorded. No official test files were accessed.

## Problems and surprises

- The first stack attempt failed because the shared XGBoost estimator had early stopping enabled but no eval set. Fixed-iteration fold models now disable early stopping and use the iteration count recorded by Phase 4; the full stack then completed.
- The first shuffle check split the shared OOF table into random rows, which was not a valid nested diagnostic because other base-fold training sets could include outer diagnostic validation labels. It is not relied on. A separate nested group-disjoint diagnostic now refits bases within the outer fit groups and reports zero domain overlap.
- Rank average achieves the best ROC-AUC among ensemble strategies but has an unusably high FPR at the fixed threshold. Logistic meta does not beat CatBoost by AP and is not recommended for serving.

## Risks and open questions

- Validation intervals are based on the later temporal holdout; Phase 4/5 base model iteration choices used this same holdout, so the stack comparison remains a model-selection estimate, not an untouched test result.
- The stack depends on multiple large artifacts (especially XGBoost) and adds roughly 0.288 seconds per 1,000 scans for bases; simplicity favors the single CatBoost/SVM candidate rather than retaining an ensemble.
- Official test data remains locked for Phase 7.

## Next phase

Phase 7: calibrate finalists and choose thresholds using validation only, redo the PhreshPhish hash-based benchmarks correctly, pre-register the final selection rule in a committed report, and only then access the official test set once per finalist with a command/commit log. No locked-test data was accessed in Phase 6.
