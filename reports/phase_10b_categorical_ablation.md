# Phase 10B — Categorical-feature ablation

## Preregistered decision rule

This rule was recorded before fitting any Phase 10B model. Use paired 95% registered-domain bootstrap percentile intervals for validation AP differences against A. A candidate (B or C) is eligible when the lower bound of its paired AP-difference interval is at least `-0.01`; otherwise the validation evidence permits an AP loss greater than 0.01 and that candidate is rejected. If both candidates are eligible, choose the eligible candidate with the higher full-validation AP; on an AP tie within `0.0001`, choose the model with fewer categorical inputs. If neither is eligible, retain A. Unseen-domain AP is reported as a secondary descriptive metric and does not alter this rule.

## Protocol

- A: current Phase 4 feature set with `registered_domain`, `public_suffix`, and `tld_category` categorical.
- B: same numeric features, with only `public_suffix` and `tld_category` categorical; `registered_domain` is excluded.
- C: numeric features only; all categorical features are excluded.
- Reuse the CatBoost parameters in `experiments/phase4_gbm_results.json`, seed 42, full frozen temporal training/validation partitions, class balancing, 3,000-iteration cap, and the existing 100-round early-stopping procedure.
- Bootstrap validation domains as paired clusters for A/B/C using 200 replicates and seed 42.
- Do not access official test data or use this ablation to alter the locked-test result.

## Results

### Execution results

The experiment used 397,480 temporal training rows and 100,775 temporal validation rows. The validation set had 46,561 registered domains; 45,526 rows across 32,379 domains were absent from the training partition. Unseen-domain AP is computed on those validation rows only.

| Variant | Features | Best iteration | AP (95% domain-bootstrap CI) | ROC-AUC (95% domain-bootstrap CI) | Paired ΔAP vs A (95% CI) | Paired ΔROC-AUC vs A (95% CI) | AP, unseen validation domains |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 53 | 313 | 0.940751 [0.916718, 0.960619] | 0.963142 [0.941811, 0.978218] | reference | reference | 0.954630 |
| B | 52 | 328 | 0.917372 [0.889642, 0.941507] | 0.940505 [0.914187, 0.960546] | -0.023379 [-0.033029, -0.013982] | -0.022636 [-0.034086, -0.011858] | 0.953151 |
| C | 50 | 359 | 0.905387 [0.873133, 0.932711] | 0.927211 [0.893818, 0.953059] | -0.035365 [-0.049641, -0.024810] | -0.035931 [-0.052230, -0.022111] | 0.945352 |

- Registered-domain bootstrap: 200 paired replicates, seed 42.
- Candidate variants eligible under the preregistered AP rule: none.
- Decision: **A**.
- Official test data accessed: **no**.
- Experiment record: `experiments/phase10b_categorical_ablation.json`.

The variant choice is based only on full temporal-validation AP and the paired confidence interval rule above. The unseen-domain metric is descriptive. No official-test outputs or benchmarks were used.

