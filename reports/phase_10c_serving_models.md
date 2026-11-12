# Phase 10C — URL-only and URL+HTML serving models

Models were trained on the frozen temporal training split. Early stopping, grouped-OOF calibrator selection/refit, and both operating thresholds used the temporal validation split only. No official test data, predictions, labels, or evaluation artifacts were read.

Training rows: 397,480; validation rows: 100,775; validation registered domains: 46,561.

| Model route | Features | Best iteration | Calibration | Validation AP (raw) | Validation ROC-AUC (raw) | BLOCK threshold / FPR / recall | WARN threshold / FPR / recall |
|---|---:|---:|---|---:|---:|---|---|
| url_only | 24 | 128 | isotonic | 0.923877 | 0.955195 | 0.98860335 / 0.000968 / 0.395196 | 0.38888889 / 0.019450 / 0.770767 |

### url_only calibration and artifacts

Grouped-OOF validation AP=0.914668, ROC-AUC=0.946892, Brier=0.062406, log-loss=0.222968, ECE=0.019554. Each validation row received one held-out calibration score (assignment min/max 1/1).

Model artifact: `models/phase10/catboost_url_only.joblib` (SHA-256 `bf6d5f134ceb413e3b0d0a0f4f05d25dea35fd842e5d94a7f39ce0226b9a3c99`); calibrator: `models/phase10/catboost_url_only_calibrator.joblib` (SHA-256 `217147031ee01fd4718e9b615d0e7c84016f65e6fabfaebbdee068c877b32716`).

| url_html | 53 | 313 | platt | 0.940751 | 0.963142 | 0.94128949 / 0.000995 / 0.570620 | 0.58389260 / 0.019989 / 0.810684 |

### url_html calibration and artifacts

Grouped-OOF validation AP=0.938509, ROC-AUC=0.960363, Brier=0.055772, log-loss=0.198037, ECE=0.020035. Each validation row received one held-out calibration score (assignment min/max 1/1).

Model artifact: `models/phase10/catboost_url_html.joblib` (SHA-256 `30fdd7e683502f502fac31128ee9f2f304d8c63d0e4505894b7600f5d7dc8ae6`); calibrator: `models/phase10/catboost_url_html_calibrator.joblib` (SHA-256 `ade1e2bc099a4bfe3a60e8e3941247a7bdb357be75ca3a2a27ed8670ad2622d2`).

The BLOCK threshold is the lowest calibrated probability whose validation FPR is no greater than 0.1%; WARN is the lowest probability whose validation FPR is no greater than 2%. Both were selected from grouped-OOF calibrated validation scores. The runtime chooses `url_only` when HTML is empty and `url_html` when supplied HTML is nonempty.

Threshold provenance is recorded as validation-only in `configs/thresholds.yaml`; it contains no manual override. The artifact SHA-256 hashes are registered in `models/MANIFEST.json`.

This step did not access the official test set and did not alter any recorded Phase 7 benchmark result.
