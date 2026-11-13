# PhishReaper model card

## Model summary

PhishReaper is a classical machine-learning phishing URL detector for analyst triage. The serving registry selects one of two calibrated CatBoost models based on whether the request includes HTML. It uses a URL TF-IDF SVM only as a request-scoped fallback when a primary route fails; the SVM margin is uncalibrated. A fallback does not replace or disable the primary model.

| Route | Inputs | Calibration | Validation AP / ROC-AUC (raw) | Grouped-OOF calibrated AP / ROC-AUC | Artifact |
|---|---|---|---:|---:|---|
| URL-only | 24 URL features | Isotonic | 0.923877 / 0.955195 | 0.914668 / 0.946892 | `models/phase10/catboost_url_only.joblib` |
| URL + HTML | 53 URL and HTML features | Platt | 0.940751 / 0.963142 | 0.938509 / 0.960363 | `models/phase10/catboost_url_html.joblib` |

The matching calibrators are in `models/phase10/`; artifact hashes are recorded in `models/MANIFEST.json`. Empty or absent HTML selects URL-only; supplied nonempty HTML selects URL + HTML. The app parses uploaded HTML locally and does not execute its JavaScript. Page fetching is disabled by default.

## Intended use and limitations

Use the output as a triage signal for a submitted URL, not as a determination that a site is safe or malicious. `BLOCK`, `WARN`, and `NO STRONG SIGNAL` are model verdicts and do not establish intent or compromise. Shared-hosting domains and URLs with little distinguishing evidence may require page content or human review. The model does not render pages.

The training data is a historical collection with different prevalence and distribution from live traffic. At 0.05% assumed phishing prevalence, even strong ranking metrics can produce many false alerts. Do not use this system as the sole control for account, payment, or access decisions. The fixed sanity list currently fails its zero-BLOCK gate (1 BLOCK and 2 WARN among 50 URLs); this result is documented and the list was not used for tuning.

## Data and method

The Phase 10 serving models used the frozen temporal split: 397,480 training rows and 100,775 validation rows across 46,561 validation registered domains. The full categorical CatBoost feature set (variant A) was retained after the predeclared Phase 10B comparison. Reduced variants B and C were rejected: paired registered-domain bootstrap AP differences versus A were -0.023379 (95% CI [-0.033029, -0.013982]) and -0.035365 (95% CI [-0.049641, -0.024810]). The rule was to choose B or C unless the AP drop exceeded 0.01 beyond the confidence interval; both failed that condition.

Phase 10 calibration and thresholds used grouped out-of-fold calibrated validation predictions only. No official test set was accessed to train or select the Phase 10 models, calibrators, or thresholds. Per-route validation values are recorded in `configs/thresholds.yaml` and `reports/phase_10c_serving_models.md`.

## Operating thresholds

BLOCK thresholds target validation FPR at or below 0.1%; WARN thresholds target validation FPR at or below 2%. Values below are grouped-OOF calibrated validation cutoffs, not test-set estimates.

| Route | BLOCK cutoff | Validation FPR / recall | WARN cutoff | Validation FPR / recall |
|---|---:|---:|---:|---:|
| URL-only | 0.98860335 | 0.000968 / 0.395196 | 0.38888889 | 0.019450 / 0.770767 |
| URL + HTML | 0.94128949 | 0.000995 / 0.570620 | 0.58389260 | 0.019989 / 0.810684 |

Threshold provenance in `configs/thresholds.yaml` is `validation temporal split only; grouped OOF-calibrated scores`; there is no manual override.

At an illustrative 0.05% phishing prevalence, reweighting validation FPR and recall estimates BLOCK precision of 17.0% for URL-only and 22.3% for URL + HTML, with about 1.16 and 1.28 alerts per 1,000 scans. WARN precision is about 1.9% and 2.0%, with about 19.8 and 20.4 alerts per 1,000. These are prevalence-shift estimates, not measured production results; they assume validation class-conditional rates transfer to the deployment.

## Previously authorized evaluation (historical model)

The following results belong to the prior authorized Phase 7 evaluation of the earlier single URL + HTML CatBoost finalist. They are not a fresh test of the Phase 10 package, URL-only model, or both current routes. A Phase 10 final-evaluation attempt was interrupted before completion and produced no metrics; see `reports/test_set_access_log.md`.

The prior test pass reported AP 0.99025 (95% registered-domain bootstrap CI [0.98783, 0.99271]) and ROC-AUC 0.98892 ([0.98527, 0.99226]). At a 0.5 probability cutoff, precision was 0.95454, recall 0.96094, F1 0.95773, FPR 0.03852, and accuracy 0.96123. At an assumed 0.05% phishing prevalence, the reweighted estimate was 1.23% precision and 38.98 alerts per 1,000. This post-hoc 0.05% scenario is distinct from `assumed_prevalence: 0.05` in `configs/thresholds.yaml`, where the value is a fraction and means 5%.

The corrected Phase 7 hash-mapped benchmark summary for that same historical model was:

| Configured base rate | Files | AP mean (std) | P@R90 mean (std) |
|---:|---:|---:|---:|
| 0.05% | 586 | 0.6373 (0.0631) | 0.0603 (0.0716) |
| 0.1% | 296 | 0.7047 (0.0435) | 0.1023 (0.0870) |
| 0.5% | 59 | 0.8256 (0.0161) | 0.2694 (0.0910) |
| 1% | 29 | 0.8618 (0.0107) | 0.4161 (0.0921) |
| 5% | 5 | 0.9203 (0.0024) | 0.7775 (0.0283) |

Phase 10A produced a descriptive seen/unseen slice of those existing Phase 7 test predictions. It read only `sha256` and `url` from official test shards and joined them to the prior predictions. At the prior BLOCK cutoff of 0.941289:

| Test domains | Rows / registered domains | AP | BLOCK recall | BLOCK FPR |
|---|---:|---:|---:|---:|
| Seen in training | 63,954 / 15,235 | 0.838875 | 0.702309 | 0.000604 |
| Unseen in training | 104,106 / 79,848 | 0.994796 | 0.835670 | 0.001960 |

This is a descriptive analysis of the prior test predictions, not a Phase 10 model evaluation or a basis for model selection. See `reports/test_set_access_log.md`, `reports/phase_10a_domain_overlap_audit.md`, and `reports/phase_7_report.md` for access and methodology.

## Privacy and security

Submitted URLs and HTML are processed by the local app and are not written to application logs. If Google Safe Browsing is enabled, the complete submitted URL, including its path and query, is sent to Google's API; submitted HTML is not sent. A Safe Browsing match overrides the model verdict, while no match is not proof of safety. Keep API keys in environment-based secret storage and out of Git. Uploaded HTML is untrusted input; page fetching remains disabled by default and deployments need appropriate egress controls if it is enabled.

## Versioning and reproducibility

Current model, calibrator, feature, and threshold versions are listed in `configs/model_config.yaml`, `configs/thresholds.yaml`, and `models/MANIFEST.json`. Training datasets and generated caches are local and are not included in Git. The detailed Phase 10 reports are in `reports/phase_10b_categorical_ablation.md`, `reports/phase_10c_serving_models.md`, and `reports/phase_10e_sanity_url_gate.md`.
