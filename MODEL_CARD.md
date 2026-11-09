# PHISH//X model card

**Version:** serving model and calibration committed at Phase 8, with the Phase 9 serving-calibration parity correction.

**Model:** CatBoost engineered URL/HTML features (`models/phase4/catboostengineered.joblib`) plus Platt calibration (`models/phase7/catboost_engineered_calibrator.joblib`).
**Fallback:** URL character TF-IDF + LinearSVC; its decision margin is not a probability.

## Intended use

PHISH//X is a local analyst-triage aid for suspicious URLs. Its signals can help prioritize manual inspection. It is not an autonomous blocklist, a guarantee of safety, or a substitute for incident response. The model was trained and evaluated on a historical PhreshPhish collection; it is not validated for any specific organization, geography, language, or live traffic distribution.

The service accepts a URL and optional user-supplied HTML. It does not render JavaScript. A local, short popular-domain list provides context only, and shared-hosting domains without HTML return `NEEDS PAGE CONTENT`.

## Data and evaluation design

The PhreshPhish snapshot used here contains 498,255 training rows. Temporal training used 397,480 rows collected from 2024-07-02 through 2025-07-09; temporal validation used 100,775 rows from 2025-07-10 through 2025-09-08. Registered-domain GroupKFold was used for tuning/OOF features and grouped calibration. The official test set was evaluated once after finalist pre-registration: 168,060 rows across 95,151 registered domains, with 45.698% phishing prevalence.

The test estimate describes this dataset and collection process. It is not a low-prevalence production estimate. Precision and alert counts below are reweighted from test class-conditional rates to hypothetical prevalence values; they are not new test samples. Confidence intervals use 200 registered-domain bootstrap replicates.

## Performance

At a 0.5 probability threshold on the locked test set:

| Metric | Estimate | 95% registered-domain bootstrap CI |
|---|---:|---:|
| Average precision | 0.99025 | [0.98783, 0.99271] |
| ROC-AUC | 0.98892 | [0.98527, 0.99226] |
| Precision | 0.95454 | [0.95200, 0.95723] |
| Recall | 0.96094 | [0.95369, 0.96886] |
| F1 | 0.95773 | [0.95302, 0.96208] |
| False-positive rate | 0.03852 | [0.03413, 0.04225] |
| Accuracy | 0.96123 | [0.95688, 0.96606] |

At the Phase 7 operating thresholds, the validation estimates were:

| Verdict | Threshold | Precision | Recall | FPR | Precision at 5% prevalence | Alerts/1,000 at 5% |
|---|---:|---:|---:|---:|---:|---:|
| WARN | 0.06818 | 0.5897 | 0.9500 | 0.2598 | 0.1614 | 294.30 |
| BLOCK | 0.94129 | 0.9956 | 0.5706 | 0.0010 | 0.9679 | 29.48 |

The selected Platt calibration was evaluated with grouped OOF probabilities on temporal validation: Brier score 0.05577, log-loss 0.19804, and expected calibration error 0.02004. The final calibrator was refit on all validation scores after method selection. Phase 9 fixed a serving transform mismatch so serving applies the same logit transform used during calibration and evaluation; a regression test now checks parity.

At the 0.5 test threshold, reweighted estimates are:

| Assumed phishing prevalence | Precision (95% CI) | Alerts per 1,000 (95% CI) |
|---:|---:|---:|
| 0.05% | 0.01233 [0.01125, 0.01392] | 38.98 [34.59, 42.70] |
| 0.5% | 0.11140 [0.10261, 0.12418] | 43.13 [38.77, 46.84] |
| 1% | 0.20128 [0.18688, 0.22179] | 47.74 [43.39, 51.44] |
| 5% | 0.56768 [0.54494, 0.59759] | 84.64 [80.38, 88.29] |
| 45% | 0.95330 [0.94902, 0.95848] | 453.61 [449.17, 458.25] |

The selected BLOCK threshold produced 63,602 true positives, 98 false positives, 91,162 true negatives, and 13,198 false negatives on test. WARN produced 75,800 true positives, 28,373 false positives, 62,887 true negatives, and 1,000 false negatives. The low-prevalence precision estimates make clear that even strong ranking metrics do not imply high precision in a low-base-rate deployment.

## Known limitations and failure classes

- The validation WARN threshold deliberately targets 95% recall and consequently flags 25.98% of benign validation examples. It is a review signal and may be too noisy for an alert queue.
- Famous legitimate domains and shared-hosting tenants can have URL/HTML patterns that look suspicious. The Phase 8 bootstrap smoke run exposed an implementation error in calibration; the corrected smoke run still marked some familiar domains BLOCK, so this remains a material model false-positive concern, not a resolved UI issue.
- URLs alone cannot distinguish a legitimate tenant path from a phishing page on shared hosting. The app requests page content for these hosts when HTML is absent.
- Dataset shift, unseen TLDs/domains, shorteners, multilingual URLs, obfuscation, and adversarial changes can reduce performance. The local allowlist is small, manually refreshed, and does not change probability.
- Test prevalence is much higher than many operational settings. Thresholds must not be re-tuned on the test set; any deployment-specific threshold study needs a separate, representative validation set.
- HTML is optional and supplied by the caller. It may be incomplete or stale. The app does not fetch pages unless an operator explicitly turns on the opt-in fetcher, and it never executes JavaScript.

## Privacy, ethics, and security

The URL and pasted HTML are security-sensitive. Structured application logs omit both. If `GOOGLE_SAFE_BROWSING_API_KEY` is configured, PHISH//X sends the complete submitted URL—including path and query string—to Google's Safe Browsing API. It sends no pasted HTML. Query strings can contain account identifiers, reset tokens, or other secrets; operators should disable the integration or redact inputs where that is not acceptable, and should review Google's current service terms and retention policies.

The model can falsely label legitimate sites or miss phishing sites. Do not use a score as evidence of user intent, as a basis for punitive action, or as a sole control for account access. Keep a human review path and provide a way to challenge incorrect classifications. The project is restricted to classical machine learning; no neural network or pretrained embedding was used.

## Artifacts and reproducibility

`models/MANIFEST.json` stores serving artifact hashes and provenance metadata. `configs/model_config.yaml` and `configs/thresholds.yaml` define model selection, limits, and thresholds. The phase-by-phase experiments, benchmark correction, CI, container, and final model comparison are described in `reports/phase_*.md`, especially `reports/phase_9_report.md`. Training requires the locally downloaded dataset and ignored feature caches; app inference requires the model volume and configuration only.
