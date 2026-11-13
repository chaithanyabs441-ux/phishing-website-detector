# PhishReaper: Temporal Evaluation and Calibrated Classical Models for Phishing URL Triage

**Manuscript status:** research-paper draft based on the repository's recorded experiments, prepared 13 November 2026. This is not a claim of an independently replicated or peer-reviewed result. Author names, affiliations, and a target venue should be added by the project authors before submission.

## Abstract

Phishing detectors are often evaluated on randomly partitioned URL collections, where repeated domains and collection-time structure can make held-out performance look better than performance on later traffic. We present PhishReaper, a reproducible classical-machine-learning system for triaging phishing URLs with either URL-only evidence or optional user-supplied HTML. Using the local 498,255-row PhreshPhish training snapshot, we froze a temporal split of 397,480 training and 100,775 validation rows, with five registered-domain-grouped folds for out-of-fold estimation. The temporal validation window spans 10 July–8 September 2025 and has a 28.2% phishing rate. CatBoost with engineered URL and HTML features achieved average precision (AP) 0.9408 and ROC-AUC 0.9631 before calibration. A preregistered categorical ablation retained the full feature set, including registered-domain identity: removing that feature reduced AP by 0.0234 (paired 95% registered-domain bootstrap interval −0.0330 to −0.0140), while removing all categorical inputs reduced AP by 0.0354 (−0.0496 to −0.0248). For serving, separate URL-only and URL+HTML CatBoost routes are calibrated using grouped out-of-fold predictions. Validation-selected BLOCK cutoffs constrain measured false-positive rate (FPR) to at most 0.1%; WARN cutoffs constrain it to at most 2%. The corresponding calibrated validation recalls are 0.395 and 0.571 at BLOCK, and 0.771 and 0.811 at WARN, for URL-only and URL+HTML respectively. These estimates are validation results, not guarantees under deployment shift. A fixed 50-site regression gate still produces one BLOCK and two WARN results, including a false positive on dropbox.com. The authorized final evaluation of the Phase 10 serving package was interrupted and yielded no metrics. PhishReaper is therefore best understood as a local analyst-triage prototype whose temporal validation is promising but whose current serving package has not passed the final locked-test evaluation or the benign-site regression gate.

**Keywords:** phishing detection; malicious URLs; CatBoost; temporal validation; domain leakage; probability calibration; false-positive rate; HTML features

## 1. Introduction

Phishing pages use URLs and webpage content to impersonate legitimate services and solicit sensitive information. Machine-learning systems can rank suspicious URLs from lexical and structural signals, while webpage content can add evidence about forms, links, scripts, and visible text. However, phishing datasets are time-dependent, domain-heavy, and often class-imbalanced. Random row splits can place related URLs or the same registered domain in both training and evaluation partitions, and a high score on an artificially balanced test collection does not directly express the precision an analyst should expect at a much lower operational prevalence. Prior work on phishing benchmarks has emphasized the need for consistent comparisons and realistic treatment of class imbalance [1], while PhreshPhish was released to provide URL/HTML pairs and benchmark sets designed for more challenging evaluation [2].

This paper describes PhishReaper, a local web application and classical-ML pipeline for URL triage. The system selects one of two calibrated CatBoost models according to evidence availability: a URL-only route when HTML is absent, and a URL+HTML route when the user supplies an HTML file. Its main research contribution is an auditable end-to-end evaluation and serving workflow: frozen temporal and registered-domain-grouped splits; shared, versioned feature extraction; grouped out-of-fold calibration; validation-only low-FPR operating thresholds; and explicit separation between validation evidence, historical test evidence, and current incomplete test access.

We make three practical observations. First, the full categorical CatBoost model was retained under a preregistered paired-domain-bootstrap noninferiority rule; removing registered-domain identity caused a validation AP loss greater than the allowed 0.01 margin. Second, the HTML-enabled route had higher validation recall than the URL-only route at the chosen FPR constraints, although it requires HTML supplied by the user. Third, high aggregate ranking metrics do not ensure benign-site safety: the current 50-site sanity gate fails because Dropbox is classified BLOCK. These results support continued research and controlled analyst use, not deployment as a standalone security control.

## 2. Related work

URL detection has used lexical features with conventional classifiers, including support-vector machines, as well as learned character- and word-level representations. URLNet is a representative neural approach that learns character and word-level URL representations [3]. PhishReaper instead investigates whether a comparatively compact classical pipeline with engineered features and CatBoost can provide useful temporal validation performance with transparent resource and serving boundaries. This is a design choice rather than a claim that classical models dominate learned representations.

Evaluation protocol is central to interpreting phishing-detection results. PhishBench provides a systematic framework for comparing feature and classifier combinations under unified conditions [1]. PhreshPhish provides URL/HTML examples and multiple benchmark subsets at specified base rates [2]. In this context, our study emphasizes temporal validation, registered-domain grouping, and explicit reporting of prevalence-dependent precision. The current results do not establish superiority to the published PhreshPhish baselines because the Phase 10 routes have not completed a fresh official-test evaluation.

## 3. Data and experimental design

### 3.1 Dataset and inclusion

Experiments used the repository's local PhreshPhish snapshot. The recorded training partition contains 498,255 URL/HTML examples: 276,729 benign and 221,526 phishing. Its recorded collection dates run from 2 July 2024 through 8 September 2025. The official test partition is recorded as 168,060 examples (91,260 benign and 76,800 phishing), but these counts describe the dataset and earlier authorized evaluation; they are not results for the Phase 10 models. The project did not record a pinned upstream dataset revision in the phase reports, so exact source snapshot reproducibility requires recording the Hugging Face repository revision and shard hashes in a future release.

Labels are binary (`benign`, `phish`). The data includes a sample SHA-256 identifier, URL, label, collection date, language metadata, and HTML. On the training snapshot, an audit found no duplicate URL rows, no URL with conflicting labels, and 46,764 rows beyond the first sharing a normalized host/path template (9.39%). This template count is a near-duplicate indicator, not a count of confirmed duplicate pages. Training HTML was present on all 498,255 rows; five HTML parses produced contained parse errors and explicit error indicators.

### 3.2 Frozen partitions and leakage controls

Splits are frozen by SHA-256. The temporal cutoff is 10 July 2025: training contains 397,480 rows dated 2 July 2024–9 July 2025, and temporal validation contains 100,775 rows dated 10 July–8 September 2025. Phishing prevalence is 48.6% in temporal training and 28.2% in validation. There are 14,182 registered domains shared between the temporal training and validation windows; the temporal protocol therefore measures later-time performance but is not fully domain-disjoint.

The complementary five-fold GroupKFold partitions are grouped by registered domain. Each of the five validation folds contains 99,651 rows; every fold has zero registered-domain overlap with its corresponding training portion and covers the full collection date range. Grouped out-of-fold (OOF) predictions are used for calibration and feature/model comparisons where recorded. Temporal and grouped results answer different questions: the former tests a later time window, while the latter tests scoring held-out domains among rows spanning the time range.

### 3.3 Features and models

The URL v2 extractor is deterministic and offline. Its 24 serving features combine URL length and component counts, IP/HTTPS indicators, suspicious-token counts, registered domain, public suffix and suffix category, entropy and character ratios, token and path statistics, internationalized-domain and encoded-byte indicators, and a known-shortener flag. Registered-domain identity is categorical in the full CatBoost model.

The HTML v2 extractor parses user-supplied HTML without executing JavaScript. Its serving schema includes 29 URL/HTML-derived fields, for 53 features in the combined route. Signals include form/input and password-field counts; form-action and title/domain mismatch; external link/resource domains; hidden iframes; redirects; mouseover/context-menu attributes; data URIs; visible login-keyword counts; and explicit missing, truncation, and parse-error markers. The HTML parser caps serving input at 10 million characters and cleaned text at 2,048 characters. The application accepts an optional HTML upload; automated target-page fetching is disabled by default.

The project compared linear and tree-based classical models, including URL TF-IDF SVMs, LightGBM, XGBoost, CatBoost, text models, and OOF-score hybrids. On the temporal validation split, CatBoost with engineered features had the strongest point AP among the core single models (0.9408). The OOF text hybrid's AP was 0.9399, with paired evidence not supporting a reliable improvement. Phase 10 ablation uses the Phase 4 CatBoost parameters, seed 42, class balancing, up to 3,000 iterations, and 100-round early stopping.

### 3.4 Model selection and ablation rule

Before the Phase 10B fits, the project recorded this rule: compare candidate variants B and C with A using paired 95% registered-domain bootstrap intervals for AP differences. A reduced candidate is eligible only if the lower bound is at least −0.01; if neither qualifies, retain A. Variant A uses registered domain, public suffix, and TLD category as categoricals. B removes registered domain but retains suffix and category; C removes all categorical features. The validation rows are paired by registered domain across model variants. Two hundred bootstrap replicates were used.

### 3.5 Calibration, metrics, and operating points

The serving package has two route-specific calibrators. It obtains calibration inputs from grouped OOF predictions on temporal validation and refits the selected calibration method on the full validation partition for serving. URL-only uses isotonic calibration; URL+HTML uses Platt calibration. The operating points are selected from grouped-OOF-calibrated validation scores: BLOCK is the lowest cutoff with observed validation FPR no higher than 0.1%; WARN is the lowest cutoff with observed FPR no higher than 2%. This thresholding operation does not make those rates a guarantee for future populations. AP and ROC-AUC are reported as ranking metrics; confidence intervals in the ablation and preceding model comparisons use registered-domain bootstrap. Where the reports do not give a paired interval or bootstrap specification for a result, none is inferred here.

## 4. Results

### 4.1 Temporal validation and categorical ablation

On the 100,775-row temporal validation set, full variant A achieved raw-score AP 0.940751 (95% registered-domain bootstrap CI 0.916718–0.960619) and ROC-AUC 0.963142 (0.941811–0.978218). B achieved AP 0.917372 and ROC-AUC 0.940505; its paired AP difference from A was −0.023379 (95% CI −0.033029 to −0.013982). C achieved AP 0.905387 and ROC-AUC 0.927211; its paired AP difference was −0.035365 (95% CI −0.049641 to −0.024810). Neither reduced variant met the preregistered noninferiority criterion, so A was retained.

There were 45,526 validation rows across 32,379 registered domains not present in the temporal training partition. AP on these unseen-domain validation rows was 0.954630 for A, 0.953151 for B, and 0.945352 for C. This is descriptive and does not change the full-validation decision rule. The result also illustrates that the aggregate loss from removing domain categoricals is not replicated as a large unseen-domain AP penalty; the model choice should therefore be interpreted as a validation-utility decision under the stated split and rule, not evidence that domain identity improves generalization in every subgroup.

### 4.2 Current calibrated serving routes

| Route | Features | Calibration | Raw validation AP / ROC-AUC | Grouped-OOF calibrated AP / ROC-AUC | BLOCK cutoff (FPR, recall) | WARN cutoff (FPR, recall) |
|---|---:|---|---:|---:|---:|---:|
| URL-only | 24 | Isotonic | 0.923877 / 0.955195 | 0.914668 / 0.946892 | 0.988603 (0.000968, 0.395196) | 0.388889 (0.019450, 0.770767) |
| URL + HTML | 53 | Platt | 0.940751 / 0.963142 | 0.938509 / 0.960363 | 0.941289 (0.000995, 0.570620) | 0.583893 (0.019989, 0.810684) |

The HTML-enabled route has higher validation recall at both operating points in this split, while the URL-only route operates without page content. The raw-score and grouped-OOF-calibrated AP values are reported separately because they come from different scoring procedures: the latter uses held-out, fold-specific calibration predictions and is the relevant measure for the calibrated serving path.

At an illustrative phishing prevalence of 0.05%, reweighting validation FPR and recall yields estimated BLOCK precision of 17.0% (URL-only) and 22.3% (URL+HTML), corresponding to 1.16 and 1.28 alerts per 1,000 scans. Estimated WARN precision is about 1.9% and 2.0%, with 19.8 and 20.4 alerts per 1,000. These values assume validation class-conditional rates transfer to deployment and are not observed production outcomes. In particular, the low-prevalence estimates are highly sensitive to small FPR changes.

### 4.3 Fixed benign-site regression gate

A fixed list of 50 major-site URLs is used only as a regression gate, not for fitting, feature selection, or threshold tuning. The current serving package returns one BLOCK and two WARN results; therefore it fails the zero-BLOCK requirement while meeting the maximum-two-WARN requirement. The BLOCK is a URL-only false positive for `dropbox.com`, with displayed probability 98.9% against a configured cutoff of 0.98860335. This failure remains unresolved and is material to any presentation of the system as safe for broad use.

### 4.4 Historical locked-test and benchmark evidence (earlier model only)

An earlier Phase 7 pass evaluated a single CatBoost URL+HTML finalist, before the Phase 10 two-route package. It reported AP 0.99025 (95% registered-domain bootstrap CI 0.98783–0.99271) and ROC-AUC 0.98892 (0.98527–0.99226). At probability cutoff 0.5, precision was 0.95454, recall 0.96094, F1 0.95773, FPR 0.03852, and accuracy 0.96123. The test set in that pass had about 45.7% phishing prevalence. These results are included only as historical project context; they do not validate the current Phase 10 models or thresholds.

The earlier finalist's corrected hash-mapped PhreshPhish benchmark results were as follows:

| Configured phishing base rate | Benchmark files | Mean AP (SD) | Mean precision at 90% recall (SD) |
|---:|---:|---:|---:|
| 0.05% | 586 | 0.6373 (0.0631) | 0.0603 (0.0716) |
| 0.1% | 296 | 0.7047 (0.0435) | 0.1023 (0.0870) |
| 0.5% | 59 | 0.8256 (0.0161) | 0.2694 (0.0910) |
| 1% | 29 | 0.8618 (0.0107) | 0.4161 (0.0921) |
| 5% | 5 | 0.9203 (0.0024) | 0.7775 (0.0283) |

The corrected benchmark is distinct from the current model validation table and cannot establish current-route performance. It emphasizes that performance varies with the benchmark's chosen base rate and sample set.

### 4.5 Current official-test status

The authorized one-pass evaluation of the Phase 10 URL-only and URL+HTML serving package was interrupted before completion. One full shard of 1,000 rows completed and later shard processing began; the total number processed is unknown. A separate accidental read of `test-000.parquet` occurred during process-status checking. No aggregate metrics or Phase 10 test predictions were produced, and the evaluator was not resumed. The access incident and interruption are documented in `reports/test_set_access_log.md` and `reports/phase_10_final_evaluation.md`. Consequently, this paper reports no official-test results for the current serving models. A new pass requires fresh authorization and a new pre-access record.

## 5. System and operational considerations

PhishReaper exposes a local web interface and JSON scan endpoint. It routes requests with absent HTML to URL-only CatBoost and requests with supplied nonempty HTML to URL+HTML CatBoost. Each primary model uses its matching serialized calibrator and validation-selected cutoffs. A manifest records artifact SHA-256 hashes and startup checks ensure the configured paths remain within the project. If primary inference raises an exception, a URL TF-IDF SVM is used for that request only; the exception does not permanently disable the primary model. The fallback produces an uncalibrated margin, so it should not be interpreted as a probability.

The result labels are `BLOCK`, `WARN`, `NO STRONG SIGNAL`, and, for certain URL-only cases, `NEEDS PAGE CONTENT`. They are triage outputs rather than ground-truth declarations. Uploaded HTML is parsed locally, and its JavaScript is not run. Page retrieval is disabled by default. If Google Safe Browsing is configured, the complete submitted URL, including path and query, is sent to Google's API; HTML is not sent. Application logs are designed to omit submitted URLs and HTML. These choices reduce unnecessary remote page retrieval and data retention, but users should be informed about the Safe Browsing URL transfer when that optional integration is enabled.

## 6. Discussion

The experiments show that a classical model can achieve strong ranking on a later temporal window, and that adding HTML-derived features improves the current route's temporal AP and recall at the selected low-FPR thresholds. However, URL+HTML metrics are measured on a set where all training examples had HTML present; real URL-only requests route to a separately trained model, which addresses that observed mismatch in availability. The HTML route still assumes that user-supplied HTML is representative of the page being assessed and can be safely parsed.

Registered-domain categoricals materially help the full temporal-validation metric under the preregistered test. Yet the Phase 10A descriptive slice of the earlier model's official test predictions showed AP 0.838875 on domains seen during training versus 0.994796 on unseen domains. This unusual direction and large difference deserve investigation, not a causal explanation: the slice is descriptive, based on a different model, and can reflect label mix, collection structure, and sample composition. It is not evidence that seen domains are inherently harder or that unseen-domain performance is universally higher. Future work should report domain-stratified support and prevalence with an untouched evaluation design and evaluate both current routes on a fresh authorized test pass.

The false positive on Dropbox is particularly important. Aggregate AP and threshold constraints cannot guarantee a desirable decision on a named benign service, and a manually raised threshold would simply move the error tradeoff. The current fixed-list failure should be resolved using independent, representative benign validation evidence and a documented policy, while preserving an untouched test set. The fixed list must not be used as a threshold-tuning set.

The illustrative 0.05% prevalence estimates show why FPR deserves direct attention: even an FPR near 0.1% can generate more benign than phishing alerts when prevalence is low. These estimates assume stationarity of class-conditional error rates and ignore changing campaign mix, repeated requests, shared hosting, and analyst deduplication. They are therefore scenarios for planning, not service-level assurances.

## 7. Limitations and future work

First, there is no completed Phase 10 official-test result. Current claims are based on temporal validation, grouped OOF calibration, ablation, and the fixed benign-site gate. Historical Phase 7 test results are explicitly not transferable to the Phase 10 package. Second, the validation set is a single time window, and the temporal split shares 14,182 registered domains between its training and validation portions. The separate domain-grouped folds mitigate a different source of leakage but do not replace a later-time, domain-disjoint external evaluation.

Third, the project reports 46,561 validation registered domains, but the manuscript depends on a dataset snapshot whose upstream revision was not pinned in the phase records. Reproduction should archive the exact dataset commit identifier, downloaded shard checksums, feature-spec hashes, split hashes, dependency lockfile, code revision, and model artifact manifest. Fourth, prevalence-weighted precision and alert rates rely on transfer of validation FPR and recall. Fifth, the 50-site gate currently fails, revealing a concrete false positive. Sixth, shared hosting, domain churn, URL redirects, language diversity, and live adversarial adaptation remain uncharacterized. Lastly, the current model uses registered-domain identity as an input and may encode dataset-specific domain-label associations; its use should be independently tested before deployment in a new domain population.

Priority future work is: (i) investigate and remediate the benign-site BLOCK using independently collected, representative data without tuning on the locked test; (ii) perform a fresh, authorized single-pass test evaluation of both Phase 10 routes with complete access logging; (iii) evaluate on later data and domain-disjoint external sets; (iv) pin the dataset and dependency revisions; (v) quantify calibration error and confidence intervals at the chosen operational thresholds; and (vi) assess operational controls such as shared rate limiting, monitoring, model-drift checks, and human review.

## 8. Conclusion

PhishReaper is a classical phishing-URL triage prototype with versioned URL and HTML features, CatBoost routes for different evidence availability, grouped-OOF calibration, validation-only low-FPR thresholds, and privacy-conscious local parsing. The full categorical model achieved temporal validation AP 0.9408, and the HTML route achieved validation recall 0.5706 at a measured FPR of 0.0995% at BLOCK. The results support the value of temporal and grouped evaluation and of reporting prevalence-aware tradeoffs. They do not yet support a claim that the current serving package is ready for production: its final official-test pass is incomplete, the major-site sanity gate has one BLOCK false positive, and the validation setting differs from operational traffic. Until these issues are addressed, outputs should inform analyst review rather than make autonomous safety decisions.

## Data and code availability

The project code, phase reports, frozen split manifests, model configuration, and artifact hashes are maintained in this repository. Raw data, generated caches, and evaluation artifacts are not committed. The PhreshPhish dataset is available from its [Hugging Face dataset page](https://huggingface.co/datasets/phreshphish/phreshphish); consult its current license and terms. For exact reproduction, authors should publish a revision-pinned data manifest and document any restrictions on redistribution. This manuscript intentionally does not include raw URLs, HTML samples, or secrets.

## References

1. Ayman El Aassal, Shahryar Baki, Avisha Das, and Rakesh M. Verma. “An In-Depth Benchmarking and Evaluation of Phishing Detection Research for Security Needs.” *IEEE Access*, 8, 22170–22192, 2020. [doi:10.1109/ACCESS.2020.2969780](https://doi.org/10.1109/ACCESS.2020.2969780).
2. Thomas Dalton, Hemanth Gowda, Girish Rao, Sachin Pargi, Alireza Hadj Khodabakhshi, Joseph Rombs, Stephan Jou, and Manish Marwah. “PhreshPhish: A Real-World, High-Quality, Large-Scale Phishing Website Dataset and Benchmark.” arXiv:2507.10854, 2025. [https://arxiv.org/abs/2507.10854](https://arxiv.org/abs/2507.10854). Dataset: [https://huggingface.co/datasets/phreshphish/phreshphish](https://huggingface.co/datasets/phreshphish/phreshphish).
3. Hung Le, Quang Pham, Doyen Sahoo, and Steven C. H. Hoi. “URLNet: Learning a URL Representation with Deep Learning for Malicious URL Detection.” arXiv:1802.03162, 2018. [https://arxiv.org/abs/1802.03162](https://arxiv.org/abs/1802.03162).

---

### Internal evidence map

This map is for manuscript maintenance and may be removed for submission. Current metrics and route details are sourced from `reports/phase_10b_categorical_ablation.md`, `reports/phase_10c_serving_models.md`, `configs/thresholds.yaml`, `reports/phase_10e_sanity_url_gate.md`, and `reports/phase_10_final_evaluation.md`. Dataset and split audits are in `reports/phase_2_report.md` and `reports/phase_3_report.md`. Classical-model comparisons are in `reports/phase_4_report.md`, `reports/phase_5_report.md`, `reports/phase_6_report.md`, and `reports/phase_9_report.md`. Historical results and access history are in `reports/phase_7_report.md` and `reports/test_set_access_log.md`. Serving/privacy behavior is in `README.md`, `MODEL_CARD.md`, `src/features/url_features.py`, `src/features/html_features_v2.py`, and `src/serving/model_registry.py`.
