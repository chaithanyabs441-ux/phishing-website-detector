# Patch changelog

## 2026-10-12 — URL-only false-positive threshold

- Raised the default URL TF-IDF phishing threshold from 0.0 to 1.5, so the reported score of 1.4541 is not labeled phishing.
- Keep non-phishing positive URL-only scores in a REVIEW band.
- Removed the sigmoid-derived 0–100 visual number; it was not a calibrated probability and conflicted with binary decisions.
- Preserved the existing vectorizer, model, feature pipeline, routes, and other models' thresholds.
- Added a regression test using the exact screenshot URL and its model score.

This threshold is a targeted, conservative mitigation, not a validation-calibrated threshold. The labeled data needed to estimate its recall impact is not present locally.
