# Phase 10D — Serving labels and per-request fallback

## Changes

- Renamed the lowest calibrated CatBoost tier from `SAFE-LOOKING` to `NO STRONG SIGNAL` in the scan response.
- Kept the safe visual class for this tier and revised the template copy to say that no strong phishing signal was found, without treating the result as a safety guarantee.
- A primary-model inference exception now selects the URL TF-IDF fallback for that request and preserves the primary model and calibrator for the next request.
- The fallback reason contains only the exception type. Exception message text, which could include a submitted URL, is not returned, retained, or logged. Scan logs continue to omit submitted URLs.

## Verification

- Regression test confirms a simulated primary failure falls back once, does not include URL text in the fallback reason, and the same primary model handles the next request.
- Regression tests verify the new lowest-tier label and its safe visual class.
- Ruff 0.14.3: all checks passed.
- Full pytest suite: 87 passed, 13 dependency deprecation warnings.
