# Phase 10E — Major-site sanity URL gate

## Changes

- Added a fixed list of 50 major-site URLs in `configs/sanity_urls.txt`.
- Added an offline pytest gate that scans the list through the serving API and requires zero `BLOCK` results and no more than two `WARN` results.
- Extended `src/models/serving_smoke.py` to run the same gate with Safe Browsing mocked offline and enough local rate-limit capacity for all URLs.
- The list is a regression check only. It was not used to fit models or select thresholds.

## Result

The current model fails the requested gate on this fixed list: **1 BLOCK and 2 WARN** across 50 URLs. The WARN cap passes; the BLOCK cap does not. The model, thresholds, and URL list were not adjusted in response to this result.

## Verification

- Ruff 0.14.3: all checks passed.
- Full pytest: 87 passed, 1 failed (`tests/test_sanity_urls.py`), 13 dependency deprecation warnings.
- The single failure is the intentionally strict sanity gate reporting the one BLOCK result.
