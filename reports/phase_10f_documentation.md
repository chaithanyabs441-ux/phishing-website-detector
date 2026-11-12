# Phase 10F — Documentation update

Updated `README.md` and restored/replaced `MODEL_CARD.md` to describe the current Phase 10 serving routes, validation-selected thresholds, validation metrics, and the Phase 10E sanity gate result. Both documents distinguish the new validation-only serving models from the historical authorized Phase 7 test evaluation and corrected benchmark. They also include the descriptive seen/unseen domain slices from the Phase 10A audit with an explicit no-selection caveat.

No official test shards, predictions, or test labels were accessed for this documentation step. No model, threshold, dataset, or benchmark result was changed. The Phase 10E gate result remains 1 BLOCK and 2 WARN on the fixed 50-URL list; the list was not used for tuning.

## Verification

- Ruff 0.14.3: `ruff check src webapp tests` passed (`All checks passed!`).
- Full pytest: 87 passed, 1 failed, 13 dependency deprecation warnings. The only failure is the pre-existing strict Phase 10E sanity gate (`tests/test_sanity_urls.py`): 1 BLOCK and 2 WARN on the fixed 50-URL list, where the expected BLOCK count is zero. No list, model, or threshold changes were made in response.
