# Root cause and follow-up

## Reported phishing URL

- The reported URL is `https://orca-app-2ulpy.ondigitalocean.app`; the screenshot shows URL-only score `0.8005`, below its `1.5` phishing threshold. The old result consequently displayed a `REVIEW` signal but the contradictory headline `LIKELY BENIGN`.
- URL-only analysis observes URL text, not the hosted page or domain reputation. In this example the URL-only features provide insufficient evidence to cross the stricter threshold; this does not establish that the site is safe.
- The existing threshold-selection report is for the URL+engineered model, not URL-only, so it cannot validate a replacement cutoff. The saved 1,000 false-positive examples contain 136 scores at or above `0.8005`, compared with 19 at or above `1.5`. The saved 1,000 false-negative examples have no scores at or above `0.8005`. These error-only samples do not substitute for a representative labeled validation set.

## Changes

- URL-only positive scores below the configured threshold now display `REVIEW REQUIRED`, rather than a benign headline. The numerical thresholds and trained models are unchanged.
- If no backend is explicitly selected, the scan form now routes to URL+HTML when non-empty HTML is pasted; URL-only remains the default when HTML is empty.
- An optional Google Safe Browsing lookup overrides the model verdict to `PHISHING` when the service explicitly reports malware or social engineering. Missing configuration, no-match responses, and service failures do not create a phishing claim; their status is surfaced and uncertain model results remain under review.

## Limits and one-line cause

The URL-only model sees only URL text, and this hosted-domain phishing URL scored `0.8005`, below the conservative `1.5` cutoff, so the UI incorrectly presented an inconclusive review as benign.

Lowering the cutoff to `0.8005` would flag 136/1,000 stored false-positive examples instead of 19/1,000 at `1.5`; the stored error samples are not a representative validation set, so this is evidence of the trade-off, not a performance estimate. Page HTML can provide additional signals, but no claim is made that it guarantees correct detection.

The optional Google Safe Browsing integration requires `GOOGLE_SAFE_BROWSING_API_KEY`; the full submitted URL is sent to Google when configured. It has only been tested with mocked API responses. A no-match response is not proof of legitimacy, and the existing model can still return `REVIEW REQUIRED`.
