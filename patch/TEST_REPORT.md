# Test report

## Commands run

- `.\.venv\Scripts\python.exe -m unittest patch.tests.test_threshold patch.tests.test_google_safe_browsing -v` — passed, 12 tests.
- Python AST syntax checks for `webapp/app.py` and `patch/tests/test_threshold.py` — passed.

## Regression coverage

- Reproduces both the previous legitimate-site example and the newly reported phishing URL score with the deployed URL-only model.
- Verifies the reported URL's positive score below the 1.5 threshold is labeled `REVIEW REQUIRED`, not `LIKELY BENIGN`.
- Verifies pasted HTML without an explicit model choice is routed through the URL+HTML model.
- Mocks a Safe Browsing phishing match and verifies it overrides an inconclusive model result to `PHISHING`.
- Covers missing API key, a no-match response, malformed/network failure behavior, and the threat-match request body without making live API calls.
- Checks just-below and exact threshold boundaries.
- Confirms the engineered and HTML model thresholds remain unchanged.
- Renders the review result template and checks that it shows the threshold and uncalibrated score, not the old sigmoid number.

## Evaluation caveat

The same-test aggregate at the old zero threshold reports URL TF-IDF precision 0.9789, recall 0.9358, FPR 0.0162, 45 FP, and 143 FN on 5,000 examples. Those results are not results for the new threshold. The local raw labeled dataset is absent, and the URL-only threshold cannot be validated from the available error-only samples. Do not interpret 1.5 as a validated optimum or as a probability.

On the saved error-only samples, 136/1,000 false-positive examples score at or above `0.8005`, versus 19/1,000 at `1.5`; none of the 1,000 stored false-negative examples score at or above `0.8005`. These saved examples do not constitute a representative validation set and must not be reported as overall accuracy metrics.

The Google Safe Browsing integration has only been tested with mocked responses. A live lookup requires the user to configure `GOOGLE_SAFE_BROWSING_API_KEY`; a no-match result cannot guarantee the URL is benign.

## Run

```powershell
.\.venv\Scripts\python.exe -m uvicorn webapp.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. From the project root, run regressions with:

```powershell
.\.venv\Scripts\python.exe -m unittest patch.tests.test_threshold patch.tests.test_google_safe_browsing -v
```
