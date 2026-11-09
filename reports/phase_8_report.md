# Phase 8 — Secure calibrated serving

## What changed

- Replaced the web app's fixed three-SVM selection flow with a registry driven by `configs/model_config.yaml`. Startup verifies every selected artifact against `models/MANIFEST.json`, rejects paths outside the repository, loads the CatBoost finalist and Platt calibrator, and automatically falls back to the manifest-verified URL TF-IDF SVM if the primary model cannot load or predict.
- Added calibrated `BLOCK`, `WARN`, and `SAFE-LOOKING` outputs using the locked Phase 7 thresholds. The UI does not expose an SVM margin as a probability. A small local popular-domain context list never changes the score; URL-only scans on shared-hosting domains request page HTML.
- Added `/api/v1/scan`, `/health`, and `/version`; URL/HTML/request limits, an in-process per-client scan limit, threadpool inference, and logs that omit submitted URLs and HTML.
- Added an opt-in HTML fetcher (disabled in config by default). It allows only HTTP(S) on standard ports, rejects local/private/link-local/metadata targets, revalidates every redirect, caps redirects/bytes/time, accepts only HTML, sends a fixed user agent, and does not retain cookies or use environment proxies.
- Added local bootstrap files for popular sites and shared-hosting domains. `configs/top_sites.txt` is deliberately small and should be replaced with a maintained local Tranco snapshot for deployment.

## Verification

- `ruff check src/serving webapp/app.py tests/test_model_registry.py tests/test_safe_fetch.py tests/test_webapp_security.py tests/test_threshold.py` — passed.
- Full project suite — 77 tests passed (14 existing dependency/platform warnings).
- Manual TestClient smoke scan using the real registry and model:

  | Input | Output |
  | --- | --- |
  | `https://www.google.com` | WARN, 86.4% |
  | `https://example.com/login` | WARN, 86.3% |
  | `https://orca-app-2ulpy.ondigitalocean.app` | BLOCK, 86.4% |
  | `https://paypal.com/signin` | WARN, 81.6% |
  | `https://github.io/account` | NEEDS PAGE CONTENT |
  | Example HTML with password field and external form action | HTTP 200, WARN, 86.4% |

- `/health` returned `{"status":"ok","backend":"catboostengineered"}`. Smoke scans were local requests; no target pages were fetched. Google Safe Browsing was not configured in this environment.

## Risks and follow-up

- Smoke probabilities for familiar sites are high and several return WARN. This is consistent with Phase 7's measured false-positive rate at the selected high-recall WARN operating point, but warrants a deployment-specific review before treating WARN as an alert. The model and thresholds were not changed from test outcomes.
- The top-sites file is a small bootstrap list, not a full Tranco snapshot. Shared-hosting coverage is also a local list that needs maintenance.
- DNS is checked before each fetch/redirect, and redirects are validated before a request. DNS rebinding between validation and connection remains a limitation of this client implementation; keep server-side fetching disabled unless deployment networking adds egress controls.
- The in-memory rate limiter is per process and resets on restart. Put a shared limiter at the deployment edge for multi-worker service.
- No neural model or browser automation was added.

## Next phase

Phase 8 serving and hardening are complete. Stop here for review before beginning Phase 9.
