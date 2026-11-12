from pathlib import Path

from fastapi.testclient import TestClient

import webapp.app as webapp
from src.reputation.google_safe_browsing import SafeBrowsingResult
from src.serving.model_registry import ModelRegistry
from src.serving.request_security import ScanRateLimiter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SANITY_URLS_PATH = PROJECT_ROOT / "configs" / "sanity_urls.txt"


def test_major_site_sanity_urls_stay_below_phishing_alert_gate():
    urls = [
        line.strip()
        for line in SANITY_URLS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert len(urls) == 50

    old_registry = webapp.REGISTRY
    old_rate_limiter = webapp.rate_limiter
    old_lookup = webapp.lookup_safe_browsing
    old_api_key = webapp.GOOGLE_SAFE_BROWSING_API_KEY
    webapp.REGISTRY = ModelRegistry(PROJECT_ROOT)
    webapp.rate_limiter = ScanRateLimiter(1000, 60)
    webapp.lookup_safe_browsing = lambda *_args: SafeBrowsingResult(status="not_configured")
    webapp.GOOGLE_SAFE_BROWSING_API_KEY = ""
    try:
        client = TestClient(webapp.app)
        labels = []
        for url in urls:
            response = client.post("/api/v1/scan", json={"url": url})
            assert response.status_code == 200
            labels.append(response.json()["prediction"])
    finally:
        webapp.REGISTRY = old_registry
        webapp.rate_limiter = old_rate_limiter
        webapp.lookup_safe_browsing = old_lookup
        webapp.GOOGLE_SAFE_BROWSING_API_KEY = old_api_key

    block_count = labels.count("BLOCK")
    warn_count = labels.count("WARN")
    assert block_count == 0 and warn_count <= 2, (
        "Sanity URL gate failed: BLOCK={}, WARN={}; expected BLOCK=0 and WARN<=2".format(
            block_count,
            warn_count,
        )
    )
