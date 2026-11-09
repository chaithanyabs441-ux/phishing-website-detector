from fastapi.testclient import TestClient

import webapp.app as webapp
from src.reputation.google_safe_browsing import SafeBrowsingResult


class StubRegistry:
    active_backend = "catboostengineered"
    fallback_model = object()
    fallback_threshold = 1.5

    def predict(self, url, html=""):
        return {
            "backend": "catboostengineered", "calibrated": True,
            "probability": 0.01, "raw_probability": 0.01,
            "url_features": {"url_length": len(url), "num_subdomains": 0, "has_https": 1,
                              "suspicious_word_count": 0, "has_ip_address": 0, "has_at_symbol": 0,
                              "is_punycode_or_idn": 0},
            "html_features": {"html_missing": int(not html), "num_forms": 0,
                               "num_password_inputs": 0, "num_external_domains": 0},
        }


def setup_function():
    webapp.REGISTRY = StubRegistry()
    webapp.rate_limiter = webapp.ScanRateLimiter(1000)
    webapp.lookup_safe_browsing = lambda *args: SafeBrowsingResult(status="not_configured")


def test_health_and_json_scan_happy_path():
    client = TestClient(webapp.app)
    assert client.get("/health").json()["status"] == "ok"
    response = client.post("/api/v1/scan", json={"url": "https://example.com"})
    assert response.status_code == 200
    assert response.json()["prediction"] == "SAFE-LOOKING"
    assert response.json()["calibrated_probability"] == "1.0%"


def test_invalid_empty_and_oversized_inputs():
    client = TestClient(webapp.app)
    assert client.post("/api/v1/scan", json={"url": ""}).status_code == 400
    assert client.post("/api/v1/scan", json={"url": "http://[::"}).status_code == 400
    assert client.post("/api/v1/scan", json={"url": "https://example.com", "html": "x" * 500001}).status_code == 413


def test_html_body_limit_and_shared_host_routing():
    client = TestClient(webapp.app)
    oversized = "html=" + ("x" * 1_200_001) + "&url=https%3A%2F%2Fexample.com"
    assert client.post("/scan", content=oversized, headers={"content-type": "application/x-www-form-urlencoded"}).status_code == 413
    result = client.post("/api/v1/scan", json={"url": "https://my-site.vercel.app"}).json()
    assert result["prediction"] == "NEEDS PAGE CONTENT"
    assert client.post("/api/v1/scan", json={"url": "https://google.com"}).json()["allowlisted"] is True
