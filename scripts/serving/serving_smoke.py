"""Run local serving smoke scans without live reputation or page fetches."""

from fastapi.testclient import TestClient
from pathlib import Path

from src.reputation.google_safe_browsing import SafeBrowsingResult
from src.security.request_limits import ScanRateLimiter
import webapp.app as webapp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SANITY_URLS_PATH = PROJECT_ROOT / "configs" / "sanity_urls.txt"


SAMPLE_URLS = [
    "https://www.google.com",
    "https://example.com/login",
    "https://orca-app-2ulpy.ondigitalocean.app",
    "https://paypal.com/signin",
    "https://github.io/account",
]


def run_smoke():
    # Keep this diagnostic offline even if a developer has a local API key.
    def no_live_reputation(*args):
        return SafeBrowsingResult(status="not_configured")

    webapp.lookup_safe_browsing = no_live_reputation
    webapp.rate_limiter = ScanRateLimiter(1000, 60)
    client = TestClient(webapp.app)
    print("health:", client.get("/health").json())
    for url in SAMPLE_URLS:
        response = client.post("/api/v1/scan", json={"url": url})
        response.raise_for_status()
        result = response.json()
        print("{} => {} ({})".format(url, result["prediction"], result["calibrated_probability"]))
    html = '<form action="https://other.example"><input type="password"></form>'
    response = client.post(
        "/api/v1/scan",
        json={"url": "https://example.com", "html": html},
    )
    response.raise_for_status()
    result = response.json()
    print("pasted HTML => {} {} ({})".format(
        response.status_code, result["prediction"], result["calibrated_probability"],
    ))

    sanity_urls = [
        line.strip()
        for line in SANITY_URLS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    results = []
    for url in sanity_urls:
        response = client.post("/api/v1/scan", json={"url": url})
        response.raise_for_status()
        results.append(response.json()["prediction"])
    block_count = results.count("BLOCK")
    warn_count = results.count("WARN")
    print(
        "sanity URL gate: total={} BLOCK={} WARN={}".format(
            len(results), block_count, warn_count,
        ),
    )
    if block_count or warn_count > 2:
        raise AssertionError(
            "Sanity URL gate failed: expected zero BLOCK and at most two WARN results.",
        )


if __name__ == "__main__":
    run_smoke()
