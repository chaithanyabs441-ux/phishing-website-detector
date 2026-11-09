"""Run local serving smoke scans without live reputation or page fetches."""

from fastapi.testclient import TestClient

from src.reputation.google_safe_browsing import SafeBrowsingResult
import webapp.app as webapp


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


if __name__ == "__main__":
    run_smoke()
