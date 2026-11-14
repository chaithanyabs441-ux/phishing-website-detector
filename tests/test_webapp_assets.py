"""Keep rendered pages and browser-cached frontend files on the same version."""

from hashlib import sha256
import re
from urllib.parse import parse_qs, urlsplit

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest

import webapp.app as webapp
from webapp import assets


def test_asset_url_changes_with_contents_even_when_filename_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "STATIC_DIR", tmp_path)
    stylesheet = tmp_path / "style.css"
    stylesheet.write_text("body { background: yellow; }", encoding="utf-8")
    previous = assets.static_url("style.css")
    stylesheet.write_text("body { background: #18191b; }", encoding="utf-8")
    current = assets.static_url("style.css")

    assert previous != current
    assert current == assets.static_url("style.css")
    assert current.startswith("/static/style.css?v=")


@pytest.mark.parametrize("page", ["home", "result", "error"])
def test_every_page_versions_and_serves_its_styles_and_scripts(monkeypatch, page):
    async def analyze(request, url, html):
        if page == "error":
            raise HTTPException(status_code=400, detail="Invalid website URL")
        return {
            "url": url,
            "prediction": "NO STRONG SIGNAL",
            "risk_level": "NO STRONG SIGNAL",
            "risk_class": "safe",
            "calibrated_probability": "1.0%",
            "model_name": "Synthetic regression fixture",
            "evidence": [],
            "feature_summary": {"HTML provided": "No"},
            "reputation_status": "not_configured",
            "reputation_message": "Not configured",
        }

    monkeypatch.setattr(webapp, "_analyze", analyze)
    client = TestClient(webapp.app)
    response = client.get("/") if page == "home" else client.post(
        "/scan", data={"url": "https://example.org"}
    )
    assert response.status_code == (400 if page == "error" else 200)
    if page == "result":
        assert 'id="results"' in response.text

    references = re.findall(r'(?:src|href)="(/static/[^\"]+)"', response.text)
    assert len(references) >= 10
    assert any(url.startswith("/static/style.css?v=") for url in references)
    assert any(url.startswith("/static/app.js?v=") for url in references)
    for url in set(references):
        parsed = urlsplit(url)
        filename = parsed.path.removeprefix("/static/")
        expected = (assets.STATIC_DIR / filename).read_bytes()
        assert parse_qs(parsed.query)["v"] == [sha256(expected).hexdigest()[:16]]
        asset = client.get(url)
        assert asset.status_code == 200
        assert asset.content == expected
