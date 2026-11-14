"""PhishReaper FastAPI application with verified CatBoost serving and SVM fallback."""

import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.reputation.google_safe_browsing import lookup_url as lookup_safe_browsing  # noqa: E402
from src.serving.model_registry import ModelRegistry  # noqa: E402
from src.security.request_limits import MaxBodySizeMiddleware, ScanRateLimiter  # noqa: E402
from src.security.safe_fetch import fetch_html  # noqa: E402
from src.serving.presentation import (  # noqa: E402
    build_evidence,
    get_prediction as get_prediction,
    get_risk_class as get_risk_class,
    get_risk_level as get_risk_level,
)
from src.serving.scan_service import ScanError, ScanService  # noqa: E402
from src.serving.settings import (  # noqa: E402
    load_env_file as load_env_file,
    load_yaml,
    read_domain_file as _read_domain_file,
)
from src.serving.url_validation import (  # noqa: E402
    normalize_url as normalize_url,
    validate_url as _validate_url,
)
from webapp.schemas import ScanPayload as ScanPayload  # noqa: E402
from webapp.assets import static_url  # noqa: E402


MODEL_CONFIG = load_yaml(PROJECT_ROOT / "configs" / "serving.yaml")
SERVING_CONFIG = MODEL_CONFIG["serving"]
THRESHOLDS = load_yaml(PROJECT_ROOT / SERVING_CONFIG["thresholds"])
REGISTRY = ModelRegistry(PROJECT_ROOT, config=MODEL_CONFIG, threshold_config=THRESHOLDS)
GOOGLE_SAFE_BROWSING_API_KEY = os.environ.get("GOOGLE_SAFE_BROWSING_API_KEY", "").strip()
logger = logging.getLogger("PhishReaper.scan")
rate_limiter = ScanRateLimiter(SERVING_CONFIG["rate_limit_per_minute"], 60)


TOP_SITES = _read_domain_file(PROJECT_ROOT / SERVING_CONFIG["top_sites_file"])
SHARED_HOSTING_DOMAINS = _read_domain_file(PROJECT_ROOT / SERVING_CONFIG["shared_hosting_file"])
CLASSIFICATION_THRESHOLDS = MODEL_CONFIG["thresholds"]
DEFAULT_MODEL = MODEL_CONFIG["default_model"]
FALLBACK_MODEL = MODEL_CONFIG["fallback_model"]


load_env_file(PROJECT_ROOT / ".env")
GOOGLE_SAFE_BROWSING_API_KEY = os.environ.get("GOOGLE_SAFE_BROWSING_API_KEY", "").strip()


def validate_url(value):
    return _validate_url(value, SERVING_CONFIG["max_url_chars"])


def predict_url_tfidf(url):
    """Compatibility entry point for the verified SVM fallback."""
    vector = REGISTRY.fallback_vectorizer.transform([url])
    return float(REGISTRY.fallback_model.decision_function(vector)[0])


def _shared_hosting(hostname):
    return _scan_service().shared_hosting(hostname)


def _evidence(url_features, html_features, registered_domain, allowlisted):
    return build_evidence(url_features, html_features, registered_domain, allowlisted)


app = FastAPI(title="PhishReaper", description="Calibrated phishing analysis with local evidence", version="2.0")
app.add_middleware(MaxBodySizeMiddleware, max_bytes=SERVING_CONFIG["max_request_bytes"])
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
templates.env.globals["static_url"] = static_url


def _check_rate_limit(request):
    host = request.client.host if request.client else "unknown"
    if not rate_limiter.allow(host):
        raise HTTPException(status_code=429, detail="Too many scans. Please wait and try again.")


def _error_page(request, message, status_code=400):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "result": {"error": message}},
        status_code=status_code,
    )


async def _read_html_upload(upload):
    if upload is None or not upload.filename:
        return ""
    if Path(upload.filename).suffix.lower() not in {".html", ".htm"}:
        raise HTTPException(status_code=400, detail="Choose an .html or .htm file.")

    max_bytes = int(SERVING_CONFIG["max_html_upload_bytes"])
    content = await upload.read(max_bytes + 1)
    await upload.close()
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail="HTML file exceeds the 10 MB upload limit.")

    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        html = content.decode("utf-16")
    else:
        try:
            html = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            html = content.decode("windows-1252", errors="replace")
    if len(html) > int(SERVING_CONFIG["max_html_chars"]):
        raise HTTPException(status_code=413, detail="HTML exceeds the configured character limit.")
    return html


def _scan_service():
    return ScanService(
        registry=REGISTRY,
        serving_config=SERVING_CONFIG,
        thresholds=THRESHOLDS,
        top_sites=TOP_SITES,
        shared_hosting_domains=SHARED_HOSTING_DOMAINS,
        reputation_lookup=lookup_safe_browsing,
        api_key=GOOGLE_SAFE_BROWSING_API_KEY,
        logger=logger,
        fetcher=fetch_html,
    )


async def _analyze(request, raw_url, html):
    _check_rate_limit(request)
    try:
        return await _scan_service().analyze(raw_url, html)
    except ScanError as error:
        raise HTTPException(status_code=error.status_code, detail=error.detail) from error


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request, "result": None})


@app.get("/health")
async def health():
    if REGISTRY.fallback_model is None:
        raise HTTPException(status_code=503, detail="No verified inference model is available")
    return {"status": "ok", "backend": REGISTRY.active_backend}


@app.get("/version")
async def version():
    return {
        "name": "PhishReaper",
        "version": app.version,
        "backend": REGISTRY.active_backend,
        "feature_version": THRESHOLDS["feature_version"],
    }


@app.post("/scan", response_class=HTMLResponse)
async def scan(
    request: Request,
    url: str = Form(""),
    model_choice: str = Form(""),
    html: str = Form(""),
    html_file: UploadFile = File(None),
):
    # Old explicit model choices are deliberately removed; all requests use the configured
    # calibrated model, with the verified URL SVM as an automatic load/inference fallback.
    try:
        if getattr(html_file, "filename", None):
            html = await _read_html_upload(html_file)
        result = await _analyze(request, url, html)
    except HTTPException as error:
        return _error_page(request, str(error.detail), error.status_code)
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request, "result": result})


@app.post("/api/v1/scan")
async def api_scan(request: Request, payload: ScanPayload):
    try:
        return await _analyze(request, payload.url, payload.html)
    except HTTPException:
        raise
