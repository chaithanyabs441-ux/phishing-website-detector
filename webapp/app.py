"""PhishReaper FastAPI application with verified CatBoost serving and SVM fallback."""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.html_features_v2 import extract_html_features_v2  # noqa: E402
from src.features.url_features import registered_domain_from_url  # noqa: E402
from src.reputation.google_safe_browsing import lookup_url as lookup_safe_browsing  # noqa: E402
from src.serving.model_registry import ModelRegistry  # noqa: E402
from src.serving.request_security import MaxBodySizeMiddleware, ScanRateLimiter  # noqa: E402
from src.serving.safe_fetch import UnsafeTarget, fetch_html  # noqa: E402


MODEL_CONFIG = yaml.safe_load((PROJECT_ROOT / "configs" / "model_config.yaml").read_text(encoding="utf-8"))
SERVING_CONFIG = MODEL_CONFIG["serving"]
THRESHOLDS = yaml.safe_load((PROJECT_ROOT / SERVING_CONFIG["thresholds"]).read_text(encoding="utf-8"))
REGISTRY = ModelRegistry(PROJECT_ROOT)
GOOGLE_SAFE_BROWSING_API_KEY = os.environ.get("GOOGLE_SAFE_BROWSING_API_KEY", "").strip()
logger = logging.getLogger("PhishReaper.scan")
rate_limiter = ScanRateLimiter(SERVING_CONFIG["rate_limit_per_minute"], 60)


def _read_domain_file(path):
    return {
        line.strip().lower().lstrip(".")
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


TOP_SITES = _read_domain_file(PROJECT_ROOT / SERVING_CONFIG["top_sites_file"])
SHARED_HOSTING_DOMAINS = _read_domain_file(PROJECT_ROOT / SERVING_CONFIG["shared_hosting_file"])
CLASSIFICATION_THRESHOLDS = MODEL_CONFIG["thresholds"]
DEFAULT_MODEL = MODEL_CONFIG["default_model"]
FALLBACK_MODEL = MODEL_CONFIG["fallback_model"]


def load_env_file(path):
    """Load local environment values without overriding exported values."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


load_env_file(PROJECT_ROOT / ".env")
GOOGLE_SAFE_BROWSING_API_KEY = os.environ.get("GOOGLE_SAFE_BROWSING_API_KEY", "").strip()


def normalize_url(value):
    value = value.strip()
    if not value:
        return ""
    if not re.match(r"^https?://", value, re.IGNORECASE):
        value = "https://" + value
    return value


def validate_url(value):
    if not value or len(value) > int(SERVING_CONFIG["max_url_chars"]):
        raise ValueError("Enter a URL under the configured length limit.")
    if any(ord(char) < 32 for char in value):
        raise ValueError("URL contains control characters.")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as error:
        raise ValueError("Enter a valid HTTP or HTTPS URL.") from error
    if parsed.scheme.lower() not in {"http", "https"} or not hostname:
        raise ValueError("Enter a valid HTTP or HTTPS URL with a hostname.")
    if any(char.isspace() for char in hostname):
        raise ValueError("The hostname cannot contain spaces.")
    try:
        hostname.encode("idna")
    except UnicodeError as error:
        raise ValueError("The hostname is malformed.") from error
    return value


def get_prediction(decision_score, threshold):
    if decision_score >= threshold:
        return "PHISHING"
    if decision_score >= 0:
        return "REVIEW REQUIRED"
    return "LIKELY BENIGN"


def get_risk_level(decision_score, threshold):
    if decision_score >= threshold:
        return "HIGH RISK"
    if decision_score >= 0:
        return "REVIEW"
    return "LOW RISK"


def get_risk_class(level):
    if level in {"BLOCK", "HIGH RISK"}:
        return "danger"
    if level in {"WARN", "REVIEW", "NEEDS PAGE CONTENT"}:
        return "warning"
    return "safe"


def predict_url_tfidf(url):
    """Compatibility entry point for the verified SVM fallback."""
    vector = REGISTRY.fallback_vectorizer.transform([url])
    return float(REGISTRY.fallback_model.decision_function(vector)[0])


def _shared_hosting(hostname):
    host = hostname.rstrip(".").lower()
    return any(host == domain or host.endswith("." + domain) for domain in SHARED_HOSTING_DOMAINS)


def _evidence(url_features, html_features, registered_domain, allowlisted):
    evidence = []
    if url_features.get("has_ip_address"):
        evidence.append({"title": "IP address in URL", "description": "The URL uses an IP address as its host.", "severity": "high"})
    if url_features.get("has_at_symbol"):
        evidence.append({"title": "@ symbol in URL", "description": "An @ symbol can obscure the actual destination.", "severity": "high"})
    if url_features.get("suspicious_word_count", 0):
        evidence.append({"title": "Account-related URL terms", "description": "The URL contains terms often used in credential prompts.", "severity": "medium"})
    if url_features.get("is_punycode_or_idn"):
        evidence.append({"title": "Internationalized hostname", "description": "The hostname contains IDN or punycode characters.", "severity": "medium"})
    if html_features:
        if html_features.get("html_missing"):
            evidence.append({"title": "Page HTML not provided", "description": "The analysis could not inspect page structure.", "severity": "low"})
        if html_features.get("num_password_inputs"):
            evidence.append({"title": "Password field", "description": "A password input is present in the supplied page HTML.", "severity": "medium"})
        if html_features.get("num_form_action_domain_mismatches"):
            evidence.append({"title": "External form destination", "description": "A form submits to a different registered domain.", "severity": "high"})
        if html_features.get("num_external_domains"):
            evidence.append({"title": "External page resources", "description": "The page references resources or links on other domains.", "severity": "low"})
        if html_features.get("has_javascript_redirect"):
            evidence.append({"title": "JavaScript redirect pattern", "description": "The supplied script text contains a location redirect pattern.", "severity": "medium"})
    if allowlisted:
        evidence.append({"title": "Local popular-domain list", "description": "The registered domain is on the local context list; this is not proof that this URL or page is safe.", "severity": "low"})
    if not evidence:
        evidence.append({"title": "No prominent structural indicator", "description": "No listed URL or HTML indicator was found.", "severity": "low"})
    return evidence


class ScanPayload(BaseModel):
    url: str = Field(..., max_length=4096)
    html: str = Field(default="", max_length=12000000)


app = FastAPI(title="PhishReaper", description="Calibrated phishing analysis with local evidence", version="2.0")
app.add_middleware(MaxBodySizeMiddleware, max_bytes=SERVING_CONFIG["max_request_bytes"])
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


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


async def _analyze(request, raw_url, html):
    _check_rate_limit(request)
    try:
        url = validate_url(normalize_url(raw_url))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    html = html or ""
    if len(html) > int(SERVING_CONFIG["max_html_chars"]):
        raise HTTPException(status_code=413, detail="HTML exceeds the configured 500,000 character limit.")

    fetch_status = "not_requested"
    if not html.strip() and SERVING_CONFIG.get("enable_html_fetch", False):
        try:
            html = await run_in_threadpool(
                fetch_html,
                url,
                SERVING_CONFIG["fetcher"]["max_bytes"],
                SERVING_CONFIG["fetcher"]["timeout_seconds"],
                SERVING_CONFIG["fetcher"]["max_redirects"],
            )
            fetch_status = "fetched"
        except Exception as error:
            fetch_status = "blocked" if isinstance(error, UnsafeTarget) else "unavailable"

    started = time.monotonic()
    prediction = await run_in_threadpool(REGISTRY.predict, url, html)
    url_features = prediction["url_features"]
    html_features = prediction.get("html_features")
    if html_features is None:
        try:
            html_features = await run_in_threadpool(extract_html_features_v2, html, url)
        except Exception:
            html_features = {"html_missing": int(not html.strip())}

    registered_domain = registered_domain_from_url(url)
    allowlisted = registered_domain in TOP_SITES
    parts = urlsplit(url)
    shared_host = _shared_hosting(parts.hostname or "")
    reputation = await run_in_threadpool(lookup_safe_browsing, url, GOOGLE_SAFE_BROWSING_API_KEY)

    if prediction["calibrated"]:
        probability = prediction["probability"]
        block_threshold = float(THRESHOLDS["thresholds"]["block"])
        warn_threshold = float(THRESHOLDS["thresholds"]["warn"])
        prediction_label = "BLOCK" if probability >= block_threshold else "WARN" if probability >= warn_threshold else "SAFE-LOOKING"
        risk_level = prediction_label
        confidence_note = "Local popular-domain context noted; the calibrated model probability is unchanged." if allowlisted else ""
        if shared_host and not html.strip() and prediction_label != "BLOCK":
            prediction_label = "NEEDS PAGE CONTENT"
            risk_level = "NEEDS PAGE CONTENT"
            confidence_note = "Shared-hosting domain: URL-only evidence is inconclusive. Add page HTML for structural analysis."
        probability_display = "{:.1%}".format(probability)
        score_display = None
        model_name = "CatBoost engineered + Platt calibration"
    else:
        score = prediction["decision_score"]
        prediction_label = get_prediction(score, REGISTRY.fallback_threshold)
        risk_level = "HIGH RISK" if prediction_label == "PHISHING" else "REVIEW" if prediction_label == "REVIEW REQUIRED" else "LOW RISK"
        confidence_note = "The verified SVM fallback is uncalibrated; no probability is shown."
        if shared_host and not html.strip() and prediction_label != "PHISHING":
            prediction_label = "NEEDS PAGE CONTENT"
            risk_level = "NEEDS PAGE CONTENT"
            confidence_note = "Shared-hosting domain: URL-only evidence is inconclusive. Add page HTML for structural analysis."
        probability_display = None
        score_display = "{:.4f}".format(score)
        model_name = "URL TF-IDF SVM fallback (uncalibrated)"

    if reputation.status == "matched":
        prediction_label = "BLOCK"
        risk_level = "BLOCK"
    if shared_host and not html.strip():
        # A confirmed threat match remains a block; otherwise require page content.
        if reputation.status != "matched":
            prediction_label = "NEEDS PAGE CONTENT"
            risk_level = "NEEDS PAGE CONTENT"

    evidence = _evidence(url_features, html_features, registered_domain, allowlisted)
    if reputation.status == "matched":
        evidence.insert(0, {"title": "Threat intelligence match", "description": "Google Safe Browsing lists this URL as {}.".format(", ".join(reputation.threat_types)), "severity": "high"})
    result = {
        "url": url,
        "prediction": prediction_label,
        "risk_level": risk_level,
        "risk_class": get_risk_class(risk_level),
        "model_name": model_name,
        "calibrated_probability": probability_display,
        "decision_score": score_display,
        "classification_threshold": None,
        "confidence_note": confidence_note,
        "allowlisted": allowlisted,
        "shared_hosting": shared_host,
        "reputation_status": reputation.status,
        "reputation_message": {
            "matched": "Listed by Google Safe Browsing.",
            "not_listed": "No Google Safe Browsing match. This does not confirm the site is safe.",
            "not_configured": "Google Safe Browsing is not configured; local model analysis ran.",
            "unavailable": "{} Local model analysis is shown.".format(reputation.detail),
        }.get(reputation.status, "Threat intelligence status is unavailable."),
        "fetch_status": fetch_status,
        "evidence": evidence,
        "feature_summary": {
            "Registered domain": registered_domain or "Unknown",
            "URL length": url_features.get("url_length", len(url)),
            "Subdomains": url_features.get("num_subdomains", 0),
            "HTTPS": "Yes" if url_features.get("has_https") else "No",
            "HTML provided": "Yes" if html.strip() else "No",
            "Forms": html_features.get("num_forms", 0),
            "Password fields": html_features.get("num_password_inputs", 0),
            "External domains": html_features.get("num_external_domains", 0),
        },
    }
    logger.info(json.dumps({
        "event": "scan_completed",
        "backend": prediction["backend"],
        "verdict": prediction_label,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
        "fetch_status": fetch_status,
    }, sort_keys=True))
    return result


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
    return {"name": "PhishReaper", "version": app.version, "backend": REGISTRY.active_backend, "feature_version": THRESHOLDS["feature_version"]}


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
