"""Scan orchestration shared by the HTML form and JSON API."""

import json
import time
from urllib.parse import urlsplit

from starlette.concurrency import run_in_threadpool

from src.features.html_features_v2 import extract_html_features_v2
from src.features.url_features import registered_domain_from_url
from src.serving.presentation import build_evidence, get_prediction, get_risk_class
from src.security.safe_fetch import UnsafeTarget
from src.serving.url_validation import normalize_url, validate_url


class ScanError(ValueError):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class ScanService:
    def __init__(self, *, registry, serving_config, thresholds, top_sites,
                 shared_hosting_domains, reputation_lookup, api_key, logger, fetcher):
        self.registry = registry
        self.serving_config = serving_config
        self.thresholds = thresholds
        self.top_sites = top_sites
        self.shared_hosting_domains = shared_hosting_domains
        self.reputation_lookup = reputation_lookup
        self.api_key = api_key
        self.logger = logger
        self.fetcher = fetcher

    def validate_url(self, value):
        return validate_url(value, self.serving_config["max_url_chars"])

    def shared_hosting(self, hostname):
        host = hostname.rstrip(".").lower()
        return any(
            host == domain or host.endswith("." + domain)
            for domain in self.shared_hosting_domains
        )

    async def analyze(self, raw_url, html):
        try:
            url = self.validate_url(normalize_url(raw_url))
        except ValueError as error:
            raise ScanError(status_code=400, detail=str(error))
        html = html or ""
        if len(html) > int(self.serving_config["max_html_chars"]):
            limit = int(self.serving_config["max_html_chars"])
            raise ScanError(
                status_code=413,
                detail=f"HTML exceeds the configured {limit:,} character limit.",
            )

        fetch_status = "not_requested"
        if not html.strip() and self.serving_config.get("enable_html_fetch", False):
            try:
                html = await run_in_threadpool(
                    self.fetcher,
                    url,
                    self.serving_config["fetcher"]["max_bytes"],
                    self.serving_config["fetcher"]["timeout_seconds"],
                    self.serving_config["fetcher"]["max_redirects"],
                )
                fetch_status = "fetched"
            except Exception as error:
                fetch_status = "blocked" if isinstance(error, UnsafeTarget) else "unavailable"

        started = time.monotonic()
        prediction = await run_in_threadpool(self.registry.predict, url, html)
        url_features = prediction["url_features"]
        html_features = prediction.get("html_features")
        if html_features is None:
            try:
                html_features = await run_in_threadpool(extract_html_features_v2, html, url)
            except Exception:
                html_features = {"html_missing": int(not html.strip())}

        registered_domain = registered_domain_from_url(url)
        allowlisted = registered_domain in self.top_sites
        parts = urlsplit(url)
        shared_host = self.shared_hosting(parts.hostname or "")
        reputation = await run_in_threadpool(self.reputation_lookup, url, self.api_key)

        if prediction["calibrated"]:
            probability = prediction["probability"]
            model_key = prediction.get("model_key", "url_html")
            model_thresholds = self.thresholds["models"][model_key]["thresholds"]
            block_threshold = float(model_thresholds["block"])
            warn_threshold = float(model_thresholds["warn"])
            prediction_label = "BLOCK" if probability >= block_threshold else "WARN" if probability >= warn_threshold else "NO STRONG SIGNAL"
            risk_level = prediction_label
            confidence_note = "Local popular-domain context noted; the calibrated model probability is unchanged." if allowlisted else ""
            if shared_host and not html.strip() and prediction_label != "BLOCK":
                prediction_label = "NEEDS PAGE CONTENT"
                risk_level = "NEEDS PAGE CONTENT"
                confidence_note = "Shared-hosting domain: URL-only evidence is inconclusive. Add page HTML for structural analysis."
            probability_display = "{:.1%}".format(probability)
            score_display = None
            calibration_name = self.thresholds["models"][model_key]["calibration_method"].capitalize()
            route_name = "URL-only" if model_key == "url_only" else "URL + HTML"
            model_name = f"CatBoost {route_name} + {calibration_name} calibration"
        else:
            score = prediction["decision_score"]
            prediction_label = get_prediction(score, self.registry.fallback_threshold)
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

        evidence = build_evidence(url_features, html_features, registered_domain, allowlisted)
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
        self.logger.info(json.dumps({
            "event": "scan_completed",
            "backend": prediction["backend"],
            "verdict": prediction_label,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
            "fetch_status": fetch_status,
        }, sort_keys=True))
        return result
