"""Bounded, deterministic HTML v2 features; scripts are parsed as text only."""

import hashlib
import json
import re
from functools import lru_cache
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from src.features.url_features import registered_domain_from_url


MAX_HTML_CHARS = 10_000_000
CLEANED_TEXT_MAX_CHARS = 2_048
LOGIN_KEYWORDS = {
    "account", "auth", "bank", "confirm", "credential", "invoice", "log-in",
    "login", "password", "pay", "payment", "secure", "sign-in", "signin",
    "verify", "wallet",
}

HTML_FEATURE_COLUMNS_V2 = [
    "html_missing",
    "html_length",
    "parsed_html_length",
    "html_truncated",
    "html_parse_error",
    "num_forms",
    "num_inputs",
    "num_password_inputs",
    "num_email_inputs",
    "has_password_and_email_inputs",
    "num_form_action_domain_mismatches",
    "num_external_domains",
    "num_external_resource_domains",
    "num_external_script_domains",
    "num_external_css_domains",
    "num_external_image_domains",
    "external_links_resources_ratio",
    "favicon_domain_mismatch",
    "title_length",
    "visible_text_length",
    "title_registered_domain_mismatch",
    "num_hidden_iframes",
    "has_meta_refresh",
    "has_javascript_redirect",
    "num_onmouseover_attributes",
    "right_click_disabled",
    "num_data_uris",
    "empty_or_hash_link_ratio",
    "visible_login_keyword_count",
    "cleaned_text",
]

FEATURE_SPEC_V2 = {
    "version": "html_features_v2",
    "parser": "BeautifulSoup html.parser; script text is inspected but never executed",
    "max_html_chars": MAX_HTML_CHARS,
    "cleaned_text_max_chars": CLEANED_TEXT_MAX_CHARS,
    "registered_domain": "shared offline tldextract snapshot, private suffixes included",
    "columns": HTML_FEATURE_COLUMNS_V2,
    "external_domain_comparison": "registered-domain comparison",
    "login_keywords": sorted(LOGIN_KEYWORDS),
    "cleaned_text_rule": "title followed by visible body text; script/style/noscript/template excluded; whitespace collapsed",
    "resource_ratio_rule": "external anchors plus external src/href/poster resources divided by all anchors plus URL-bearing resource tags",
    "missing_html_rule": "non-string, empty, or whitespace-only input sets html_missing=1",
}
FEATURE_SPEC_HASH_V2 = hashlib.sha256(
    json.dumps(FEATURE_SPEC_V2, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


@lru_cache(maxsize=200_000)
def _registered_domain_cached(url):
    return registered_domain_from_url(url)


def _absolute_url(page_url, value):
    if not isinstance(value, str) or not value.strip():
        return ""
    candidate = value.strip()
    if candidate.startswith("//"):
        safe_page_url = page_url if isinstance(page_url, str) else ""
        scheme = urlsplit(safe_page_url).scheme or "https"
        return f"{scheme}:{candidate}"
    return urljoin(page_url if isinstance(page_url, str) else "", candidate)


def _is_external(page_domain, page_url, value):
    absolute = _absolute_url(page_url, value)
    try:
        parsed = urlsplit(absolute)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return ""
    except ValueError:
        return ""
    domain = _registered_domain_cached(absolute)
    if domain and domain != page_domain:
        return domain
    return ""


def _attrs(tag):
    for key in ("src", "href", "action", "poster", "data"):
        value = tag.get(key)
        if isinstance(value, str):
            yield value


def _is_hidden_iframe(tag):
    if tag.has_attr("hidden") or str(tag.get("aria-hidden", "")).lower() == "true":
        return True
    style = re.sub(r"\s+", "", str(tag.get("style", "")).lower())
    if "display:none" in style or "visibility:hidden" in style:
        return True
    return any(str(tag.get(key, "")).strip() == "0" for key in ("width", "height"))


def _empty_features(html_length=0, parsed_length=0, html_missing=1, truncated=0, parse_error=0):
    features = {column: 0 for column in HTML_FEATURE_COLUMNS_V2}
    features.update({
        "html_missing": int(html_missing),
        "html_length": int(html_length),
        "parsed_html_length": int(parsed_length),
        "html_truncated": int(truncated),
        "html_parse_error": int(parse_error),
        "cleaned_text": "",
    })
    return features


def extract_html_features_v2(
    html,
    page_url,
    max_html_chars=MAX_HTML_CHARS,
    cleaned_text_max_chars=CLEANED_TEXT_MAX_CHARS,
):
    """Extract bounded HTML features and cleaned title/text for one page.

    Parsing is synchronous and capped to ``max_html_chars``. JavaScript is
    never run; script contents are searched as plain text for redirect and
    context-menu patterns.
    """
    html_missing = int(not isinstance(html, str) or not html.strip())
    raw_html = html if isinstance(html, str) else ""
    original_length = len(raw_html)
    parsed_html = raw_html[:max_html_chars]
    truncated = int(original_length > max_html_chars)
    try:
        soup = BeautifulSoup(parsed_html, "html.parser")
    except Exception:
        return _empty_features(
            original_length, len(parsed_html), html_missing, truncated, parse_error=1
        )

    page_domain = _registered_domain_cached(page_url if isinstance(page_url, str) else "")
    forms = soup.find_all("form")
    inputs = soup.find_all("input")
    password_inputs = soup.find_all("input", attrs={"type": re.compile(r"^password$", re.I)})
    email_inputs = soup.find_all("input", attrs={"type": re.compile(r"^email$", re.I)})
    anchors = soup.find_all("a")
    scripts = soup.find_all("script")
    links = soup.find_all("link")

    form_mismatches = 0
    for form in forms:
        action = form.get("action")
        if _is_external(page_domain, page_url, action):
            form_mismatches += 1

    link_values = [anchor.get("href") for anchor in anchors]
    resource_tags = soup.find_all(["img", "source", "script", "iframe", "link", "video", "audio"])
    resource_values = []
    for tag in resource_tags:
        value = tag.get("src") or tag.get("href") or tag.get("poster")
        if isinstance(value, str) and value.strip():
            resource_values.append((tag, value))

    external_domains = set()
    external_resource_domains = set()
    external_script_domains = set()
    external_css_domains = set()
    external_image_domains = set()
    external_count = 0
    for value in link_values:
        domain = _is_external(page_domain, page_url, value)
        if domain:
            external_count += 1
            external_domains.add(domain)
    for tag, value in resource_values:
        domain = _is_external(page_domain, page_url, value)
        if not domain:
            continue
        external_count += 1
        external_domains.add(domain)
        external_resource_domains.add(domain)
        name = tag.name.lower()
        if name == "script":
            external_script_domains.add(domain)
        elif name == "link":
            rel = {str(item).lower() for item in (tag.get("rel") or [])}
            if "stylesheet" in rel:
                external_css_domains.add(domain)
        elif name in {"img", "source"}:
            external_image_domains.add(domain)

    favicon_mismatch = 0
    for link in links:
        rel = {str(item).lower() for item in (link.get("rel") or [])}
        if "icon" in rel:
            favicon_mismatch = int(bool(_is_external(page_domain, page_url, link.get("href"))))
            if favicon_mismatch:
                break

    title_tag = soup.find("title")
    title = title_tag.get_text(" ", strip=True) if title_tag else ""
    script_text = "\n".join(script.get_text(" ", strip=True) for script in scripts)
    lower_script = script_text.lower()
    has_js_redirect = int(bool(re.search(
        r"(?:window\s*\.\s*location|document\s*\.\s*location|location\s*\.\s*(?:href|replace|assign))",
        lower_script,
    )))
    right_click_disabled = int(bool(re.search(
        r"(?:oncontextmenu|contextmenu)[\s=:]|contextmenu[\s\S]{0,120}preventdefault",
        lower_script,
    )))
    onmouseover_count = sum(tag.has_attr("onmouseover") for tag in soup.find_all(True))
    hidden_iframes = sum(_is_hidden_iframe(tag) for tag in soup.find_all("iframe"))
    meta_refresh = int(any(
        str(meta.get("http-equiv", "")).strip().lower() == "refresh"
        for meta in soup.find_all("meta")
    ))
    data_uri_count = sum(
        value.strip().lower().startswith("data:")
        for tag in soup.find_all(True)
        for value in _attrs(tag)
    )
    empty_or_hash = sum(
        not isinstance(anchor.get("href"), str)
        or not anchor.get("href", "").strip()
        or anchor.get("href", "").strip().startswith("#")
        for anchor in anchors
    )

    for tag in soup.find_all(["script", "style", "noscript", "template"]):
        tag.decompose()
    if title_tag:
        title_tag.decompose()
    visible_text = soup.get_text(" ", strip=True)
    normalized_visible = re.sub(r"\s+", " ", visible_text).strip()
    title_tokens = set(re.findall(r"[a-z0-9]+", title.lower()))
    visible_tokens = set(re.findall(r"[a-z0-9]+", normalized_visible.lower()))
    login_count = sum(token in LOGIN_KEYWORDS for token in visible_tokens)
    brand_tokens = set(re.findall(r"[a-z0-9]+", page_domain.split(".")[0].lower()))
    title_domain_mismatch = int(bool(title_tokens and brand_tokens and not title_tokens.intersection(brand_tokens)))
    cleaned_text = re.sub(r"\s+", " ", f"{title} {normalized_visible}").strip()
    cleaned_text = cleaned_text[:cleaned_text_max_chars]
    total_links_and_resources = len(anchors) + len(resource_values)

    features = {
        "html_missing": html_missing,
        "html_length": original_length,
        "parsed_html_length": len(parsed_html),
        "html_truncated": truncated,
        "html_parse_error": 0,
        "num_forms": len(forms),
        "num_inputs": len(inputs),
        "num_password_inputs": len(password_inputs),
        "num_email_inputs": len(email_inputs),
        "has_password_and_email_inputs": int(bool(password_inputs and email_inputs)),
        "num_form_action_domain_mismatches": form_mismatches,
        "num_external_domains": len(external_domains),
        "num_external_resource_domains": len(external_resource_domains),
        "num_external_script_domains": len(external_script_domains),
        "num_external_css_domains": len(external_css_domains),
        "num_external_image_domains": len(external_image_domains),
        "external_links_resources_ratio": external_count / total_links_and_resources if total_links_and_resources else 0.0,
        "favicon_domain_mismatch": favicon_mismatch,
        "title_length": len(title),
        "visible_text_length": len(normalized_visible),
        "title_registered_domain_mismatch": title_domain_mismatch,
        "num_hidden_iframes": hidden_iframes,
        "has_meta_refresh": meta_refresh,
        "has_javascript_redirect": has_js_redirect,
        "num_onmouseover_attributes": int(onmouseover_count),
        "right_click_disabled": right_click_disabled,
        "num_data_uris": int(data_uri_count),
        "empty_or_hash_link_ratio": empty_or_hash / len(anchors) if anchors else 0.0,
        "visible_login_keyword_count": login_count,
        "cleaned_text": cleaned_text,
    }
    return features


def extract_html_features_v2_batch(
    records,
    max_html_chars=MAX_HTML_CHARS,
    cleaned_text_max_chars=CLEANED_TEXT_MAX_CHARS,
):
    """Batch interface for ``(html, page_url)`` records, in input order."""
    rows = [
        extract_html_features_v2(html, url, max_html_chars, cleaned_text_max_chars)
        for html, url in records
    ]
    return rows
