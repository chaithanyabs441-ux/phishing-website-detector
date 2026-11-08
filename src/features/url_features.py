import re
import ipaddress
import math
from collections import Counter
from urllib.parse import urlparse, urlsplit

import tldextract


SUSPICIOUS_WORDS = [
    "login",
    "signin",
    "verify",
    "verification",
    "secure",
    "account",
    "update",
    "confirm",
    "password",
    "bank",
    "wallet",
    "payment",
    "invoice",
    "auth",
]

URL_FEATURE_COLUMNS_V1 = [
    "url_length",
    "hostname_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_digits",
    "num_special_chars",
    "num_subdomains",
    "num_query_params",
    "has_at_symbol",
    "has_ip_address",
    "has_https",
    "suspicious_word_count",
]

URL_FEATURE_COLUMNS_V2 = URL_FEATURE_COLUMNS_V1 + [
    "registered_domain",
    "public_suffix",
    "tld_category",
    "shannon_entropy",
    "digit_ratio",
    "letter_ratio",
    "longest_token_length",
    "path_depth",
    "is_punycode_or_idn",
    "hex_encoding_count",
    "is_shortener",
]

SHORTENER_DOMAINS = {
    "bit.ly",
    "buff.ly",
    "cutt.ly",
    "goo.gl",
    "is.gd",
    "ow.ly",
    "rebrand.ly",
    "t.co",
    "tinyurl.com",
    "youtu.be",
}

# Empty suffix_list_urls disables network fetching. tldextract falls back to
# the Public Suffix List snapshot bundled in the installed package.
_OFFLINE_TLD_EXTRACTOR = tldextract.TLDExtract(
    suffix_list_urls=(),
    cache_dir=None,
    fallback_to_snapshot=True,
    include_psl_private_domains=True,
)


def extract_url_features(url):
    parsed = urlparse(url)

    hostname = parsed.netloc
    path = parsed.path
    query = parsed.query

    url_lower = url.lower()

    features = {
        "url_length": len(url),

        "hostname_length": len(hostname),

        "path_length": len(path),

        "num_dots": url.count("."),

        "num_hyphens": url.count("-"),

        "num_digits": sum(c.isdigit() for c in url),

        "num_special_chars": sum(
            not c.isalnum() for c in url
        ),

        "num_subdomains": max(
            0,
            len(hostname.split(".")) - 2
        ),

        "num_query_params": (
            query.count("&") + 1
            if query
            else 0
        ),

        "has_at_symbol": int("@" in url),

        "has_ip_address": int(
            re.match(
                r"^(\d{1,3}\.){3}\d{1,3}$",
                hostname.split(":")[0]
            ) is not None
        ),

        "has_https": int(
            parsed.scheme.lower() == "https"
        ),

        "suspicious_word_count": sum(
            word in url_lower
            for word in SUSPICIOUS_WORDS
        ),
    }

    return features


def extract_url_features_v1(url):
    """Return the original model-compatible URL feature representation."""
    return extract_url_features(url)


def _parse_url_v2(url):
    value = url.strip() if isinstance(url, str) else ""
    if not value:
        return value, urlsplit(""), ""

    has_scheme = re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value)
    candidate = value if has_scheme or value.startswith("//") else "//" + value

    try:
        parsed = urlsplit(candidate)
        hostname = parsed.hostname or ""
    except ValueError:
        parsed = urlsplit("")
        hostname = ""

    return value, parsed, hostname.lower()


def _registered_domain_parts(hostname):
    if not hostname:
        return "", "", 0, False

    try:
        ipaddress.ip_address(hostname)
        return hostname, "", 0, True
    except ValueError:
        pass

    extracted = _OFFLINE_TLD_EXTRACTOR(hostname)
    registered_domain = extracted.registered_domain or ""
    suffix = extracted.suffix or ""
    if registered_domain:
        subdomain_count = max(
            0,
            len(hostname.rstrip(".").split("."))
            - len(registered_domain.split(".")),
        )
    else:
        subdomain_count = 0
    return registered_domain, suffix, subdomain_count, False


def _tld_category(suffix, is_ip):
    if is_ip:
        return "ip"
    if not suffix:
        return "unknown"
    if "." in suffix:
        final_label = suffix.rsplit(".", 1)[-1]
        return "country_code_multi" if len(final_label) == 2 else "multi_label"
    if len(suffix) == 2:
        return "country_code"
    return "generic"


def _shannon_entropy(value):
    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )


def extract_url_features_v2(url):
    """Extract the deterministic, richer URL feature schema v2.

    Public-suffix parsing uses the package-bundled snapshot and makes no
    network request. This function is separate from v1 so existing artifacts
    keep receiving their original feature values.
    """
    value, parsed, hostname = _parse_url_v2(url)
    path = parsed.path or ""
    query = parsed.query or ""
    tokens = re.findall(r"[a-z0-9]+", value.lower())
    registered_domain, suffix, subdomain_count, is_ip = (
        _registered_domain_parts(hostname)
    )

    tld_category = _tld_category(suffix, is_ip)

    hostname_labels = hostname.rstrip(".").split(".") if hostname else []
    is_punycode_or_idn = int(
        any(label.startswith("xn--") for label in hostname_labels)
        or any(ord(char) > 127 for char in hostname)
    )
    characters = len(value)
    digit_count = sum(char.isdigit() for char in value)
    letter_count = sum(char.isalpha() for char in value)
    digit_tokens = sum(token in SUSPICIOUS_WORDS for token in tokens)
    shortener = int(registered_domain in SHORTENER_DOMAINS)

    features = {
        "url_length": len(value),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "num_dots": value.count("."),
        "num_hyphens": value.count("-"),
        "num_digits": digit_count,
        "num_special_chars": sum(not char.isalnum() for char in value),
        "num_subdomains": subdomain_count,
        "num_query_params": query.count("&") + 1 if query else 0,
        "has_at_symbol": int("@" in value),
        "has_ip_address": int(is_ip),
        "has_https": int(parsed.scheme.lower() == "https"),
        "suspicious_word_count": digit_tokens,
        "registered_domain": registered_domain,
        "public_suffix": suffix,
        "tld_category": tld_category,
        "shannon_entropy": _shannon_entropy(value),
        "digit_ratio": digit_count / characters if characters else 0.0,
        "letter_ratio": letter_count / characters if characters else 0.0,
        "longest_token_length": max((len(token) for token in tokens), default=0),
        "path_depth": sum(bool(segment) for segment in path.split("/")),
        "is_punycode_or_idn": is_punycode_or_idn,
        "hex_encoding_count": len(re.findall(r"%[0-9a-fA-F]{2}", value)),
        "is_shortener": shortener,
    }

    return features
