import re
from urllib.parse import urlparse


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