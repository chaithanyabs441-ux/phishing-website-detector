"""URL input normalization and validation for scan requests."""

import re
from urllib.parse import urlsplit


def normalize_url(value):
    value = value.strip()
    if not value:
        return ""
    if not re.match(r"^https?://", value, re.IGNORECASE):
        value = "https://" + value
    return value


def validate_url(value, max_url_chars):
    if not value or len(value) > int(max_url_chars):
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
