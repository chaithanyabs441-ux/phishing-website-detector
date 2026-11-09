"""Opt-in, bounded HTML fetcher with SSRF checks before every request."""

import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

import httpx


USER_AGENT = "PHISH-X-HTML-Fetch/1.0"
REDIRECT_CODES = {301, 302, 303, 307, 308}


class UnsafeTarget(ValueError):
    pass


def _resolved_addresses(hostname, port, resolver=socket.getaddrinfo):
    lowered = hostname.rstrip(".").lower()
    if lowered in {"localhost", "metadata", "metadata.google.internal"} or lowered.endswith((".localhost", ".local", ".internal")):
        raise UnsafeTarget("Local and metadata hostnames are blocked")
    try:
        literal = ipaddress.ip_address(lowered.strip("[]"))
        addresses = {str(literal)}
    except ValueError:
        try:
            records = resolver(hostname, port, type=socket.SOCK_STREAM)
        except OSError as error:
            raise UnsafeTarget("Hostname could not be resolved") from error
        addresses = {record[4][0] for record in records}
    if not addresses:
        raise UnsafeTarget("Hostname resolved to no addresses")
    for address in addresses:
        parsed = ipaddress.ip_address(address.split("%", 1)[0])
        if not parsed.is_global:
            raise UnsafeTarget("Non-public IP addresses are blocked")
    return addresses


def validate_target(url, resolver=socket.getaddrinfo):
    try:
        parts = urlsplit(url)
        port = parts.port
    except (TypeError, ValueError) as error:
        raise UnsafeTarget("Malformed URL") from error
    if parts.scheme.lower() not in {"http", "https"}:
        raise UnsafeTarget("Only HTTP and HTTPS URLs are allowed")
    if not parts.hostname or parts.username is not None or parts.password is not None:
        raise UnsafeTarget("URL must have a hostname and no user information")
    expected_port = 443 if parts.scheme.lower() == "https" else 80
    if port not in (None, expected_port):
        raise UnsafeTarget("Only ports 80 and 443 are allowed for their matching schemes")
    _resolved_addresses(parts.hostname, expected_port, resolver)
    return parts


def fetch_html(url, max_bytes=1_000_000, timeout_seconds=4, max_redirects=3,
               resolver=socket.getaddrinfo, client_factory=httpx.Client):
    """Fetch a small HTML document; every redirect is validated before request."""
    current = url
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    for redirect_count in range(max_redirects + 1):
        try:
            validate_target(current, resolver)
        except UnsafeTarget:
            raise
        with client_factory(
            follow_redirects=False,
            timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 2)),
            trust_env=False,
        ) as client:
            with client.stream("GET", current, headers=headers) as response:
                if response.status_code in REDIRECT_CODES:
                    location = response.headers.get("location")
                    if not location:
                        raise UnsafeTarget("Redirect response has no location")
                    if redirect_count >= max_redirects:
                        raise UnsafeTarget("Redirect limit exceeded")
                    current = urljoin(current, location)
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if content_type not in {"text/html", "application/xhtml+xml"}:
                    raise UnsafeTarget("Response is not HTML")
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > max_bytes:
                    raise UnsafeTarget("HTML response exceeds the size limit")
                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > max_bytes:
                        raise UnsafeTarget("HTML response exceeds the size limit")
                return bytes(body).decode("utf-8", errors="replace")
    raise UnsafeTarget("Redirect limit exceeded")
