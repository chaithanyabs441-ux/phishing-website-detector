import json
from dataclasses import dataclass
from typing import Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_ENDPOINT = (
    "https://safebrowsing.googleapis.com/v4/threatMatches:find"
)
REQUEST_TIMEOUT_SECONDS = 5
MAX_RESPONSE_BYTES = 1_000_000


@dataclass(frozen=True)
class SafeBrowsingResult:
    status: str
    threat_types: Tuple[str, ...] = ()
    detail: str = ""


def lookup_url(url, api_key):
    if not api_key:
        return SafeBrowsingResult(
            status="not_configured",
            detail="Google Safe Browsing API key is not configured.",
        )

    request_url = "{}?{}".format(
        API_ENDPOINT,
        urlencode({"key": api_key}),
    )
    request_body = {
        "client": {
            "clientId": "phishx-local",
            "clientVersion": "1.0",
        },
        "threatInfo": {
            "threatTypes": [
                "MALWARE",
                "SOCIAL_ENGINEERING",
            ],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}],
        },
    }
    request = Request(
        request_url,
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(
            request,
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as response:
            response_body = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as error:
        return SafeBrowsingResult(
            status="unavailable",
            detail="Google Safe Browsing returned HTTP {}.".format(
                error.code
            ),
        )
    except (URLError, TimeoutError, OSError):
        return SafeBrowsingResult(
            status="unavailable",
            detail="Google Safe Browsing could not be reached.",
        )

    if len(response_body) > MAX_RESPONSE_BYTES:
        return SafeBrowsingResult(
            status="unavailable",
            detail="Google Safe Browsing returned an oversized response.",
        )

    try:
        payload = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return SafeBrowsingResult(
            status="unavailable",
            detail="Google Safe Browsing returned an invalid response.",
        )

    if not isinstance(payload, dict):
        return SafeBrowsingResult(
            status="unavailable",
            detail="Google Safe Browsing returned an invalid response.",
        )

    matches = payload.get("matches", [])
    if not isinstance(matches, list):
        return SafeBrowsingResult(
            status="unavailable",
            detail="Google Safe Browsing returned an invalid response.",
        )

    threat_types = tuple(
        sorted({
            match.get("threatType")
            for match in matches
            if isinstance(match, dict)
            and isinstance(match.get("threatType"), str)
        })
    )

    if threat_types:
        return SafeBrowsingResult(
            status="matched",
            threat_types=threat_types,
        )

    return SafeBrowsingResult(status="not_listed")
