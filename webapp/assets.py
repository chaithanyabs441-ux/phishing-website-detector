"""Content-versioned URLs for the application's presentation assets."""

from hashlib import sha256
from pathlib import Path


STATIC_DIR = Path(__file__).resolve().parent / "static"


def static_url(filename: str) -> str:
    """Invalidate cached assets whenever their contents change, including in development."""
    version = sha256((STATIC_DIR / filename).read_bytes()).hexdigest()[:16]
    return f"/static/{filename}?v={version}"
