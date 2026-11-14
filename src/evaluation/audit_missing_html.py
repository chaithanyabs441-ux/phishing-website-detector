"""Compatibility entry point for :mod:`src.data.audit_missing_html`."""

import importlib
from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    runpy.run_module("src.data.audit_missing_html", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("src.data.audit_missing_html")
