"""Compatibility entry point for :mod:`src.security.request_limits`."""

import importlib
import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    runpy.run_module("src.security.request_limits", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("src.security.request_limits")
