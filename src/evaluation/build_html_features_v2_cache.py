"""Compatibility entry point for :mod:`src.data.html_feature_cache`."""

import importlib
from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    runpy.run_module("src.data.html_feature_cache", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("src.data.html_feature_cache")
