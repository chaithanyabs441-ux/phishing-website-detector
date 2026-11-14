"""Compatibility entry point for :mod:`src.data.repair_html_feature_rows_v1`."""

import importlib
import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    runpy.run_module("src.data.repair_html_feature_rows_v1", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("src.data.repair_html_feature_rows_v1")
