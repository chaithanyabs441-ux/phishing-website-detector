"""Compatibility entry point for :mod:`scripts.serving.serving_smoke`."""

import importlib
import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    runpy.run_module("scripts.serving.serving_smoke", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("scripts.serving.serving_smoke")
