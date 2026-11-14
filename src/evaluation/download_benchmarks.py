"""Compatibility entry point for :mod:`src.data.download_benchmarks`."""

import importlib
from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    runpy.run_module("src.data.download_benchmarks", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("src.data.download_benchmarks")
