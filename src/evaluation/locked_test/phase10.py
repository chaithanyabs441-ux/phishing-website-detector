"""Compatibility entry point for :mod:`src.evaluation.locked_test.evaluate_serving_routes_phase10`."""

import importlib
import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    runpy.run_module("src.evaluation.locked_test.evaluate_serving_routes_phase10", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("src.evaluation.locked_test.evaluate_serving_routes_phase10")
