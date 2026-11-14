"""Compatibility entry point for :mod:`src.evaluation.calibration_validation`."""

import importlib
import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    runpy.run_module("src.evaluation.calibration_validation", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("src.evaluation.calibration_validation")
