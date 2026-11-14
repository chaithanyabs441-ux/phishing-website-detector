"""Compatibility entry point for :mod:`src.training.train_url_html_svm`."""

import importlib
import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    runpy.run_module("src.training.train_url_html_svm", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("src.training.train_url_html_svm")
