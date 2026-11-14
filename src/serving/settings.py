"""Load local runtime settings and domain context files."""

import os
from pathlib import Path

import yaml


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def read_domain_file(path):
    return {
        line.strip().lower().lstrip(".")
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def load_env_file(path):
    """Load local environment values without overriding exported values."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("\"'"))
