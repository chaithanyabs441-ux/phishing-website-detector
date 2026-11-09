import hashlib

import joblib
import numpy as np
import pytest

from src.serving.model_registry import ModelRegistry, _apply_calibrator
from src.evaluation.phase7_calibration import apply_calibrator


def test_registry_checks_manifest_hash_before_loading(tmp_path):
    artifact = tmp_path / "safe.joblib"
    joblib.dump({"trusted": True}, artifact)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    registry = ModelRegistry.__new__(ModelRegistry)
    registry.root = tmp_path.resolve()
    registry.entries = {"safe.joblib": {"sha256": digest}}
    assert registry._load_verified("safe.joblib") == {"trusted": True}
    registry.entries["safe.joblib"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 verification failed"):
        registry._load_verified("safe.joblib")


def test_registry_rejects_paths_outside_project_root(tmp_path):
    registry = ModelRegistry.__new__(ModelRegistry)
    registry.root = tmp_path.resolve()
    registry.entries = {}
    with pytest.raises(ValueError, match="escapes the project root"):
        registry._load_verified("../outside.joblib")


def test_serving_platt_transform_matches_phase7_calibration():
    from webapp.app import REGISTRY

    raw_probabilities = [0.01, 0.5, 0.83662053, 0.99958644]
    expected = apply_calibrator(REGISTRY.calibrator, raw_probabilities)
    actual = _apply_calibrator(REGISTRY.calibrator, raw_probabilities)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
