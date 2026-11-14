"""Apply trusted fitted calibration artifacts without training dependencies."""

import numpy as np
from scipy.special import logit


def apply_calibrator(calibrator, raw_probabilities):
    raw_probabilities = np.asarray(raw_probabilities, dtype=np.float64)
    if calibrator["method"] == "platt":
        features = logit(np.clip(raw_probabilities, 1e-8, 1.0 - 1e-8)).reshape(-1, 1)
        return calibrator["model"].predict_proba(features)[:, 1]
    if calibrator["method"] == "isotonic":
        return calibrator["model"].predict(raw_probabilities)
    raise ValueError(f"Unsupported calibrator: {calibrator['method']}")
