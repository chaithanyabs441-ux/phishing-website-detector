from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase10_thresholds_have_validation_source_and_no_manual_override():
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "thresholds.yaml").read_text(encoding="utf-8"))

    assert "manual_override" not in config
    assert config["source"].lower().startswith("validation")
    assert set(config["models"]) == {"url_only", "url_html"}
    for model in config["models"].values():
        assert model["selection_constraints"]["block_max_fpr"] == 0.001
        assert model["selection_constraints"]["warn_max_fpr"] == 0.02
