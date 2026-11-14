from src.training.training_smoke import run_smoke_training


def test_tiny_sample_classical_training_runs():
    average_precision = run_smoke_training()
    assert 0.0 <= average_precision <= 1.0
