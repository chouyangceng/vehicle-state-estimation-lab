from pathlib import Path


def test_experiment_writes_metrics():
    from vehicle_state_estimation.experiments import run

    output = run(seed=7, steps=50, output_dir=Path("artifacts/test-experiment"))
    assert (output / "metrics.json").exists()
    assert (output / "state_trace.csv").exists()
