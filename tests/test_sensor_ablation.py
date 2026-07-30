import json

import numpy as np
import pytest

from vehicle_state_estimation.experiments.sensor_ablation import (
    AblationResult,
    SensorSuite,
    run_sensor_ablation,
)
from vehicle_state_estimation.simulation.maneuvers import ManeuverConfig, generate_maneuver


def test_sensor_suite_normalizes_and_rejects_unknown_sensor() -> None:
    suite = SensorSuite("combo", sensors=("gnss", "imu", "wheel_speed"))

    assert suite.sensors == ("imu", "wheel_speed", "gnss")
    with pytest.raises(ValueError, match="unknown sensor"):
        SensorSuite("bad", sensors=("camera",))


def test_ablation_is_deterministic_and_baseline_relative() -> None:
    trace = generate_maneuver(ManeuverConfig(kind="sine_steer", steps=60))
    suites = (
        SensorSuite("imu", ("imu",)),
        SensorSuite("imu_gnss", ("imu", "gnss")),
        SensorSuite("all", ("imu", "wheel_speed", "gnss")),
    )
    first = run_sensor_ablation(trace, suites=suites, baseline="imu", finite_difference_step=1e-5)
    second = run_sensor_ablation(trace, suites=suites, baseline="imu", finite_difference_step=1e-5)

    assert [item.suite for item in first] == ["imu", "imu_gnss", "all"]
    assert [item.suite for item in first] == [item.suite for item in second]
    np.testing.assert_allclose(first[1].gramian, second[1].gramian)
    assert first[0].information_gain == pytest.approx(0.0)
    assert first[1].information_gain > 0.0
    assert first[1].crlb.shape == (4,)
    assert isinstance(first[0].ill_conditioned, bool)


def test_ablation_serialization_writes_json_and_csv(tmp_path) -> None:
    trace = generate_maneuver(ManeuverConfig(kind="straight", steps=30))
    results = run_sensor_ablation(trace, suites=(SensorSuite("imu", ("imu",)),))
    result = results[0]
    assert isinstance(result, AblationResult)

    json_path = result.to_json(tmp_path / "result.json")
    csv_path = AblationResult.write_csv(results, tmp_path / "results.csv")
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload["suite"] == "imu"
    assert payload["crlb"]
    assert "information_gain" in csv_path.read_text(encoding="utf-8").splitlines()[0]


def test_ablation_rejects_invalid_baseline_and_empty_suite() -> None:
    trace = generate_maneuver(ManeuverConfig(steps=20))
    with pytest.raises(ValueError, match="baseline"):
        run_sensor_ablation(trace, suites=(SensorSuite("imu", ("imu",)),), baseline="gnss")
    with pytest.raises(ValueError, match="at least one"):
        SensorSuite("empty", sensors=())
