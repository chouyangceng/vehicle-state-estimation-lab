import json

import numpy as np
import pytest

from vehicle_state_estimation.experiments.sensor_ablation import (
    AblationResult,
    SensorSuite,
    gnss_measurement_model,
    imu_measurement_model,
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


def test_measurement_models_match_maneuver_trace_observations() -> None:
    trace = generate_maneuver(ManeuverConfig(kind="sine_steer", steps=40, acceleration=1.2))
    imu, _ = trace.observe_imu()
    gnss, _ = trace.observe_gnss()

    for index, state in enumerate(trace.state):
        np.testing.assert_allclose(
            imu_measurement_model(
                state,
                steering=trace.steering[index],
                acceleration=trace.acceleration[index],
            ),
            imu[index],
            atol=1e-10,
        )
        world_velocity = np.array(
            [
                state[0] * np.cos(state[3]) - state[1] * np.sin(state[3]),
                state[0] * np.sin(state[3]) + state[1] * np.cos(state[3]),
            ]
        )
        offset = trace.position[index] - trace.config.dt * world_velocity
        np.testing.assert_allclose(
            gnss_measurement_model(state, position_offset=offset, dt=trace.config.dt),
            gnss[index],
            atol=1e-10,
        )


def test_json_serialization_represents_unbounded_crlb_as_null_and_mask(tmp_path) -> None:
    trace = generate_maneuver(ManeuverConfig(kind="straight", steps=30))
    result = run_sensor_ablation(trace, suites=(SensorSuite("imu", ("imu",)),))[0]

    payload = json.loads(result.to_json(tmp_path / "result.json").read_text(encoding="utf-8"))

    assert "Infinity" not in (tmp_path / "result.json").read_text(encoding="utf-8")
    # Constant-speed straight-line IMU data leave longitudinal velocity
    # unobservable through ``ax=acceleration``; lateral velocity and yaw rate
    # remain observable from the lateral acceleration and gyro channels.
    assert payload["crlb_unbounded"] == [True, False, False, True]
    assert payload["crlb"][0] is None
    assert all(value is not None for value in payload["crlb"][1:3])
    assert payload["crlb"][3] is None


@pytest.mark.parametrize(
    "field, value",
    [
        ("state", np.zeros((4, 3))),
        ("position", np.zeros((3, 2))),
        ("time", np.array([0.0, np.nan])),
        ("steering", np.zeros(3)),
    ],
)
def test_ablation_rejects_malformed_trace_arrays(field: str, value: np.ndarray) -> None:
    from dataclasses import replace

    trace = generate_maneuver(ManeuverConfig(steps=4))
    malformed = replace(trace, **{field: value})
    with pytest.raises(ValueError, match="trace"):
        run_sensor_ablation(malformed)


def test_ablation_requires_strictly_positive_regularization() -> None:
    trace = generate_maneuver(ManeuverConfig(steps=20))
    with pytest.raises(ValueError, match="regularization.*positive"):
        run_sensor_ablation(trace, regularization=0.0)
