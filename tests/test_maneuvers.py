import numpy as np
import pytest

from vehicle_state_estimation.simulation.maneuvers import (
    ManeuverConfig,
    ManeuverTrace,
    generate_maneuver,
    gnss_observation,
    imu_observation,
    wheel_speed_observation,
)


@pytest.mark.parametrize("kind", ["straight", "sine_steer", "double_lane_change", "low_friction_switch"])
def test_generate_maneuver_is_finite_and_deterministic(kind: str) -> None:
    config = ManeuverConfig(kind=kind, steps=180, dt=0.02)
    first = generate_maneuver(config)
    second = generate_maneuver(config)

    assert isinstance(first, ManeuverTrace)
    assert first.state.shape == (180, 4)
    assert first.friction.shape == (180,)
    assert np.all(np.isfinite(first.state))
    assert np.all(np.isfinite(first.steering))
    assert np.all(np.isfinite(first.excitation_energy))
    np.testing.assert_array_equal(first.state, second.state)
    np.testing.assert_array_equal(first.friction, second.friction)


def test_maneuvers_provide_bounded_excitation_and_low_friction_switch() -> None:
    sine = generate_maneuver(ManeuverConfig(kind="sine_steer", steps=400))
    low = generate_maneuver(ManeuverConfig(kind="low_friction_switch", steps=400))

    assert sine.excitation_energy[-1] > 1e-4
    assert np.max(np.abs(sine.steering)) <= sine.config.steering_amplitude + 1e-12
    assert low.friction[0] > low.friction[-1]
    assert np.all((low.friction >= low.config.low_friction) & (low.friction <= low.config.friction))


def test_sensor_observation_callbacks_return_measurements_and_covariance() -> None:
    trace = generate_maneuver(ManeuverConfig(kind="double_lane_change", steps=80))
    imu, imu_cov = imu_observation(trace)
    wheel, wheel_cov = wheel_speed_observation(trace)
    gnss, gnss_cov = gnss_observation(trace)

    assert imu.shape == (80, 3)
    assert wheel.shape == (80, 4)
    assert gnss.shape == (80, 2)
    assert imu_cov.shape == (3, 3)
    assert wheel_cov.shape == (4, 4)
    assert gnss_cov.shape == (2, 2)
    assert np.all(np.linalg.eigvalsh(imu_cov) > 0)
    np.testing.assert_allclose(trace.observe_imu()[0], imu)
    np.testing.assert_allclose(trace.observe_wheel_speed()[0], wheel)
    np.testing.assert_allclose(trace.observe_gnss()[0], gnss)


@pytest.mark.parametrize("kind", ["invalid", "SINE", ""])
def test_invalid_maneuver_kind_is_rejected(kind: str) -> None:
    with pytest.raises(ValueError, match="kind"):
        ManeuverConfig(kind=kind)


@pytest.mark.parametrize("acceleration", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_acceleration_is_rejected(acceleration: float) -> None:
    with pytest.raises(ValueError, match="acceleration"):
        ManeuverConfig(acceleration=acceleration)


def test_unreasonably_large_step_is_rejected_before_integration() -> None:
    with pytest.raises(ValueError, match="dt"):
        ManeuverConfig(dt=1.0)
