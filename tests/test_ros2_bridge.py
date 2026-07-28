from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from vehicle_state_estimation_ros.bridge import (
    imu_to_measurement,
    navsatfix_is_valid,
    navsatfix_to_local_position,
    odometry_to_measurement,
    quaternion_to_yaw,
    wheel_speed_to_velocity,
)


def test_imu_to_measurement_uses_si_fields() -> None:
    msg = SimpleNamespace(
        linear_acceleration=SimpleNamespace(x=1.2, y=-0.4, z=9.81),
        angular_velocity=SimpleNamespace(x=0.1, y=0.2, z=-0.3),
    )

    measurement = imu_to_measurement(msg)

    np.testing.assert_allclose(measurement, [1.2, -0.4, 9.81, -0.3])


def test_odometry_to_measurement_extracts_vehicle_velocity_and_position() -> None:
    msg = SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=12.0, y=-2.5),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        ),
        twist=SimpleNamespace(
            twist=SimpleNamespace(
                linear=SimpleNamespace(x=8.0, y=0.4),
                angular=SimpleNamespace(z=0.08),
            )
        ),
    )

    measurement = odometry_to_measurement(msg)

    np.testing.assert_allclose(measurement, [8.0, 0.4, 0.08, 12.0, -2.5, 0.0])


def test_quaternion_to_yaw_rejects_zero_norm_quaternion() -> None:
    with pytest.raises(ValueError, match="non-zero norm"):
        quaternion_to_yaw(SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0))


def test_navsatfix_to_local_position_uses_equirectangular_enu_projection() -> None:
    reference = SimpleNamespace(latitude=30.0, longitude=114.0, altitude=20.0)
    fix = SimpleNamespace(latitude=30.0001, longitude=114.0002, altitude=23.5)

    position = navsatfix_to_local_position(fix, reference)

    assert position.shape == (3,)
    np.testing.assert_allclose(position, [19.26, 11.12, 3.5], atol=0.15)


def test_navsatfix_to_local_position_rejects_invalid_coordinates() -> None:
    reference = SimpleNamespace(latitude=30.0, longitude=114.0, altitude=20.0)
    invalid_fix = SimpleNamespace(latitude=91.0, longitude=114.0, altitude=23.5)

    with pytest.raises(ValueError, match="latitude"):
        navsatfix_to_local_position(invalid_fix, reference)


def test_navsatfix_is_valid_rejects_no_fix_and_nan() -> None:
    no_fix = SimpleNamespace(
        status=SimpleNamespace(status=-1), latitude=30.0, longitude=114.0, altitude=20.0
    )
    nan_fix = SimpleNamespace(
        status=SimpleNamespace(status=0), latitude=float("nan"), longitude=114.0, altitude=20.0
    )

    assert navsatfix_is_valid(no_fix) is False
    assert navsatfix_is_valid(nan_fix) is False


def test_wheel_speed_to_velocity_accepts_joint_state_angular_velocity() -> None:
    msg = SimpleNamespace(velocity=[10.0, 10.2, 9.8, 10.1])

    velocity = wheel_speed_to_velocity(msg, wheel_radius=0.3)

    assert velocity == pytest.approx(3.0075)
