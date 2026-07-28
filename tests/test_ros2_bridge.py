from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROS2_PACKAGE = Path(__file__).parents[1] / "ros2" / "vehicle_state_estimation_ros"
sys.path.insert(0, str(ROS2_PACKAGE))

from vehicle_state_estimation_ros.bridge import (  # noqa: E402
    imu_to_measurement,
    odometry_to_measurement,
    quaternion_to_yaw,
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
