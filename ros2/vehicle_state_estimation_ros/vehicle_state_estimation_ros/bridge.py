"""ROS 2 message adapters with no hard dependency on ROS 2 at import time.

The functions that read messages accept duck-typed objects, which keeps the
algorithm package testable on a regular Python installation. Message
construction is isolated in :func:`measurement_to_odometry` and imports
``nav_msgs`` only when it is called in a ROS 2 process.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class Ros2UnavailableError(RuntimeError):
    """Raised when a ROS message factory is used outside a ROS 2 environment."""


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    return vector


def imu_to_measurement(msg: object) -> np.ndarray:
    """Convert ``sensor_msgs/msg/Imu`` to ``[ax, ay, az, yaw_rate]`` SI values."""

    acceleration = msg.linear_acceleration  # type: ignore[attr-defined]
    angular_velocity = msg.angular_velocity  # type: ignore[attr-defined]
    return _finite_vector(
        [acceleration.x, acceleration.y, acceleration.z, angular_velocity.z], "IMU measurement"
    )


def quaternion_to_yaw(quaternion: object) -> float:
    """Return yaw (rad) from a quaternion-like object using the ROS ENU convention."""

    x = float(quaternion.x)  # type: ignore[attr-defined]
    y = float(quaternion.y)  # type: ignore[attr-defined]
    z = float(quaternion.z)  # type: ignore[attr-defined]
    w = float(quaternion.w)  # type: ignore[attr-defined]
    norm = float(np.hypot(np.hypot(x, y), np.hypot(z, w)))
    if norm <= 1e-12:
        raise ValueError("quaternion must have a non-zero norm")
    x, y, z, w = (value / norm for value in (x, y, z, w))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(yaw)


def odometry_to_measurement(msg: object) -> np.ndarray:
    """Convert ``nav_msgs/msg/Odometry`` to ``[vx, vy, yaw_rate, x, y, yaw]``."""

    pose = msg.pose.pose  # type: ignore[attr-defined]
    twist = msg.twist.twist  # type: ignore[attr-defined]
    return _finite_vector(
        [
            twist.linear.x,
            twist.linear.y,
            twist.angular.z,
            pose.position.x,
            pose.position.y,
            quaternion_to_yaw(pose.orientation),
        ],
        "odometry measurement",
    )


def measurement_to_odometry(
    values: Sequence[float], *, frame_id: str = "odom", child_frame_id: str = "base_link"
) -> object:
    """Create a ``nav_msgs/msg/Odometry`` from a six-element estimate.

    ROS 2 is intentionally imported inside this function so that importing
    ``vehicle_state_estimation_ros.bridge`` remains possible without rclpy.
    """

    estimate = _finite_vector(values, "state estimate")
    if estimate.shape != (6,):
        raise ValueError("state estimate must contain [vx, vy, yaw_rate, x, y, yaw]")
    try:
        from nav_msgs.msg import Odometry
    except ImportError as exc:  # pragma: no cover - depends on ROS installation
        raise Ros2UnavailableError(
            "nav_msgs is unavailable; source a ROS 2 environment before publishing messages"
        ) from exc

    message = Odometry()
    message.header.frame_id = frame_id
    message.child_frame_id = child_frame_id
    message.pose.pose.position.x = float(estimate[3])
    message.pose.pose.position.y = float(estimate[4])
    half_yaw = float(estimate[5]) * 0.5
    message.pose.pose.orientation.z = float(np.sin(half_yaw))
    message.pose.pose.orientation.w = float(np.cos(half_yaw))
    message.twist.twist.linear.x = float(estimate[0])
    message.twist.twist.linear.y = float(estimate[1])
    message.twist.twist.angular.z = float(estimate[2])
    return message
