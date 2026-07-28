"""ROS 2 node wrapping the vehicle state estimation core filter."""

from __future__ import annotations

import numpy as np

from .bridge import (
    Ros2UnavailableError,
    imu_to_measurement,
    measurement_to_odometry,
    odometry_to_measurement,
)

try:  # ROS 2 is optional for the research algorithms and unit tests.
    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import Imu

    ROS2_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the host environment
    rclpy = None  # type: ignore[assignment]
    ROS2_AVAILABLE = False


if ROS2_AVAILABLE:
    from vehicle_state_estimation.filters.ekf import ExtendedKalmanFilter

    class EstimatorNode(Node):
        """Estimate planar ego motion from an odometry stream and monitor IMU health."""

        def __init__(self) -> None:
            super().__init__("vehicle_state_estimator")
            self.declare_parameter("input_odometry_topic", "/odometry/filtered")
            self.declare_parameter("input_imu_topic", "/imu/data")
            self.declare_parameter("output_topic", "/state_estimate")
            self.declare_parameter("diagnostics_topic", "/diagnostics")
            self.declare_parameter("measurement_std", 0.05)

            self._filter = ExtendedKalmanFilter.identity(6)
            self._measurement_noise = float(self.get_parameter("measurement_std").value) ** 2
            odometry_topic = str(self.get_parameter("input_odometry_topic").value)
            imu_topic = str(self.get_parameter("input_imu_topic").value)
            output_topic = str(self.get_parameter("output_topic").value)
            diagnostics_topic = str(self.get_parameter("diagnostics_topic").value)
            self._last_imu: np.ndarray | None = None

            self._publisher = self.create_publisher(Odometry, output_topic, 10)
            self._diagnostics_publisher = self.create_publisher(DiagnosticArray, diagnostics_topic, 10)
            self.create_subscription(Odometry, odometry_topic, self._on_odometry, 10)
            self.create_subscription(Imu, imu_topic, self._on_imu, 10)

        def _on_imu(self, message: Imu) -> None:
            self._last_imu = imu_to_measurement(message)

        def _on_odometry(self, message: Odometry) -> None:
            observation = odometry_to_measurement(message)
            estimate = self._filter.update(
                observation,
                np.eye(6),
                np.eye(6) * self._measurement_noise,
            )
            output = measurement_to_odometry(estimate)
            output.header = message.header
            output.child_frame_id = message.child_frame_id or "base_link"
            self._publisher.publish(output)
            self._publish_diagnostics(message.header)

        def _publish_diagnostics(self, header: object) -> None:
            diagnostics = DiagnosticArray()
            diagnostics.header = header
            status = DiagnosticStatus()
            status.name = "vehicle_state_estimator/imu"
            status.hardware_id = "vehicle_state_estimation_lab"
            if self._last_imu is None:
                status.level = DiagnosticStatus.WARN
                status.message = "waiting for IMU data"
            else:
                status.level = DiagnosticStatus.OK
                status.message = "IMU stream healthy"
                status.values.append(KeyValue(key="yaw_rate", value=f"{self._last_imu[3]:.6f}"))
            diagnostics.status.append(status)
            self._diagnostics_publisher.publish(diagnostics)


else:

    class EstimatorNode:  # pragma: no cover - exercised only without ROS 2
        """Placeholder that gives an actionable error when ROS 2 is not installed."""

        def __init__(self) -> None:
            raise Ros2UnavailableError(
                "rclpy is unavailable; source ROS 2 Humble/Iron before starting EstimatorNode"
            )


def main(args: list[str] | None = None) -> None:
    """Run the estimator node when invoked by ``ros2 run``."""

    if not ROS2_AVAILABLE:
        raise Ros2UnavailableError("rclpy is unavailable; install ROS 2 to run this node")
    rclpy.init(args=args)
    node = EstimatorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

