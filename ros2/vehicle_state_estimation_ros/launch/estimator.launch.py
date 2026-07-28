"""Launch the optional vehicle state estimator with a reproducible parameter file."""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("vehicle_state_estimation_ros")
    return LaunchDescription(
        [
            Node(
                package="vehicle_state_estimation_ros",
                executable="estimator_node",
                name="vehicle_state_estimator",
                output="screen",
                parameters=[f"{package_share}/config/estimator.yaml"],
            )
        ]
    )
