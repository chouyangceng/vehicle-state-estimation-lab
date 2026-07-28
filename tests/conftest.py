from __future__ import annotations

import sys
from pathlib import Path

ROS2_PACKAGE = Path(__file__).parents[1] / "ros2" / "vehicle_state_estimation_ros"
if str(ROS2_PACKAGE) not in sys.path:
    sys.path.insert(0, str(ROS2_PACKAGE))
