# ROS 2 接口

本项目提供一个可选的 ROS 2 Humble/Iron Python 包，位于
`ros2/vehicle_state_estimation_ros`。核心估计算法仍然是普通 Python 包，未安装
ROS 2 时可以继续运行全部仿真和单元测试。

## 安装与启动

在已经 source ROS 2 的工作空间中，先安装本仓库的核心算法（节点会复用其中的
`ExtendedKalmanFilter`），再构建 ROS 2 包：

```bash
python3 -m pip install -e .
mkdir -p ~/ros2_ws/src
ln -s "$PWD/ros2/vehicle_state_estimation_ros" ~/ros2_ws/src/vehicle_state_estimation_ros
cd ~/ros2_ws
colcon build --symlink-install --packages-select vehicle_state_estimation_ros
source install/setup.bash
ros2 run vehicle_state_estimation_ros estimator_node
```

或者使用参数文件启动：

```bash
ros2 launch vehicle_state_estimation_ros estimator.launch.py
```

默认接口如下：

| 方向 | 话题 | 消息类型 | 说明 |
| --- | --- | --- | --- |
| 输入 | `/odometry/filtered` | `nav_msgs/msg/Odometry` | 平面速度、位置和航向观测 |
| 输入 | `/imu/data` | `sensor_msgs/msg/Imu` | IMU 加速度和横摆角速度 |
| 输入 | `/gnss/fix` | `sensor_msgs/msg/NavSatFix` | WGS-84 定位，首帧作为 ENU 原点 |
| 输入 | `/wheel_states` | `sensor_msgs/msg/JointState` | 轮端角速度，按 `wheel_radius` 转为 m/s |
| 输出 | `/state_estimate` | `nav_msgs/msg/Odometry` | EKF 平滑后的车辆状态 |
| 输出 | `/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | IMU 数据健康状态 |

## 设计说明

- `bridge.py` 只依赖 NumPy，使用 duck-typed 字段读取 ROS 消息，因此可以在没有
  `rclpy` 的 CI 环境中测试。
- `NavSatFix` 使用小范围等距近似投影到 ENU；`JointState.velocity` 的多轮速度取均值，
  也兼容离线记录中常见的 `Float64.data` 字段。
- 当前示例节点的 EKF 更新使用 `Odometry` 六维观测；IMU、GNSS 和轮速输入用于链路
  监视与诊断，尚未自动融合进 EKF，避免在缺少时间同步和协方差标定时产生错误融合。
  轮速约定为车辆前进方向为正，若底盘驱动器输出相反符号，应在驱动桥或参数层取反。
- `estimator_node.py` 在 ROS 2 不可用时不会破坏核心包导入，而是在尝试启动节点
  时给出明确的安装提示。
- 状态向量为 `[vx, vy, yaw_rate, x, y, yaw]`，单位分别为 m/s、rad/s、m 和 rad。
- 该节点用于算法验证和消息链路演示；真实车辆部署前应根据车辆标定结果重新设置
  协方差、坐标系和时间同步策略。
- 核心项目要求 Python 3.11 或更高版本；ROS 2 Humble/Iron 的 Python 环境应满足该
  版本要求，或在独立虚拟环境中安装核心包后再构建 ROS 2 工作空间。
- 发布器和订阅器使用深度为 10 的默认可靠 QoS，适合本地实验；跨主机或高丢包链路
  可在后续实验中改为 sensor-data QoS，并确保输入消息的 `header.stamp` 来自同一时钟。

## 本地验证

```bash
python -m pytest tests/test_ros2_bridge.py -q
```

在未安装 ROS 2 的机器上，测试会验证消息字段转换而不会尝试启动节点；安装 ROS 2
后可在 colcon 工作空间中进一步运行 launch/integration 测试。
