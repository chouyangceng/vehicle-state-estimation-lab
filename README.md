# Vehicle State Estimation Lab

面向车辆工程本科生的车辆状态估计与路面附着识别实验平台。项目用可解释的车辆动力学、Fiala 轮胎模型、EKF/UKF 和传感器故障注入，研究质心侧偏角、横摆率和附着系数等难以直接测量的状态。

## 30 秒运行

    python -m pip install -e .
    python examples/quickstart.py

默认结果写入 `artifacts/state-estimation/`，包括状态曲线、CSV 轨迹和 JSON 指标。

## 完整实验

    python experiments/run_experiment.py --seed 7 --steps 200
    python -m pytest -q
    python -m ruff check .

## 高级基准实验

    python examples/advanced_benchmark.py

高级示例会运行可复现车辆传感器仿真、在线附着系数估计、GNSS 失效注入和 Monte Carlo 统计。新增模块包括 `simulation/vehicle_simulator.py`、`simulation/faults.py`、`estimators/friction.py`、`filters/imm.py` 和 `metrics/monte_carlo.py`。

核心代码位于 `src/vehicle_state_estimation/`。`models` 保存车辆和轮胎模型，`filters` 保存 EKF/UKF，`simulation` 提供噪声、延迟、丢包和偏置故障，`metrics` 提供 RMSE/NIS/NEES。

## 研究问题

- 运动学和动力学自行车模型在不同速度下有什么差异？
- 为什么横向速度和质心侧偏角难以直接测量？
- 传感器漂移、延迟和失锁如何影响状态估计？
- 如何用一致性指标判断滤波器是否过度自信？
- 多模型概率如何在直线、转弯和低附着工况之间切换？
- 单次实验结果与 Monte Carlo 均值、标准差和 P95 有什么区别？

## 可选扩展

项目核心不依赖 CARLA、ROS 2 或 GPU。可在后续加入 CARLA IMU/GNSS 适配器、ROS 2 消息接口和 nuScenes CAN bus 离线读取。没有这些依赖时，合成实验仍可完整复现。

## 局限性

当前结果来自可控仿真和合成传感器，不能替代真实车辆试验。轮胎参数、传感器噪声和车辆质量需要根据具体车辆重新标定。

## License

Apache-2.0

## ROS 2 接口

项目包含可选的 ROS 2 Python 包：`ros2/vehicle_state_estimation_ros`。它提供
`sensor_msgs/Imu`、`nav_msgs/Odometry` 和 `diagnostic_msgs/DiagnosticArray` 的
标准消息接口，以及 `ros2 launch vehicle_state_estimation_ros estimator.launch.py`
启动文件。详细安装步骤、话题和参数见 [`docs/ros2.md`](docs/ros2.md)。核心算法不
依赖 ROS 2，未安装 `rclpy` 时仍可运行原有仿真与测试。
