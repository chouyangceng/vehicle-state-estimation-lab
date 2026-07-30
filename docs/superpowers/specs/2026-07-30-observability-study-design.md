# 车辆状态估计可观测性与参数可辨识性研究设计

## 目标

把 `vehicle-state-estimation-lab` 从“能运行滤波器的实验平台”提升为可复现实验研究工具，回答：在不同车辆激励下，IMU、轮速、GNSS 传感器组合对车辆侧向状态和路面附着系数的可观测性有何影响？

## 研究对象

- 状态向量：纵向速度、侧向速度、横摆率和航向角。
- 待辨识参数：路面附着系数、前后轴等效侧偏刚度（先提供可选参数列）。
- 观测组合：IMU、轮速、GNSS 位置，支持单传感器和组合消融。
- 激励工况：直线巡航、正弦转向、双移线近似、低附着切换。

## 数学指标

对离散非线性系统在名义轨迹附近进行有限差分线性化，累计经验可观测 Gramian：

\[
W_o = \sum_k \Phi_k^T H_k^T R_k^{-1}H_k\Phi_k
\]

其中 `Φ_k` 是状态转移灵敏度，`H_k` 是观测雅可比，`R_k` 是传感器协方差。输出：

- 有效秩：基于相对奇异值阈值判断可观测方向数量。
- 条件数：最大/最小有效奇异值比，反映数值病态程度。
- 信息增益：相对 IMU-only 的 `logdet(W + λI)` 增量。
- CRLB：`pinv(W + λI)` 对角线平方根，作为局部理论下界。

低速场景使用显式速度门限和正则化 `λ`，避免把数值奇异直接解释为物理不可观测。

## 软件结构

- `src/vehicle_state_estimation/metrics/observability.py`：指标计算、有限差分灵敏度、输入校验。
- `src/vehicle_state_estimation/simulation/maneuvers.py`：生成标准化转向/附着工况。
- `src/vehicle_state_estimation/experiments/sensor_ablation.py`：传感器组合与工况矩阵运行器。
- `experiments/observability_study.py`：CLI、结果文件、图表和结论摘要。
- `configs/observability.yaml`：工况、噪声、差分步长、正则化和传感器组合配置。
- `tests/test_observability.py`：矩阵性质、低速正则化、传感器排序和可复现性测试。
- `docs/observability_study.md`：数学推导、解释边界、实验步骤和结果阅读指南。

## 数据流

1. 根据固定 seed 生成工况和真值轨迹。
2. 为每个传感器组合生成带协方差的观测模型。
3. 对状态和可选参数做有限差分，累计 Gramian/Fisher 信息。
4. 计算秩、条件数、信息增益和 CRLB。
5. 输出逐工况明细、组合排名、PNG 图和中文 Markdown 摘要。

## 错误处理与边界

- 所有矩阵、协方差和有限差分步长必须有限且维度匹配。
- 协方差使用对称化和最小正则化；禁止用普通逆替代伪逆。
- 低速和短轨迹直接报告 `ill_conditioned=true`，不伪造可观测结论。
- 结果明确标注“基于仿真模型和局部线性化”，不声称等价于实车性能。

## 验证标准

- 新增测试全部通过，现有测试不回归。
- 对称半正定 Gramian 的最小特征值不低于数值容差。
- 固定 seed 下两次研究结果逐项一致。
- CLI 能在无 ROS 2 环境运行并生成 CSV/JSON/PNG/Markdown。
- Ruff、compileall 和 GitHub Actions 全部通过。
