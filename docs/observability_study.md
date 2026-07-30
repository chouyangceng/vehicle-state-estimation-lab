# 可观测性与传感器消融实验

## 研究问题

车辆状态估计不仅取决于滤波器，也取决于工况是否为目标状态提供了足够激励。本实验固定低阶自行车模型和有限差分步长，只改变转向工况与传感器组合，回答“哪些状态方向能被当前传感器组合辨识”这一设计问题。

状态向量为 `x = [vx, vy, yaw_rate, yaw]`。每个时刻对观测模型求 Jacobian `H_k`，并按独立噪声协方差 `R_k` 累加经验信息矩阵：

```text
G = Σ H_kᵀ R_k⁻¹ H_k
```

有效秩表示有多少个状态方向获得了显著信息；正则化条件数反映数值病态程度；CRLB 给出局部无偏估计的理论标准差下界；信息增益是相对 IMU 基线的正则化 log-det 差值。

## 使用方式

```bash
python experiments/observability_study.py --fast --seed 7
python experiments/observability_study.py \
  --config configs/observability.yaml \
  --output artifacts/observability-study
```

`--fast` 使用较短轨迹，适合 CI 和开发；默认模式使用更长轨迹，适合记录结果。`--seed` 当前只作为实验元数据保存（`seed_semantics=metadata_only_deterministic_replay`），用于标记可复现实验；当前确定性轨迹不依赖随机采样，未来接入随机噪声和 Monte Carlo 时可沿用该接口。

配置文件使用项目内置的轻量 YAML 子集解析器，不依赖 PyYAML。解析器会自动去除 UTF-8 BOM，并对未知键、缩进/列表格式错误以及空的 `maneuvers` 或 `sensor_suites` 直接抛出带行号的 `ValueError`，避免静默回退到默认配置。

## 输出字段

`results.json` 的 `results` 数组每项对应一个工况与传感器组合：

- `gramian`：4×4 对称信息矩阵。
- `effective_rank`、`condition_number`：可观测性质量指标。
- `information_gain`：相对 `baseline`（默认 IMU）的信息增益。
- `crlb`、`crlb_unbounded`：逐状态 CRLB；不可观测方向使用 JSON `null`。
- `ill_conditioned`、`low_speed`：安全提示标志。

`ranking.csv` 用一个透明的启发式分数排序（有效秩、信息增益、条件数和 CRLB 的组合），仅用于选取值得深入分析的组合，不能替代统计检验。只要某个状态方向的 CRLB 不可界定（JSON 中为 `null`），该组合的分数就是 `-inf`，不会被误排在所有方向都有有限下界的组合之前。

## 假设与局限

模型采用一阶横向/横摆动力学、理想时间同步和对角高斯噪声。Gramian 是工作点附近的局部线性化结果，低速时纵向速度和航向可能天然难以区分；条件数中的正则化只改善数值稳定性，不会创造真实信息。实验尚未包含轮胎温度、侧倾、执行器饱和或真实 CAN 时间戳。

建议的后续工作是：使用实车数据重新标定 `R`；在不同速度和附着系数下做 Monte Carlo 置信区间；将结果与 EKF/UKF 的实际 RMSE 和 NEES 一起报告；最后在 ROS 2 或 CARLA 中进行在线验证。
