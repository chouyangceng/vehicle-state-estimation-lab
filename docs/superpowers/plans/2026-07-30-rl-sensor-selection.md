# RL 自适应传感器选择实施计划

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 构建仅依赖 NumPy 的可复现 Q-learning 研究平台，让智能体在 IMU、轮速、GNSS 组合之间动态决策，并输出可比较、可部署、可解释的实验结果。

**Architecture:** 复用现有 ManeuverTrace 与传感器线性化模型；将窗口信息矩阵、离散状态、环境、Q-learning、基线评估、CLI 和 ROS 2 推理解耦。训练只在线下进行，ROS 2 仅加载已验证策略并安全推理。

**Tech Stack:** Python 3.10+, NumPy, Matplotlib Agg, pytest, Ruff, setuptools 和现有 ROS 2 Python 包；核心测试不依赖 Gymnasium、PyTorch、GPU 或 rclpy。

---

## 文件边界

- src/vehicle_state_estimation/experiments/sensor_ablation.py：新增窗口传感器信息矩阵 API。
- src/vehicle_state_estimation/rl/discretization.py：分箱和有限状态编码。
- src/vehicle_state_estimation/rl/environment.py：动作、协方差转移、奖励、故障和预览。
- src/vehicle_state_estimation/rl/q_learning.py：Q 表、训练、策略序列化。
- src/vehicle_state_estimation/rl/baselines.py：四个基线策略。
- src/vehicle_state_estimation/rl/evaluation.py：统一 episode 和统计指标。
- experiments/rl_sensor_selection.py：配置、训练、评估、绘图和中文报告。
- ros2/.../policy_bridge.py：不依赖 rclpy 的策略推理桥。
- ros2/.../policy_node.py：可选 ROS 2 推理节点。
- configs/rl_sensor_selection.yaml、docs/rl_sensor_selection.md：配置和研究说明。
- tests/test_rl_*.py、tests/test_ros2_policy.py：行为和回归测试。

### Task 1: 暴露窗口传感器信息

Files:
- Modify: src/vehicle_state_estimation/experiments/sensor_ablation.py
- Modify: src/vehicle_state_estimation/experiments/__init__.py
- Test: tests/test_sensor_ablation.py

- [ ] Step 1: 写失败测试

测试 sensor_information_matrix(trace, sensors, start, stop) 的四维输出、对称性、半正定性、独立传感器可加性；并用 start=-1、stop=start、stop 超出样本数验证 ValueError。

    trace = generate_maneuver(ManeuverConfig(kind="sine_steer", steps=40))
    imu = sensor_information_matrix(trace, ("imu",), start=5, stop=15)
    wheel = sensor_information_matrix(trace, ("wheel_speed",), start=5, stop=15)
    both = sensor_information_matrix(trace, ("imu", "wheel_speed"), start=5, stop=15)
    assert both.shape == (4, 4)
    np.testing.assert_allclose(both, both.T, atol=1e-12)
    assert np.linalg.eigvalsh(both).min() >= -1e-9
    np.testing.assert_allclose(both, imu + wheel, rtol=1e-8, atol=1e-8)

- [ ] Step 2: 运行 RED

    python -m pytest tests/test_sensor_ablation.py -q --basetemp .pytest-rl-task1

预期：sensor_information_matrix 不可导入。

- [ ] Step 3: 实现最小 API

签名为：

    def sensor_information_matrix(
        trace: ManeuverTrace,
        sensors: Sequence[str],
        *,
        start: int = 0,
        stop: int | None = None,
        finite_difference_step: float = 1e-5,
        covariance_floor: float = 1e-9,
    ) -> FloatArray:

调用现有 _measurement_model、_sensor_covariance、finite_difference_jacobian；对每个时间样本累加 H.T @ solve(R, H)，最后对称化。复用该 API 改造 run_sensor_ablation 的整段 Gramian 计算，避免 RL 和消融研究使用不同物理模型。

- [ ] Step 4: GREEN 和回归

    python -m pytest tests/test_sensor_ablation.py tests/test_observability.py -q --basetemp .pytest-rl-task1
    python -m ruff check src/vehicle_state_estimation/experiments tests/test_sensor_ablation.py --no-cache

- [ ] Step 5: 提交

    git add src/vehicle_state_estimation/experiments tests/test_sensor_ablation.py
    git commit -m "feat: expose windowed sensor information"

### Task 2: 实现状态离散化

Files:
- Create: src/vehicle_state_estimation/rl/__init__.py
- Create: src/vehicle_state_estimation/rl/discretization.py
- Test: tests/test_rl_discretization.py

- [ ] Step 1: 写失败测试

覆盖默认状态数 3 x 3 x 4 x 8 x 8 = 2304、边界 [2, 15] m/s、[0.02, 0.12] rad/s、[0.25, 0.50, 0.80]、健康掩码 0 到 7、上一动作 -1 到 6。测试 NaN、负值、非法掩码、非递增边界被拒绝。

- [ ] Step 2: RED

    python -m pytest tests/test_rl_discretization.py -q --basetemp .pytest-rl-task2

预期：RL 包尚不存在。

- [ ] Step 3: 实现

新增 DiscretizationConfig 和 StateDiscretizer。

    class DiscretizationConfig:
        speed_edges = (2.0, 15.0)
        yaw_rate_edges = (0.02, 0.12)
        uncertainty_edges = (0.25, 0.50, 0.80)

    class StateDiscretizer:
        shape = (3, 3, 4, 8, 8)
        number_of_states = 2304
        encode(speed, absolute_yaw_rate, normalized_uncertainty,
               health_mask, previous_action) -> int
        decode(state) -> tuple[int, int, int, int, int]

使用 np.digitize 和 np.ravel_multi_index；所有输入先做有限值、范围和整数校验。并列状态编码必须可逆。

- [ ] Step 4: GREEN

    python -m pytest tests/test_rl_discretization.py -q --basetemp .pytest-rl-task2
    python -m ruff check src/vehicle_state_estimation/rl tests/test_rl_discretization.py --no-cache

- [ ] Step 5: 提交

    git add src/vehicle_state_estimation/rl tests/test_rl_discretization.py
    git commit -m "feat: add RL state discretization"

### Task 3: 构建传感器选择环境

Files:
- Create: src/vehicle_state_estimation/rl/environment.py
- Modify: src/vehicle_state_estimation/rl/__init__.py
- Test: tests/test_rl_environment.py

- [ ] Step 1: 写失败测试

测试 reset/step 确定性、传感器成本、奖励分量之和、非法动作回退、零健康掩码安全终止、preview_action 不改变环境、当前窗口信息矩阵秩与后验精度秩的区别。

    trace = generate_maneuver(ManeuverConfig(kind="sine_steer", steps=30))
    env = SensorSelectionEnv(trace, config=EnvironmentConfig(window_size=10))
    state = env.reset(seed=7)
    transition = env.step(6)
    assert transition.terminated is False
    assert transition.info.selected_sensors == ("imu", "wheel_speed", "gnss")
    assert transition.info.sensor_cost == pytest.approx(8.0)
    assert transition.reward == pytest.approx(
        sum(transition.info.reward_components.values())
    )

健康掩码 0b101 时动作 6 必须回退到动作 4；掩码为零必须返回 mask=0、severity=ERROR 并终止。

- [ ] Step 2: RED

    python -m pytest tests/test_rl_environment.py -q --basetemp .pytest-rl-task3

- [ ] Step 3: 实现动作和配置

固定动作顺序：

    ACTIONS = (
        ("imu",), ("wheel_speed",), ("gnss",),
        ("imu", "wheel_speed"), ("imu", "gnss"),
        ("wheel_speed", "gnss"), ("imu", "wheel_speed", "gnss"),
    )
    SENSOR_BITS = {"imu": 1, "wheel_speed": 2, "gnss": 4}

EnvironmentConfig 默认 window_size=10、process_variance=(0.08,0.08,0.02,0.02)、initial_variance=(4,4,0.5,0.5)、variance_cap=(100,100,25,25)、sensor_costs=(2,1,5)、rank_tolerance=1e-9。RewardWeights 默认 uncertainty=2.0、sensor_cost=0.25、switching=0.10、information_gain=0.40、unobservable=3.0、invalid_action=2.0。验证数组长度、有限值、正窗口、非负成本和权重。

- [ ] Step 4: 实现转移和奖励

每步取一个窗口。先计算 prior = covariance + diag(process_variance * window_duration)，再计算当前传感器信息 J，后验总精度为 pinv(prior) + J。对后验精度做特征分解，在有效方向使用倒数，在数值零空间使用 variance_cap 作为训练代理。

可观测性只用 J 的有效秩判断。奖励分量必须明确记录：

    reward = -w_uncertainty * normalized_uncertainty
             -w_cost * normalized_sensor_cost
             -w_switch * switched
             +w_information * information_gain
             -w_unobservable * (rank(J) < 4)
             -w_invalid * fallback

preview_action 通过复制内部 covariance、window_index、previous_action 计算，不写回环境；step 只写回一次。下一状态特征是下一窗口速度、绝对横摆率、不确定度、健康掩码和上一动作。

- [ ] Step 5: GREEN、回归和提交

    python -m pytest tests/test_rl_environment.py tests/test_sensor_ablation.py tests/test_maneuvers.py -q --basetemp .pytest-rl-task3
    python -m ruff check src/vehicle_state_estimation/rl tests/test_rl_environment.py --no-cache
    git add src/vehicle_state_estimation/rl tests/test_rl_environment.py
    git commit -m "feat: add adaptive sensor selection environment"

### Task 4: 实现 Q-learning 和策略序列化

Files:
- Create: src/vehicle_state_estimation/rl/q_learning.py
- Modify: src/vehicle_state_estimation/rl/__init__.py
- Test: tests/test_q_learning.py

- [ ] Step 1: 写失败测试

覆盖 Q 更新公式、terminal 不 bootstrap、epsilon 起止值、固定动作并列选择、相同 seed 产生相同 Q 表、空 valid_actions 拒绝、策略文件往返一致、未知版本/错误形状/NaN Q 值拒绝。

    agent = QLearningAgent(4, 2, seed=7)
    agent.q_table[1] = [2.0, 4.0]
    agent.update(0, 1, reward=3.0, next_state=1, terminated=False,
                 learning_rate=0.5, discount=0.9)
    assert agent.q_table[0, 1] == pytest.approx(3.3)

- [ ] Step 2: RED

    python -m pytest tests/test_q_learning.py -q --basetemp .pytest-rl-task4

- [ ] Step 3: 实现

TrainingConfig 字段为 episodes=400、learning_rate=0.15、discount=0.95、epsilon_start=1.0、epsilon_end=0.05、epsilon_decay_episodes=300；epsilon 线性衰减并验证范围。

QLearningAgent 保存 shape=(number_of_states, number_of_actions) 的 float64 Q 表，select_action 使用 np.random.Generator；探索随机，利用时 np.argmax 实现最低动作编号并列规则；terminal 更新的 bootstrap 为 0。

- [ ] Step 4: 训练和策略文件

train_q_learning 接收 episode_factory(index, rng)，使用独立的训练 RNG，返回 agent 和每 episode 的 reward、cost、uncertainty、unobservable_count。策略 JSON 必须含 format_version=1、actions、discretization、q_table、training、metadata；使用 ensure_ascii=False、allow_nan=False、sort_keys=True 和固定 separators，加载器验证精确动作顺序和 Q 表形状。

- [ ] Step 5: GREEN 和提交

    python -m pytest tests/test_q_learning.py tests/test_rl_environment.py -q --basetemp .pytest-rl-task4
    python -m ruff check src/vehicle_state_estimation/rl tests/test_q_learning.py --no-cache
    git add src/vehicle_state_estimation/rl tests/test_q_learning.py
    git commit -m "feat: add deterministic Q-learning policy"

### Task 5: 添加基线和公平评估

Files:
- Create: src/vehicle_state_estimation/rl/baselines.py
- Create: src/vehicle_state_estimation/rl/evaluation.py
- Modify: src/vehicle_state_estimation/rl/__init__.py
- Test: tests/test_rl_evaluation.py

- [ ] Step 1: 写失败测试

测试 AlwaysAllPolicy、LowestCostPolicy、RandomPolicy、GreedyInformationPolicy 只选 valid_actions；所有策略收到相同不可变 EpisodeSpec；结果含 mean/P95 uncertainty、return、normalized_cost、saving、switches、duty_cycle、unobservable_fraction。

- [ ] Step 2: RED

    python -m pytest tests/test_rl_evaluation.py -q --basetemp .pytest-rl-task5

- [ ] Step 3: 实现固定基线

AlwaysAllPolicy 从 valid_actions 选择传感器数量最多者；LowestCostPolicy 选择成本最低者；RandomPolicy 使用独立 seed；GreedyInformationPolicy 使用 env.preview_action(action).reward 最大者。每个策略用固定 action index 作为 tie-break。QTablePolicy 使用同一规则从 Q 表推理。

- [ ] Step 4: 实现 EpisodeSpec 和指标

build_evaluation_episodes 覆盖四种 manoeuvre 和四类健康配置：全健康、GNSS 暂时失效、IMU 暂时失效、轮速暂时失效；每个 repeat 产生不可变 episode_id、ManeuverConfig、健康调度和 seed。evaluate_policies 对同一 specs 顺序运行每个策略，记录每个窗口和每个工况，再聚合 mean/P95 uncertainty、return、relative cost、saving、switch count、sensor duty cycle、unobservable fraction 和 per-manoeuvre rows。

只有 cost 更低、uncertainty 不超过 always_all 的 tolerance、unobservable_fraction 不超过 tolerance 时才写 dominant=true，否则写 different_tradeoff。

- [ ] Step 5: GREEN 和提交

    python -m pytest tests/test_rl_evaluation.py tests/test_q_learning.py tests/test_rl_environment.py -q --basetemp .pytest-rl-task5
    python -m ruff check src/vehicle_state_estimation/rl tests/test_rl_evaluation.py --no-cache
    git add src/vehicle_state_estimation/rl tests/test_rl_evaluation.py
    git commit -m "feat: add RL baselines and evaluation"

### Task 6: 构建实验 CLI、配置和研究报告

Files:
- Create: experiments/rl_sensor_selection.py
- Create: configs/rl_sensor_selection.yaml
- Create: tests/test_rl_experiment.py
- Create: docs/rl_sensor_selection.md
- Modify: README.md
- Modify: docs/中文说明.md

- [ ] Step 1: 写失败验收测试

run_study(output_dir, seed=17, fast=True) 必须产生 policy.json、training_history.csv、evaluation.json、policy_comparison.csv、training_curve.png、tradeoff.png、action_usage.png、summary.md 八个文件。两次相同 seed 的 policy、history、evaluation 和 comparison CSV 字节一致；JSON 不含 NaN/Infinity。非空输出目录默认抛 FileExistsError 且保留用户文件。配置未知键、UTF-8 BOM 和错误列表都要有行号测试。

- [ ] Step 2: RED

    python -m pytest tests/test_rl_experiment.py -q --basetemp .pytest-rl-task6

- [ ] Step 3: 实现配置解析和编排

配置固定为 seed=7、episodes=400、evaluation_repeats=8、steps=240、window_size=10、learning_rate=0.15、discount=0.95、epsilon_start=1.0、epsilon_end=0.05、epsilon_decay_episodes=300、sensor_costs=[2,1,5]、speed_edges=[2,15]、yaw_rate_edges=[0.02,0.12]、uncertainty_edges=[0.25,0.50,0.80]、uncertainty_tolerance=0.10、unobservable_tolerance=0.02。

read_config 使用 UTF-8-sig，去除注释、拒绝未知键、报告原始行号、验证所有范围。--fast 仅改 episodes=24、evaluation_repeats=1、steps=60，不修改奖励、成本、分箱和 seed。run_study 用 SeedSequence(seed).spawn(2) 分离训练与评估 RNG，训练后冻结策略，用同一 EpisodeSpec 评估五个策略，最后才写文件。

- [ ] Step 4: 实现输出

JSON 使用 allow_nan=False；CSV 使用固定字段顺序；Matplotlib 在导入 pyplot 前使用 Agg 并关闭每个 figure。summary.md 必须写 dominance 或 different_tradeoff、奖励分量、种子、各策略权衡、假设和仿真限制。overwrite 只允许删除八个已知输出文件，禁止递归清空任意目录。

- [ ] Step 5: GREEN、冒烟、文档

    python -m pytest tests/test_rl_experiment.py -q --basetemp .pytest-rl-task6
    python experiments/rl_sensor_selection.py --fast --seed 7 --output .rl-smoke
    python -m ruff check experiments/rl_sensor_selection.py tests/test_rl_experiment.py --no-cache

docs/rl_sensor_selection.md、README.md、docs/中文说明.md 写明状态、动作、奖励、基线、指标、复现命令和仿真结论边界。

- [ ] Step 6: 提交

    git add experiments/rl_sensor_selection.py configs/rl_sensor_selection.yaml tests/test_rl_experiment.py docs/rl_sensor_selection.md README.md docs/中文说明.md
    git commit -m "feat: add RL sensor selection study"

### Task 7: 增加安全 ROS 2 策略推理

Files:
- Create: ros2/vehicle_state_estimation_ros/vehicle_state_estimation_ros/policy_bridge.py
- Create: ros2/vehicle_state_estimation_ros/vehicle_state_estimation_ros/policy_node.py
- Create: ros2/vehicle_state_estimation_ros/config/policy.yaml
- Create: ros2/vehicle_state_estimation_ros/launch/policy.launch.py
- Modify: ros2/vehicle_state_estimation_ros/setup.py
- Modify: ros2/vehicle_state_estimation_ros/package.xml
- Modify: docs/ros2.md
- Test: tests/test_ros2_policy.py

- [ ] Step 1: 写失败的 ROS 无关桥接测试

用 duck-typed Odometry 和临时 policy.json 测试正常推理、健康掩码导致的回退、健康掩码为零的 mask=0/ERROR、无效 covariance 变为 uncertainty=1.0 而不是崩溃。

    decision = runtime.decide(odometry, health_mask=0b111, previous_action=-1)
    assert decision.sensor_mask == 0b101
    assert decision.fallback is False

- [ ] Step 2: RED

    python -m pytest tests/test_ros2_policy.py -q --basetemp .pytest-rl-task7

- [ ] Step 3: 实现 PolicyRuntime

PolicyDecision 字段为 state、action、sensor_mask、fallback、severity、reason。PolicyRuntime.load 调用 QLearningAgent.load_policy 和 StateDiscretizer；decide 从 Odometry 的速度、横摆率、有限 covariance 对角线构造状态。policy action 包含不健康传感器时回退到全部健康传感器；健康掩码为零返回 sensor_mask=0、severity=ERROR。

- [ ] Step 4: 实现 ROS node 和打包

policy_node.py 只在 ROS 进程导入 rclpy。订阅 Odometry 和 DiagnosticArray，发布 std_msgs/UInt8 的 sensor_enable_mask 与 DiagnosticArray。参数为 policy_path、odometry_topic、diagnostics_topic、output_topic、input_timeout_sec。策略加载失败进入全健康回退并输出 ERROR。setup.py 增加 policy_node console script、配置和 launch 安装；package.xml 增加 std_msgs。

- [ ] Step 5: GREEN、回归和提交

    python -m pytest tests/test_ros2_policy.py tests/test_ros2_bridge.py -q --basetemp .pytest-rl-task7
    python -m ruff check ros2 tests/test_ros2_policy.py --no-cache
    git add ros2 tests/test_ros2_policy.py docs/ros2.md
    git commit -m "feat: add ROS 2 RL policy inference"

### Task 8: 最终科研审查、验证和 GitHub 交付

- [ ] Step 1: 完整验证

    $base = 'C:\Users\asjs\Documents\Codex\2026-07-25\e-02\pytest-rl-final'
    New-Item -ItemType Directory -Force -Path $base | Out-Null
    python -m pytest -q --basetemp $base
    python -m ruff check . --no-cache
    python -m compileall -q src experiments ros2 tests

预期 pytest 全部通过，Ruff 无 findings，compileall 返回 0。

- [ ] Step 2: 两次端到端复现

使用 seed=23 对两个新目录运行 --fast CLI。比较 policy.json、training_history.csv、evaluation.json、policy_comparison.csv 的 SHA-256，必须完全一致；严格解析 JSON，确认所有 PNG 非空。

- [ ] Step 3: 科学语义审查

确认当前窗口可观测性使用 rank(J)，故障传感器不贡献信息，所有策略共享 EpisodeSpec，奖励分量之和等于总奖励，报告措辞遵守 dominance 判据，ROS 2 不训练且不修改 Q 表。发现问题时先写失败回归测试再修复。

- [ ] Step 4: 最终代码审查

审查从 f7d87e5 到 HEAD 的完整 diff，关注数值稳定性、策略安全回退、随机性隔离、文件写入安全、科学结论和文档一致性。每个接受的 finding 都遵循 RED、最小修复、GREEN、独立提交。

- [ ] Step 5: 推送并确认 CI

    git status --short --branch
    git push origin main
    gh run list --repo chouyangceng/vehicle-state-estimation-lab --branch main --limit 3

等待精确最终 HEAD 的 workflow 结论为 success，不 force-push。

- [ ] Step 6: 记录交付证据

    报告最终 SHA、pytest 数量、Ruff/compile 状态、CI URL、生成物、项目绝对路径、复现命令和仿真限制。
