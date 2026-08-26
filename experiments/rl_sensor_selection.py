"""Train and compare interpretable sensor-selection policies."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vehicle_state_estimation.rl import (
    ACTION_MASKS,
    SENSOR_NAMES,
    AlwaysAllPolicy,
    GreedyInformationPolicy,
    LowestCostPolicy,
    RandomPolicy,
    evaluate_policy,
    train_q_learning,
)
from vehicle_state_estimation.rl.scenarios import MANEUVERS, make_research_environment


def run_study(
    *,
    output_dir: str | Path = "artifacts/rl-sensor-selection",
    seed: int = 7,
    fast: bool = False,
    overwrite: bool = False,
) -> Path:
    """Run training, controlled baselines, plots and a bounded conclusion."""

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    train_episodes = 48 if fast else 320
    evaluation_episodes = 12 if fast else 48
    windows = 12 if fast else 28

    def factory(episode: int):
        return make_research_environment(episode, seed=seed, windows=windows)

    agent, history = train_q_learning(factory, episodes=train_episodes, seed=seed)

    def learned_policy(state: int, environment: Any) -> int:
        del environment
        return agent.select_action(state)

    policies = {
        "q_learning": learned_policy,
        "always_all": AlwaysAllPolicy(),
        "lowest_cost": LowestCostPolicy(),
        "random": RandomPolicy(seed=seed + 1),
        "greedy_information": GreedyInformationPolicy(),
    }
    evaluations = {
        name: evaluate_policy(factory, policy, episodes=evaluation_episodes)
        for name, policy in policies.items()
    }
    paired_comparisons = _paired_comparisons(evaluations)
    metadata = {
        "seed": seed,
        "fast": fast,
        "train_episodes": train_episodes,
        "evaluation_episodes": evaluation_episodes,
        "windows_per_episode": windows,
        "maneuvers": list(MANEUVERS),
        "sensor_names": list(SENSOR_NAMES),
        "action_masks": list(ACTION_MASKS),
        "state_bins": {
            "speed": [2.0, 15.0],
            "excitation": [0.02, 0.12],
            "uncertainty": [0.25, 0.5, 0.8],
        },
        "research_boundary": "deterministic low-order simulation; not a real-vehicle claim",
    }
    policy_payload = agent.to_policy_dict(metadata=metadata)
    _write_json(output / "policy.json", policy_payload)
    _write_json(
        output / "evaluation.json",
        {"metadata": metadata, "policies": evaluations, "paired_comparisons": paired_comparisons},
    )
    _write_history(output / "training_history.csv", history)
    _write_comparison(output / "policy_comparison.csv", evaluations)
    _write_paired_comparison(output / "paired_comparison.csv", paired_comparisons)
    _write_training_plot(output / "training_curve.png", history)
    _write_tradeoff_plot(output / "tradeoff.png", evaluations)
    _write_action_plot(output / "action_usage.png", evaluations)
    _write_summary(output / "summary.md", metadata, evaluations, paired_comparisons)
    return output


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _write_comparison(path: Path, evaluations: dict[str, dict[str, Any]]) -> None:
    fields = [
        "policy",
        "mean_return",
        "mean_uncertainty",
        "p95_uncertainty",
        "mean_sensor_cost",
        "unobservable_rate",
        "fallback_rate",
        "switching_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, result in evaluations.items():
            steps = result["steps"]
            writer.writerow(
                {
                    "policy": name,
                    "mean_return": result["mean_return"],
                    "mean_uncertainty": result["mean_uncertainty"],
                    "p95_uncertainty": result["p95_uncertainty"],
                    "mean_sensor_cost": result["mean_sensor_cost"],
                    "unobservable_rate": result["unobservable_steps"] / steps,
                    "fallback_rate": result["fallback_steps"] / steps,
                    "switching_rate": result["switching_steps"] / steps,
                }
            )


def _mean_ci(values: np.ndarray) -> dict[str, float]:
    """Normal-approximation interval for paired episode differences."""
    mean = float(np.mean(values))
    half_width = (
        1.96 * float(np.std(values, ddof=1)) / np.sqrt(values.size)
        if values.size > 1
        else 0.0
    )
    return {"mean": mean, "ci95_low": mean - half_width, "ci95_high": mean + half_width}


def _paired_comparisons(evaluations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    learned = evaluations["q_learning"]["episode_metrics"]
    result = {}
    for name, evaluation in evaluations.items():
        if name == "q_learning":
            continue
        baseline = evaluation["episode_metrics"]
        if len(learned) != len(baseline):
            raise ValueError("paired policy evaluation requires matching episodes")
        result[name] = {}
        for metric in ("return", "mean_uncertainty", "mean_sensor_cost"):
            differences = np.asarray(
                [left[metric] - right[metric] for left, right in zip(learned, baseline, strict=True)],
                dtype=float,
            )
            result[name][metric] = _mean_ci(differences)
    return result


def _write_paired_comparison(path: Path, comparisons: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("baseline", "metric", "mean_delta", "ci95_low", "ci95_high"))
        for baseline, metrics in comparisons.items():
            for metric, interval in metrics.items():
                writer.writerow(
                    (baseline, metric, interval["mean"], interval["ci95_low"], interval["ci95_high"])
                )


def _write_training_plot(path: Path, history: list[dict[str, float | int]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes = np.asarray([item["episode"] for item in history])
    returns = np.asarray([item["return"] for item in history], dtype=float)
    window = max(2, min(20, len(history) // 5))
    moving = np.convolve(returns, np.ones(window) / window, mode="valid")
    figure, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    axes[0].plot(episodes, returns, color="#8aa4b8", alpha=0.55, label="episode")
    axes[0].plot(episodes[window - 1 :], moving, color="#ef8354", linewidth=2, label="moving mean")
    axes[0].set_ylabel("return")
    axes[0].legend(frameon=False)
    axes[1].plot(
        episodes,
        [item["mean_uncertainty"] for item in history],
        color="#2a9d8f",
    )
    axes[1].set_ylabel("uncertainty")
    axes[2].plot(
        episodes,
        [item["mean_sensor_cost"] for item in history],
        color="#e9c46a",
    )
    axes[2].set_xlabel("training episode")
    axes[2].set_ylabel("sensor cost")
    figure.suptitle("Q-learning sensor selection — reproducible training")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_tradeoff_plot(path: Path, evaluations: dict[str, dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    offsets = {
        "q_learning": (-55, -18),
        "always_all": (-95, -30),
        "lowest_cost": (8, 5),
        "random": (8, 8),
        "greedy_information": (-38, 28),
    }
    markers = ("o", "s", "^", "D", "P")
    for marker, (name, result) in zip(markers, evaluations.items(), strict=True):
        episode_metrics = result["episode_metrics"]
        cost_values = np.asarray([item["mean_sensor_cost"] for item in episode_metrics])
        uncertainty_values = np.asarray([item["mean_uncertainty"] for item in episode_metrics])
        cost_error = 1.96 * np.std(cost_values, ddof=1) / np.sqrt(cost_values.size)
        uncertainty_error = 1.96 * np.std(uncertainty_values, ddof=1) / np.sqrt(
            uncertainty_values.size
        )
        axis.errorbar(
            result["mean_sensor_cost"],
            result["mean_uncertainty"],
            xerr=cost_error,
            yerr=uncertainty_error,
            marker=marker,
            markersize=9,
            capsize=3,
            linestyle="none",
            label=name,
        )
        axis.annotate(
            name,
            (result["mean_sensor_cost"], result["mean_uncertainty"]),
            xytext=offsets[name],
            textcoords="offset points",
        )
    axis.set_xlabel("mean normalized sensor cost (lower is better)")
    axis.set_ylabel("mean normalized covariance trace (lower is better)")
    axis.set_title("Cost–uncertainty policy trade-off")
    axis.grid(alpha=0.25)
    axis.margins(x=0.08, y=0.12)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_action_plot(path: Path, evaluations: dict[str, dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(evaluations)
    counts = np.asarray(
        [[evaluations[name]["action_counts"][str(action)] for action in range(7)] for name in names],
        dtype=float,
    )
    totals = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    shares = counts / totals
    figure, axis = plt.subplots(figsize=(10, 5.5))
    bottom = np.zeros(len(names))
    labels = ["IMU", "Wheel", "GNSS", "I+W", "I+G", "W+G", "All"]
    for action, label in enumerate(labels):
        axis.bar(names, shares[:, action], bottom=bottom, label=label)
        bottom += shares[:, action]
    axis.set_ylabel("effective action share")
    axis.set_title("Sensor-combination usage by policy")
    axis.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    axis.tick_params(axis="x", rotation=16)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_summary(
    path: Path,
    metadata: dict[str, Any],
    evaluations: dict[str, dict[str, Any]],
    paired_comparisons: dict[str, Any],
) -> None:
    learned = evaluations["q_learning"]
    baseline = evaluations["always_all"]
    learned_unobservable = learned["unobservable_steps"] / learned["steps"]
    baseline_unobservable = baseline["unobservable_steps"] / baseline["steps"]
    qualifies = (
        learned["mean_sensor_cost"] < baseline["mean_sensor_cost"]
        and learned["mean_uncertainty"] <= 1.10 * baseline["mean_uncertainty"]
        and learned_unobservable <= baseline_unobservable + 0.05
    )
    conclusion = (
        "在预先声明的 10% 不确定度和 5 个百分点不可观测率容差内，学习策略降低了平均传感器成本。"
        if qualifies
        else "学习策略展示了不同的成本—不确定度权衡，但未同时满足预先声明的优越性条件。"
    )
    rows = [
        f"| {name} | {result['mean_return']:.3f} | {result['mean_sensor_cost']:.3f} | {result['mean_uncertainty']:.4f} | {result['unobservable_steps'] / result['steps']:.1%} |"
        for name, result in evaluations.items()
    ]
    paired_rows = [
        (
            f"| {name} | {metrics['mean_sensor_cost']['mean']:.3f} "
            f"[{metrics['mean_sensor_cost']['ci95_low']:.3f}, "
            f"{metrics['mean_sensor_cost']['ci95_high']:.3f}] | "
            f"{metrics['mean_uncertainty']['mean']:.4f} "
            f"[{metrics['mean_uncertainty']['ci95_low']:.4f}, "
            f"{metrics['mean_uncertainty']['ci95_high']:.4f}] |"
        )
        for name, metrics in paired_comparisons.items()
    ]
    text = [
        "# RL 自适应传感器选择实验",
        "",
        "## 结论",
        "",
        conclusion,
        "",
        "| 策略 | 平均回报 | 平均成本 | 平均不确定度 | 不可观测率 |",
        "|---|---:|---:|---:|---:|",
        *rows,
        "",
        "## 配对 episode 差异（Q-learning − 基线）",
        "",
        "| 基线 | 平均成本差 [95% 区间] | 平均不确定度差 [95% 区间] |",
        "|---|---:|---:|",
        *paired_rows,
        "",
        "负成本差表示 Q-learning 更省，负不确定度差表示 Q-learning 的协方差代理更低。",
        "区间是 episode 间标准误的正态近似，不是实车总体置信保证。",
        "",
        "## 可复现设置",
        "",
        f"- 随机种子：`{metadata['seed']}`；训练 {metadata['train_episodes']} episodes；评估 {metadata['evaluation_episodes']} episodes。",
        f"- 每个 episode 为 {metadata['windows_per_episode']} 个决策窗口，循环覆盖 {', '.join(metadata['maneuvers'])}。",
        "- 对照策略：全传感器、最低成本、带种子随机、单步信息贪心；全部使用相同工况与故障调度。",
        "",
        "## 指标边界",
        "",
        "- 不确定度是低阶模型的有限协方差代理；可观测性另由纯传感器信息矩阵的秩判断。",
        "- 本结果不代表真实车辆能耗、定位安全或量产功能安全认证，不能替代道路和硬件在环试验。",
        "- 下一步应在真实 CAN、IMU、轮速与 GNSS 时间同步数据上重新标定信息矩阵和成本模型。",
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/rl-sensor-selection")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)
    return run_study(
        output_dir=arguments.output,
        seed=arguments.seed,
        fast=arguments.fast,
        overwrite=arguments.overwrite,
    )


if __name__ == "__main__":
    main()
