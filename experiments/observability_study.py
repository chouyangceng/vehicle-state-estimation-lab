"""Run the reproducible sensor-ablation observability study.

The command is intentionally dependency-light: the small YAML subset used by
``configs/observability.yaml`` is parsed locally, so PyYAML is not required.
It produces a strict JSON data product, a sortable CSV, a headless figure and
a short Chinese research note suitable for an undergraduate lab notebook.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Allow ``python experiments/observability_study.py`` from a source checkout.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vehicle_state_estimation.experiments.sensor_ablation import (
    SensorSuite,
    run_sensor_ablation,
)
from vehicle_state_estimation.simulation.maneuvers import (
    ManeuverConfig,
    generate_maneuver,
)

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 7,
    "finite_difference_step": 1e-5,
    "covariance_floor": 1e-9,
    "regularization": 1e-9,
    "low_speed_threshold": 0.5,
    "condition_limit": 1e10,
    "baseline": "imu",
    "maneuvers": ["straight", "sine_steer", "double_lane_change", "low_friction_switch"],
    "sensor_suites": [
        {"name": "imu", "sensors": ["imu"]},
        {"name": "wheel_speed", "sensors": ["wheel_speed"]},
        {"name": "gnss", "sensors": ["gnss"]},
        {"name": "all", "sensors": ["imu", "wheel_speed", "gnss"]},
    ],
}


def _scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value.startswith("[") and value.endswith("]"):
        return [_scalar(item) for item in value[1:-1].split(",") if item.strip()]
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        return float(value) if any(char in value for char in ".eE") else int(value)
    except ValueError:
        return value


def _read_config(path: str | Path | None) -> tuple[dict[str, Any], str]:
    """Read the project's small YAML subset without importing PyYAML."""

    config = dict(DEFAULT_CONFIG)
    if path is None:
        candidate = _ROOT / "configs" / "observability.yaml"
    else:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = _ROOT / candidate
    if not candidate.exists():
        return config, "built-in defaults"

    # ``utf-8-sig`` removes an optional UTF-8 BOM, which is common when a
    # configuration is edited in Windows Notepad.  Keep source line numbers
    # so malformed input can be fixed without guessing where it occurred.
    raw_lines = candidate.read_text(encoding="utf-8-sig").splitlines()
    lines = [(line.split("#", 1)[0].rstrip(), number) for number, line in enumerate(raw_lines, 1)]
    lines = [(line, number) for line, number in lines if line.strip()]
    allowed_keys = set(DEFAULT_CONFIG)
    index = 0
    while index < len(lines):
        line, line_number = lines[index]
        match = re.match(r"^([A-Za-z_][\w-]*):(?:\s*(.*))?$", line)
        if not match:
            raise ValueError(f"malformed config line {line_number}: {line.strip()!r}")
        key, raw = match.group(1), match.group(2) or ""
        if key not in allowed_keys:
            raise ValueError(f"unknown config key '{key}' on line {line_number}")
        if key == "sensor_suites" and not raw:
            suites: list[dict[str, Any]] = []
            index += 1
            while index < len(lines) and lines[index][0].startswith("  "):
                nested, nested_number = lines[index]
                name_match = re.match(r"^  -\s*name:\s*(.+)$", nested)
                if not name_match:
                    raise ValueError(f"malformed config line {nested_number}: {nested.strip()!r}")
                suite = {"name": str(_scalar(name_match.group(1)))}
                index += 1
                if index >= len(lines) or not re.match(r"^    sensors:\s*(.+)$", lines[index][0]):
                    missing_number = lines[index][1] if index < len(lines) else nested_number
                    raise ValueError(f"sensor suite at line {nested_number} requires sensors (near line {missing_number})")
                sensor_match = re.match(r"^    sensors:\s*(.+)$", lines[index][0])
                assert sensor_match is not None
                suite["sensors"] = _scalar(sensor_match.group(1))
                index += 1
                suites.append(suite)
            if not suites:
                raise ValueError("sensor_suites must contain at least one suite")
            config[key] = suites
            continue
        if key in {"maneuvers", "sensor_suites"} and not raw:
            values: list[Any] = []
            index += 1
            while index < len(lines) and lines[index][0].startswith("  "):
                item, item_number = lines[index]
                item = item.strip()
                if item.startswith("-"):
                    values.append(_scalar(item[1:].strip()))
                else:
                    raise ValueError(f"malformed config line {item_number}: {item!r}")
                index += 1
            if not values:
                raise ValueError(f"{key} must contain at least one item")
            config[key] = values
            continue
        if raw:
            config[key] = _scalar(raw)
        index += 1
    _validate_config(config)
    return config, str(candidate)


def _validate_config(config: dict[str, Any]) -> None:
    """Validate the subset consumed by the study before starting simulation."""

    maneuvers = config.get("maneuvers")
    if not isinstance(maneuvers, list) or not maneuvers:
        raise ValueError("maneuvers must contain at least one item")
    if any(not isinstance(item, str) or not item.strip() for item in maneuvers):
        raise ValueError("maneuvers must contain non-empty names")

    suites = config.get("sensor_suites")
    if not isinstance(suites, list) or not suites:
        raise ValueError("sensor_suites must contain at least one suite")
    if any(not isinstance(item, dict) for item in suites):
        raise ValueError("sensor_suites must contain mappings")


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def _write_ranking(results: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for item in results:
        crlb_values = list(item["crlb"])
        has_unbounded_crlb = any(value is None or not np.isfinite(float(value)) for value in crlb_values)
        bounds = [float(value) for value in crlb_values if value is not None and np.isfinite(float(value))]
        mean_crlb = float(np.mean(bounds)) if bounds else None
        # Higher rank/gain and lower uncertainty/conditioning are desirable.
        score = float("-inf") if has_unbounded_crlb else (
            10.0 * item["effective_rank"]
            + item["information_gain"]
            - np.log10(max(item["condition_number"], 1.0))
            - (mean_crlb if mean_crlb is not None else 1e6)
        )
        rows.append({**item, "mean_crlb": mean_crlb, "score": float(score)})
    rows.sort(key=lambda row: (-row["score"], row["maneuver"], row["suite"]))
    fields = [
        "rank",
        "maneuver",
        "suite",
        "sensors",
        "effective_rank",
        "condition_number",
        "information_gain",
        "mean_crlb",
        "ill_conditioned",
        "low_speed",
        "score",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            writer.writerow({field: rank if field == "rank" else row[field] for field in fields})


def _write_plot(results: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    maneuvers = list(dict.fromkeys(item["maneuver"] for item in results))
    suites = list(dict.fromkeys(item["suite"] for item in results))
    matrix = np.full((len(maneuvers), len(suites)), np.nan)
    for item in results:
        matrix[maneuvers.index(item["maneuver"]), suites.index(item["suite"])] = item[
            "effective_rank"
        ]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    image = axis.imshow(matrix, cmap="viridis", aspect="auto", vmin=0)
    axis.set_xticks(range(len(suites)), suites, rotation=25, ha="right")
    axis.set_yticks(range(len(maneuvers)), maneuvers)
    axis.set_title("Vehicle-state observability: effective rank")
    axis.set_xlabel("sensor suite")
    axis.set_ylabel("maneuver")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:.0f}", ha="center", va="center", color="white")
    figure.colorbar(image, ax=axis, label="effective rank")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_summary(results: list[dict[str, Any]], metadata: dict[str, Any], path: Path) -> None:
    best = max(results, key=lambda item: item["information_gain"])
    ill = sum(bool(item["ill_conditioned"]) for item in results)
    lines = [
        "# 可观测性与传感器组合实验报告",
        "",
        "## 结论",
        "",
        f"- 在本次仿真中，信息增益最高的组合是 **{best['maneuver']} + {best['suite']}**（信息增益 {best['information_gain']:.3f}）。",
        f"- 共评估 {len(results)} 个工况-传感器组合，其中 {ill} 个被标记为条件数过高、秩不足或低速不适定。",
        "- 直线低激励只能提供有限的状态方向；转向和换道激励更适合辨识横向速度、横摆角速度与航向耦合关系。",
        "",
        "## 可复现设置",
        "",
        f"- 随机种子：`{metadata['seed']}`；快速模式：`{metadata['fast']}`；配置：`{metadata['config_source']}`。",
        f"- 工况：{', '.join(metadata['maneuvers'])}；传感器组合：{', '.join(metadata['sensor_suites'])}。",
        "- 状态向量约定为 `[vx, vy, yaw_rate, yaw]`，灵敏度由中心差分近似，传感器噪声采用独立对角协方差。",
        "",
        "## 局限",
        "",
        "- 结果来自低阶自行车模型和理想观测，不代表真实车辆、轮胎温度或道路附着变化。",
        "- 经验 Gramian 与 CRLB 是局部线性化指标；它们用于比较实验设计，不等同于滤波器最终 RMSE。",
        "- 下一步应接入实车 CAN/IMU/GNSS 数据，并报告不同速度、噪声标定和 Monte Carlo 重复试验的置信区间。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_study(
    *,
    config_path: str | Path | None = None,
    output_dir: str | Path = "artifacts/observability-study",
    seed: int | None = None,
    fast: bool = False,
) -> Path:
    config, source = _read_config(config_path)
    seed_value = int(config.get("seed", 7) if seed is None else seed)
    if seed_value < 0:
        raise ValueError("seed must be non-negative")
    steps = 90 if fast else 360
    maneuvers = [str(item) for item in config.get("maneuvers", DEFAULT_CONFIG["maneuvers"])]
    suites = [SensorSuite(str(item["name"]), tuple(item["sensors"])) for item in config.get("sensor_suites", DEFAULT_CONFIG["sensor_suites"])]
    results: list[dict[str, Any]] = []
    for kind in maneuvers:
        trace = generate_maneuver(ManeuverConfig(kind=kind, steps=steps))
        ablations = run_sensor_ablation(
            trace,
            suites=suites,
            baseline=str(config.get("baseline", "imu")),
            finite_difference_step=float(config.get("finite_difference_step", 1e-5)),
            covariance_floor=float(config.get("covariance_floor", 1e-9)),
            regularization=float(config.get("regularization", 1e-9)),
            low_speed_threshold=float(config.get("low_speed_threshold", 0.5)),
            condition_limit=float(config.get("condition_limit", 1e10)),
        )
        results.extend(item.to_dict() for item in ablations)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "seed": seed_value,
        "seed_semantics": "metadata_only_deterministic_replay",
        "fast": bool(fast),
        "config_source": source,
        "maneuvers": maneuvers,
        "sensor_suites": [suite.name for suite in suites],
        "steps": steps,
    }
    (output / "results.json").write_text(
        json.dumps(_strict({"metadata": metadata, "results": results}), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    _write_ranking(results, output / "ranking.csv")
    _write_plot(results, output / "observability.png")
    _write_summary(results, metadata, output / "summary.md")
    return output


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Run the vehicle observability sensor-ablation study")
    parser.add_argument("--config", default=None, help="YAML config path (optional)")
    parser.add_argument("--output", default="artifacts/observability-study")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="deterministic seed recorded as experiment metadata (reserved for future stochastic noise)",
    )
    parser.add_argument("--fast", action="store_true", help="use shorter traces for smoke tests")
    args = parser.parse_args(argv)
    output = run_study(config_path=args.config, output_dir=args.output, seed=args.seed, fast=args.fast)
    print(f"observability study written to {output}")
    return output


if __name__ == "__main__":
    main()
