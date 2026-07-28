from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..metrics.consistency import rmse


def run(seed: int = 7, steps: int = 200, output_dir: str | Path = "artifacts/state-estimation") -> Path:
    if steps < 2:
        raise ValueError("steps must be at least 2")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.arange(steps, dtype=float) * 0.02
    truth_vx = 10.0 + 0.6 * np.sin(0.7 * t)
    truth_vy = 0.35 * np.sin(1.3 * t)
    truth_r = np.gradient(truth_vy, 0.02) / np.maximum(truth_vx, 0.5)
    measurement_vx = truth_vx + rng.normal(0.0, 0.12, steps)
    measurement_vy = truth_vy + rng.normal(0.0, 0.06, steps)
    measurement_r = truth_r + rng.normal(0.0, 0.025, steps)
    estimate_vx = np.empty(steps)
    estimate_vy = np.empty(steps)
    estimate_r = np.empty(steps)
    estimate_vx[0], estimate_vy[0], estimate_r[0] = measurement_vx[0], measurement_vy[0], measurement_r[0]
    for i in range(1, steps):
        gain = 0.18 if i < steps // 2 else 0.10
        estimate_vx[i] = estimate_vx[i - 1] + gain * (measurement_vx[i] - estimate_vx[i - 1])
        estimate_vy[i] = estimate_vy[i - 1] + gain * (measurement_vy[i] - estimate_vy[i - 1])
        estimate_r[i] = estimate_r[i - 1] + gain * (measurement_r[i] - estimate_r[i - 1])
    metrics = {
        "seed": seed,
        "steps": steps,
        "vx_rmse": rmse(estimate_vx, truth_vx),
        "vy_rmse": rmse(estimate_vy, truth_vy),
        "yaw_rate_rmse": rmse(estimate_r, truth_r),
    }
    np.savetxt(
        output / "state_trace.csv",
        np.column_stack([t, truth_vx, truth_vy, truth_r, estimate_vx, estimate_vy, estimate_r]),
        delimiter=",",
        header="time,truth_vx,truth_vy,truth_yaw_rate,estimate_vx,estimate_vy,estimate_yaw_rate",
        comments="",
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        for axis, truth, estimate, label in zip(
            axes,
            [truth_vx, truth_vy, truth_r],
            [estimate_vx, estimate_vy, estimate_r],
            ["vx (m/s)", "vy (m/s)", "yaw rate (rad/s)"],
            strict=True,
        ):
            axis.plot(t, truth, label="truth")
            axis.plot(t, estimate, label="estimate", alpha=0.8)
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.3)
        axes[0].legend()
        axes[-1].set_xlabel("time (s)")
        figure.tight_layout()
        figure.savefig(output / "state_estimation.png", dpi=140)
        plt.close(figure)
    except ImportError:
        pass
    return output
