from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MonteCarloSummary:
    trials: int
    rmse_mean: float
    rmse_std: float
    rmse_p95: float


def run_monte_carlo(trials: int = 20, steps: int = 100, seed: int = 7) -> MonteCarloSummary:
    if trials <= 0 or steps <= 1:
        raise ValueError("trials must be positive and steps must exceed one")
    rng = np.random.default_rng(seed)
    errors = []
    truth = np.linspace(0.0, 1.0, steps)
    for _ in range(trials):
        measurement = truth + rng.normal(0.0, 0.05, steps)
        errors.append(float(np.sqrt(np.mean((measurement - truth) ** 2))))
    values = np.asarray(errors)
    return MonteCarloSummary(trials, float(values.mean()), float(values.std()), float(np.percentile(values, 95)))
