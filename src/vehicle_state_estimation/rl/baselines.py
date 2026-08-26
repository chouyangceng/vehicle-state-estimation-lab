from __future__ import annotations

import numpy as np

from .environment import SensorSelectionEnv


class AlwaysAllPolicy:
    """Request all sensors and let the environment apply its fault fallback."""

    def __call__(self, state: int, environment: SensorSelectionEnv | None = None) -> int:
        del state
        del environment
        return 6


class LowestCostPolicy:
    """Request wheel speed, the default lowest-cost individual sensor."""

    def __call__(self, state: int, environment: SensorSelectionEnv | None = None) -> int:
        del state
        del environment
        return 1


class RandomPolicy:
    """Seeded uniform action baseline for reproducible comparisons."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def __call__(self, state: int, environment: SensorSelectionEnv | None = None) -> int:
        del state
        del environment
        return int(self._rng.integers(7))


class GreedyInformationPolicy:
    """Maximize current-window information with a small sensor-cost penalty."""

    def __init__(self, cost_weight: float = 0.05) -> None:
        self.cost_weight = float(cost_weight)

    def __call__(self, state: int, environment: SensorSelectionEnv | None = None) -> int:
        del state
        if environment is None:
            raise ValueError("GreedyInformationPolicy requires the current environment")
        return environment.greedy_information_action(self.cost_weight)
