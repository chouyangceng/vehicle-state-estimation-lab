from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorResult:
    value: np.ndarray
    is_valid: bool
    timestamp: float = 0.0


@dataclass
class SensorFault:
    drop_probability: float = 0.0
    bias: float = 0.0
    delay_steps: int = 0
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.drop_probability <= 1.0 or self.delay_steps < 0:
            raise ValueError("drop probability must be in [0, 1] and delay must be non-negative")
        self._rng = np.random.default_rng(self.seed)

    def apply(self, value: np.ndarray, timestamp: float = 0.0) -> SensorResult:
        array = np.asarray(value, dtype=float)
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError("sensor value must be a non-empty finite array")
        if self._rng.random() < self.drop_probability:
            return SensorResult(np.full_like(array, np.nan), False, timestamp)
        return SensorResult(array + self.bias, True, timestamp + self.delay_steps)
