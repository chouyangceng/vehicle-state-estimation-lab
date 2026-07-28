from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AdaptiveFrictionEstimator:
    """A bounded recursive friction estimate from lateral force/load pairs."""

    initial: float = 0.8
    learning_rate: float = 0.15
    minimum: float = 0.1
    maximum: float = 1.2
    estimate: float | None = None

    def __post_init__(self) -> None:
        if not self.minimum > 0 or self.maximum <= self.minimum or not 0 < self.learning_rate <= 1:
            raise ValueError("friction bounds and learning rate are invalid")
        self.estimate = float(np.clip(self.initial, self.minimum, self.maximum))

    def update(self, lateral_force: float, normal_load: float) -> float:
        if normal_load <= 0 or not np.all(np.isfinite([lateral_force, normal_load])):
            raise ValueError("force and normal load must be finite and normal load must be positive")
        measured = abs(lateral_force) / normal_load
        self.estimate = float(np.clip(self.estimate + self.learning_rate * (measured - self.estimate), self.minimum, self.maximum))
        return self.estimate
