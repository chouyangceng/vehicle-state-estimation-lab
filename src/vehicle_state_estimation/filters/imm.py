from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IMMEstimator:
    """Lightweight interacting multiple-model estimator for scalar hypotheses."""

    probabilities: np.ndarray

    def __init__(self, probabilities: list[float] | np.ndarray) -> None:
        values = np.asarray(probabilities, dtype=float)
        if values.ndim != 1 or values.size < 2 or np.any(values < 0) or values.sum() <= 0:
            raise ValueError("probabilities must be a non-negative vector with at least two models")
        self.probabilities = values / values.sum()

    def update(self, model_states: list[float] | np.ndarray, likelihoods: list[float] | np.ndarray | None = None) -> tuple[float, np.ndarray]:
        states = np.asarray(model_states, dtype=float)
        if states.shape != self.probabilities.shape or not np.all(np.isfinite(states)):
            raise ValueError("model states and probabilities must have equal finite shapes")
        weights = self.probabilities.copy() if likelihoods is None else self.probabilities * np.asarray(likelihoods, dtype=float)
        if weights.shape != states.shape or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("likelihoods must be non-negative and compatible")
        self.probabilities = weights / weights.sum()
        return float(self.probabilities @ states), self.probabilities.copy()
