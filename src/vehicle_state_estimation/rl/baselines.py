from __future__ import annotations

import numpy as np


class AlwaysAllPolicy:
    """Request all sensors and let the environment apply its fault fallback."""

    def __call__(self, state: int) -> int:
        del state
        return 6


class LowestCostPolicy:
    """Request wheel speed, the default lowest-cost individual sensor."""

    def __call__(self, state: int) -> int:
        del state
        return 1


class RandomPolicy:
    """Seeded uniform action baseline for reproducible comparisons."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def __call__(self, state: int) -> int:
        del state
        return int(self._rng.integers(7))
