from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validated_edges(values: tuple[float, ...], name: str) -> np.ndarray:
    edges = np.asarray(values, dtype=float)
    if edges.ndim != 1 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError(f"{name} boundaries must be finite and strictly increasing")
    return edges


@dataclass(frozen=True)
class StateDiscretizer:
    """Map transparent operating-condition features to one stable state index."""

    speed_edges: tuple[float, ...] = (2.0, 15.0)
    excitation_edges: tuple[float, ...] = (0.02, 0.12)
    uncertainty_edges: tuple[float, ...] = (0.25, 0.50, 0.80)
    sensor_count: int = 3
    action_count: int = 7

    def __post_init__(self) -> None:
        _validated_edges(self.speed_edges, "speed")
        _validated_edges(self.excitation_edges, "excitation")
        _validated_edges(self.uncertainty_edges, "uncertainty")
        if self.sensor_count <= 0 or self.action_count <= 0:
            raise ValueError("sensor_count and action_count must be positive")

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return (
            len(self.speed_edges) + 1,
            len(self.excitation_edges) + 1,
            len(self.uncertainty_edges) + 1,
            2**self.sensor_count,
            self.action_count + 1,
        )

    @property
    def state_count(self) -> int:
        return int(np.prod(self.shape))

    def encode(
        self,
        speed: float,
        excitation: float,
        uncertainty: float,
        health_mask: int,
        previous_action: int | None,
    ) -> int:
        values = np.asarray([speed, excitation, uncertainty], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("continuous state values must be finite")
        if not 0 <= health_mask < 2**self.sensor_count:
            raise ValueError("health_mask is outside the configured sensor mask")
        if previous_action is not None and not 0 <= previous_action < self.action_count:
            raise ValueError("previous_action is outside the action space")

        bins = (
            int(np.searchsorted(self.speed_edges, speed, side="right")),
            int(np.searchsorted(self.excitation_edges, excitation, side="right")),
            int(np.searchsorted(self.uncertainty_edges, uncertainty, side="right")),
            int(health_mask),
            0 if previous_action is None else previous_action + 1,
        )
        return int(np.ravel_multi_index(bins, self.shape))
