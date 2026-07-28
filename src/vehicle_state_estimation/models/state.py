from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VehicleState:
    """Ego state in SI units: m, s, rad and rad/s."""

    vx: float
    vy: float
    yaw_rate: float
    yaw: float
    x: float
    y: float
    friction: float
    timestamp: float

    def __post_init__(self) -> None:
        values = np.asarray(self.to_array(), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("vehicle state must contain only finite values")
        if self.vx < 0:
            raise ValueError("longitudinal speed cannot be negative")
        if self.friction <= 0:
            raise ValueError("friction coefficient must be positive")

    def to_array(self) -> np.ndarray:
        return np.array(
            [self.vx, self.vy, self.yaw_rate, self.yaw, self.x, self.y, self.friction, self.timestamp],
            dtype=float,
        )

    @classmethod
    def from_array(cls, values: np.ndarray) -> VehicleState:
        values = np.asarray(values, dtype=float)
        if values.shape != (8,):
            raise ValueError(f"expected state shape (8,), got {values.shape}")
        return cls(*values.tolist())
