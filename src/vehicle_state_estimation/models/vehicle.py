from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .tire import fiala_lateral_force


@dataclass(frozen=True)
class VehicleParameters:
    mass: float = 1500.0
    yaw_inertia: float = 2500.0
    front_axle: float = 1.2
    rear_axle: float = 1.6
    cornering_front: float = 80000.0
    cornering_rear: float = 80000.0
    friction: float = 0.9
    gravity: float = 9.81


@dataclass(frozen=True)
class DynamicBicycleModel:
    parameters: VehicleParameters = VehicleParameters()

    @classmethod
    def default(cls) -> DynamicBicycleModel:
        return cls()

    def __post_init__(self) -> None:
        p = self.parameters
        if p.mass <= 0 or p.yaw_inertia <= 0 or p.front_axle <= 0 or p.rear_axle <= 0:
            raise ValueError("vehicle mass, inertia and axle distances must be positive")

    def derivative(self, state: np.ndarray, steering: float, acceleration: float) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        if state.shape != (4,) or not np.all(np.isfinite(state)):
            raise ValueError("state must be a finite vector with shape (4,)")
        vx, vy, yaw_rate, _yaw = state
        p = self.parameters
        vx_safe = max(abs(vx), 0.5)
        alpha_f = steering - math.atan2(vy + p.front_axle * yaw_rate, vx_safe)
        alpha_r = -math.atan2(vy - p.rear_axle * yaw_rate, vx_safe)
        fy_f = fiala_lateral_force(alpha_f, p.cornering_front, p.friction, p.mass * p.gravity * 0.55)
        fy_r = fiala_lateral_force(alpha_r, p.cornering_rear, p.friction, p.mass * p.gravity * 0.45)
        dvx = acceleration - (fy_f * math.sin(steering)) / p.mass + vy * yaw_rate
        dvy = (fy_f * math.cos(steering) + fy_r) / p.mass - vx * yaw_rate
        dr = (p.front_axle * fy_f * math.cos(steering) - p.rear_axle * fy_r) / p.yaw_inertia
        return np.array([dvx, dvy, dr, yaw_rate], dtype=float)
