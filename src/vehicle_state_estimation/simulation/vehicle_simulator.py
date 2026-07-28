from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimulatorConfig:
    steps: int = 200
    dt: float = 0.02
    seed: int = 7
    imu_noise: float = 0.02
    wheel_noise: float = 0.04

    def __post_init__(self) -> None:
        if self.steps < 2 or self.dt <= 0 or self.imu_noise < 0 or self.wheel_noise < 0:
            raise ValueError("simulation configuration is invalid")


@dataclass(frozen=True)
class SensorLog:
    time: np.ndarray
    truth: np.ndarray
    imu: np.ndarray
    wheel_speed: np.ndarray
    gnss: np.ndarray


class VehicleSimulator:
    def __init__(self, config: SimulatorConfig | None = None) -> None:
        self.config = config or SimulatorConfig()

    def run(self) -> SensorLog:
        c = self.config
        rng = np.random.default_rng(c.seed)
        time = np.arange(c.steps, dtype=float) * c.dt
        vx = 12.0 + 1.2 * np.sin(0.45 * time)
        vy = 0.45 * np.sin(0.9 * time)
        yaw_rate = np.gradient(vy, c.dt) / np.maximum(vx, 0.5)
        yaw = np.cumsum(yaw_rate) * c.dt
        truth = np.column_stack([vx, vy, yaw_rate, yaw])
        acceleration = np.column_stack([np.gradient(vx, c.dt), np.gradient(vy, c.dt), yaw_rate])
        imu = acceleration + rng.normal(0.0, c.imu_noise, acceleration.shape)
        wheel_speed = vx[:, None] / 0.32 + rng.normal(0.0, c.wheel_noise, (c.steps, 4))
        gnss = np.column_stack([np.cumsum(vx * np.cos(yaw)) * c.dt, np.cumsum(vx * np.sin(yaw)) * c.dt])
        gnss += rng.normal(0.0, 0.15, gnss.shape)
        return SensorLog(time, truth, imu, wheel_speed, gnss)
