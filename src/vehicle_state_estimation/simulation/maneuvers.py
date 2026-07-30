"""Deterministic vehicle manoeuvres and ideal sensor observations.

The generator intentionally uses a bounded, low-order bicycle surrogate.  It is
not intended to replace a high-fidelity vehicle model; it provides repeatable
excitation signals for observability experiments and unit tests without a
simulator dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

ManeuverKind = Literal["straight", "sine_steer", "double_lane_change", "low_friction_switch"]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ManeuverConfig:
    """Parameters shared by all generated manoeuvres (SI units)."""

    kind: str = "sine_steer"
    steps: int = 500
    dt: float = 0.02
    speed: float = 12.0
    acceleration: float = 0.0
    steering_amplitude: float = 0.08
    steering_frequency: float = 0.35
    friction: float = 0.9
    low_friction: float = 0.35
    friction_switch_time: float = 4.0
    wheelbase: float = 2.8
    wheel_radius: float = 0.32
    imu_noise_std: float = 0.02
    wheel_noise_std: float = 0.04
    gnss_noise_std: float = 0.15

    def __post_init__(self) -> None:
        allowed = {"straight", "sine_steer", "double_lane_change", "low_friction_switch"}
        if self.kind not in allowed:
            raise ValueError(f"kind must be one of {sorted(allowed)}")
        if not isinstance(self.steps, int) or isinstance(self.steps, bool) or self.steps < 2:
            raise ValueError("steps must be an integer of at least 2")
        if not np.isfinite(self.dt) or self.dt <= 0 or self.dt > 0.25:
            raise ValueError("dt must be finite, positive, and no greater than 0.25 s")
        if not np.isfinite(self.acceleration) or abs(self.acceleration) > 50.0:
            raise ValueError("acceleration must be finite and bounded by 50 m/s^2")
        for name in ("speed", "steering_amplitude", "friction", "low_friction", "wheelbase", "wheel_radius"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.low_friction > self.friction:
            raise ValueError("low_friction cannot exceed friction")
        if not np.isfinite(self.steering_frequency) or self.steering_frequency <= 0:
            raise ValueError("steering_frequency must be finite and positive")
        if not np.isfinite(self.friction_switch_time) or self.friction_switch_time < 0:
            raise ValueError("friction_switch_time must be finite and non-negative")
        for name in ("imu_noise_std", "wheel_noise_std", "gnss_noise_std"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ManeuverTrace:
    """Time history returned by :func:`generate_maneuver`.

    ``state`` follows the package convention ``[vx, vy, yaw_rate, yaw]``;
    ``position`` is the world-frame ``[x, y]`` trajectory.
    """

    config: ManeuverConfig
    time: FloatArray
    state: FloatArray
    position: FloatArray
    steering: FloatArray
    acceleration: FloatArray
    friction: FloatArray
    excitation_energy: FloatArray = field(repr=False)

    def observe_imu(self) -> tuple[FloatArray, FloatArray]:
        return imu_observation(self)

    def observe_wheel_speed(self) -> tuple[FloatArray, FloatArray]:
        return wheel_speed_observation(self)

    def observe_gnss(self) -> tuple[FloatArray, FloatArray]:
        return gnss_observation(self)


def _steering_signal(config: ManeuverConfig, time: FloatArray) -> FloatArray:
    amplitude = config.steering_amplitude
    if config.kind == "straight":
        return np.zeros_like(time)
    phase = 2.0 * np.pi * config.steering_frequency * time
    if config.kind == "sine_steer":
        return amplitude * np.sin(phase)
    if config.kind == "double_lane_change":
        # Two smooth, opposite lane changes separated by a half period.
        envelope = np.sin(np.pi * np.clip(time / max(time[-1], config.dt), 0.0, 1.0)) ** 2
        return amplitude * envelope * np.sin(phase)
    # A repeatable steering manoeuvre that crosses a friction discontinuity.
    return 0.75 * amplitude * np.sin(phase)


def generate_maneuver(config: ManeuverConfig | None = None) -> ManeuverTrace:
    """Generate a bounded deterministic manoeuvre and its ideal trajectory."""

    config = config or ManeuverConfig()
    time = np.arange(config.steps, dtype=float) * config.dt
    steering = _steering_signal(config, time)
    acceleration = np.full(config.steps, config.acceleration, dtype=float)
    friction = np.where(
        (config.kind == "low_friction_switch") & (time >= config.friction_switch_time),
        config.low_friction,
        config.friction,
    )

    state = np.empty((config.steps, 4), dtype=float)
    position = np.empty((config.steps, 2), dtype=float)
    state[0] = [config.speed, 0.0, 0.0, 0.0]
    position[0] = [0.0, 0.0]
    # First-order yaw/lateral dynamics remain numerically well behaved for
    # unusually long traces and make friction changes visible in the response.
    lateral_time_constant = 0.35
    yaw_time_constant = 0.20
    for index in range(1, config.steps):
        vx, vy, yaw_rate, yaw = state[index - 1]
        dt = config.dt
        vx = float(np.clip(vx + acceleration[index - 1] * dt, 0.1, 50.0))
        desired_rate = friction[index - 1] * vx * np.tan(steering[index - 1]) / config.wheelbase
        yaw_rate += dt * (desired_rate - yaw_rate) / yaw_time_constant
        vy += dt * (0.18 * vx * steering[index - 1] - vy) / lateral_time_constant
        yaw += dt * yaw_rate
        state[index] = [vx, vy, yaw_rate, yaw]
        position[index] = position[index - 1] + dt * np.array([
            vx * np.cos(yaw) - vy * np.sin(yaw),
            vx * np.sin(yaw) + vy * np.cos(yaw),
        ])
    excitation_energy = np.cumsum(steering**2) * config.dt
    return ManeuverTrace(config, time, state, position, steering, acceleration, friction, excitation_energy)


def _check_trace(trace: ManeuverTrace) -> None:
    if not isinstance(trace, ManeuverTrace):
        raise TypeError("trace must be a ManeuverTrace")
    expected = {
        "time": (trace.time, 1),
        "state": (trace.state, 2),
        "position": (trace.position, 2),
        "steering": (trace.steering, 1),
        "acceleration": (trace.acceleration, 1),
        "friction": (trace.friction, 1),
        "excitation_energy": (trace.excitation_energy, 1),
    }
    try:
        sample_count = trace.state.shape[0]
    except AttributeError as error:
        raise ValueError("trace state must have shape (N, 4)") from error
    for name, (value, ndim) in expected.items():
        array = np.asarray(value)
        if array.ndim != ndim:
            raise ValueError(f"trace {name} must have {ndim} dimensions")
        if name == "state" and array.shape[1] != 4:
            raise ValueError("trace state must have shape (N, 4)")
        if name == "position" and array.shape[1] != 2:
            raise ValueError("trace position must have shape (N, 2)")
        if array.shape[0] != sample_count:
            raise ValueError(f"trace {name} length must match state")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"trace {name} must contain only finite values")


def imu_observation(trace: ManeuverTrace) -> tuple[FloatArray, FloatArray]:
    """Return ideal body IMU ``[a_x, a_y, yaw_rate]`` and covariance."""

    _check_trace(trace)
    lateral_acceleration = (
        0.18 * trace.state[:, 0] * trace.steering - trace.state[:, 1]
    ) / 0.35 + trace.state[:, 0] * trace.state[:, 2]
    values = np.column_stack((trace.acceleration, lateral_acceleration, trace.state[:, 2]))
    covariance = np.eye(3) * trace.config.imu_noise_std**2
    return values, covariance


def wheel_speed_observation(trace: ManeuverTrace) -> tuple[FloatArray, FloatArray]:
    """Return ideal four-wheel angular speeds and covariance."""

    _check_trace(trace)
    values = np.repeat((trace.state[:, 0] / trace.config.wheel_radius)[:, None], 4, axis=1)
    covariance = np.eye(4) * trace.config.wheel_noise_std**2
    return values, covariance


def gnss_observation(trace: ManeuverTrace) -> tuple[FloatArray, FloatArray]:
    """Return ideal world-frame ``[x, y]`` fixes and covariance."""

    _check_trace(trace)
    covariance = np.eye(2) * trace.config.gnss_noise_std**2
    return trace.position.copy(), covariance


__all__ = [
    "ManeuverConfig",
    "ManeuverKind",
    "ManeuverTrace",
    "generate_maneuver",
    "gnss_observation",
    "imu_observation",
    "wheel_speed_observation",
]
