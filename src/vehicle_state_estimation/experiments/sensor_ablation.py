"""Deterministic sensor-suite ablation for vehicle-state observability studies.

The runner deliberately keeps the measurement models small and transparent.  It
is intended for comparing excitation/sensor choices, not as a replacement for a
production estimator or a high-fidelity vehicle simulator.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..metrics.observability import (
    cramer_rao_lower_bound,
    effective_rank,
    empirical_observability_gramian,
    finite_difference_jacobian,
    information_gain,
    observability_report,
    regularized_condition_number,
)
from ..simulation.maneuvers import ManeuverTrace

FloatArray = NDArray[np.float64]
SENSOR_ORDER = ("imu", "wheel_speed", "gnss")
_DEFAULT_SUITES = {
    "imu": ("imu",),
    "wheel_speed": ("wheel_speed",),
    "gnss": ("gnss",),
    "all": SENSOR_ORDER,
}


def _as_sensor_tuple(value: Sequence[str]) -> tuple[str, ...]:
    sensors = tuple(str(item) for item in value)
    if not sensors:
        raise ValueError("sensors must contain at least one sensor")
    if len(set(sensors)) != len(sensors):
        raise ValueError("sensors must not contain duplicates")
    unknown = sorted(set(sensors) - set(SENSOR_ORDER))
    if unknown:
        raise ValueError(f"unknown sensor: {unknown[0]}")
    return tuple(sensor for sensor in SENSOR_ORDER if sensor in sensors)


@dataclass(frozen=True)
class SensorSuite:
    """Named, validated combination of supported observations."""

    name: str
    sensors: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("suite name must not be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "sensors", _as_sensor_tuple(self.sensors))


@dataclass(frozen=True)
class AblationResult:
    """One sensor-suite result with machine-readable metrics."""

    maneuver: str
    suite: str
    sensors: tuple[str, ...]
    gramian: FloatArray
    effective_rank: int
    condition_number: float
    information_gain: float
    crlb: FloatArray
    ill_conditioned: bool
    low_speed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation (including CRLB bounds)."""

        return {
            "maneuver": self.maneuver,
            "suite": self.suite,
            "sensors": list(self.sensors),
            "gramian": self.gramian.tolist(),
            "effective_rank": int(self.effective_rank),
            "condition_number": float(self.condition_number),
            "information_gain": float(self.information_gain),
            "crlb": self.crlb.tolist(),
            "ill_conditioned": bool(self.ill_conditioned),
            "low_speed": bool(self.low_speed),
        }

    def to_json(self, path: str | Path) -> Path:
        """Write this result as an indented UTF-8 JSON document."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=True), encoding="utf-8")
        return target

    @staticmethod
    def write_json(results: Sequence["AblationResult"], path: str | Path) -> Path:
        """Write a list of results as one JSON array."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = [result.to_dict() for result in results]
        target.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
        return target

    @staticmethod
    def write_csv(results: Sequence["AblationResult"], path: str | Path) -> Path:
        """Write compact scalar metrics for ranking and plotting tools."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        columns = (
            "maneuver",
            "suite",
            "sensors",
            "effective_rank",
            "condition_number",
            "information_gain",
            "crlb",
            "ill_conditioned",
            "low_speed",
        )
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for result in results:
                writer.writerow(
                    {
                        "maneuver": result.maneuver,
                        "suite": result.suite,
                        "sensors": "+".join(result.sensors),
                        "effective_rank": result.effective_rank,
                        "condition_number": result.condition_number,
                        "information_gain": result.information_gain,
                        "crlb": ";".join(str(value) for value in result.crlb),
                        "ill_conditioned": result.ill_conditioned,
                        "low_speed": result.low_speed,
                    }
                )
        return target


def _coerce_suite(value: SensorSuite | str | Mapping[str, Any]) -> SensorSuite:
    if isinstance(value, SensorSuite):
        return value
    if isinstance(value, str):
        if value not in _DEFAULT_SUITES:
            raise ValueError(f"unknown sensor suite: {value}")
        return SensorSuite(value, _DEFAULT_SUITES[value])
    if isinstance(value, Mapping):
        try:
            return SensorSuite(str(value["name"]), tuple(value["sensors"]))
        except KeyError as error:
            raise ValueError("suite mapping requires name and sensors") from error
    raise TypeError("suites must contain SensorSuite, names, or mappings")


def _sort_suites(suites: Sequence[SensorSuite]) -> list[SensorSuite]:
    order = {sensor: index for index, sensor in enumerate(SENSOR_ORDER)}
    return sorted(suites, key=lambda suite: (tuple(order[item] for item in suite.sensors), suite.name))


def _measurement_model(sensor: str, wheel_radius: float):
    if sensor == "imu":
        return lambda state: np.array(
            [state[0], state[1] + state[0] * state[2], state[2]], dtype=float
        )
    if sensor == "wheel_speed":
        return lambda state: np.repeat(state[0] / wheel_radius, 4).astype(float)
    if sensor == "gnss":
        return lambda state: np.array(
            [
                state[0] * np.cos(state[3]) - state[1] * np.sin(state[3]),
                state[0] * np.sin(state[3]) + state[1] * np.cos(state[3]),
            ],
            dtype=float,
        )
    raise ValueError(f"unknown sensor: {sensor}")


def _sensor_covariance(trace: ManeuverTrace, sensor: str, floor: float) -> FloatArray:
    if sensor == "imu":
        covariance = np.eye(3) * trace.config.imu_noise_std**2
    elif sensor == "wheel_speed":
        covariance = np.eye(4) * trace.config.wheel_noise_std**2
    else:
        covariance = np.eye(2) * trace.config.gnss_noise_std**2
    return covariance + floor * np.eye(covariance.shape[0])


def _block_diagonal(matrices: Sequence[FloatArray]) -> FloatArray:
    """Assemble independent sensor covariances without a SciPy-only helper."""

    size = sum(matrix.shape[0] for matrix in matrices)
    result = np.zeros((size, size), dtype=float)
    offset = 0
    for matrix in matrices:
        width = matrix.shape[0]
        result[offset : offset + width, offset : offset + width] = matrix
        offset += width
    return result


def run_sensor_ablation(
    trace: ManeuverTrace,
    suites: Sequence[SensorSuite | str | Mapping[str, Any]] | None = None,
    *,
    baseline: str = "imu",
    finite_difference_step: float = 1e-5,
    covariance_floor: float = 1e-9,
    regularization: float = 1e-9,
    low_speed_threshold: float = 0.5,
    condition_limit: float = 1e10,
) -> list[AblationResult]:
    """Evaluate sensor combinations on a deterministic :class:`ManeuverTrace`.

    The state convention is ``[vx, vy, yaw_rate, yaw]``.  Each result uses the
    same finite-difference step and covariance floor, so information gain is a
    reproducible log-determinant difference against ``baseline``.
    """

    if not isinstance(trace, ManeuverTrace):
        raise TypeError("trace must be a ManeuverTrace")
    for value, name in ((finite_difference_step, "finite_difference_step"),
                        (covariance_floor, "covariance_floor"),
                        (regularization, "regularization"),
                        (low_speed_threshold, "low_speed_threshold")):
        if not np.isfinite(value) or value < 0.0 or (name == "finite_difference_step" and value == 0.0):
            raise ValueError(f"{name} must be finite and non-negative")
    if not np.isfinite(condition_limit) or condition_limit <= 0.0:
        raise ValueError("condition_limit must be finite and positive")
    if suites is None:
        suites = tuple(_DEFAULT_SUITES)
    normalized = _sort_suites([_coerce_suite(suite) for suite in suites])
    if not normalized:
        raise ValueError("suites must contain at least one suite")
    if len({suite.name for suite in normalized}) != len(normalized):
        raise ValueError("suite names must be unique")
    baseline_suite = next((suite for suite in normalized if suite.name == baseline), None)
    if baseline_suite is None:
        raise ValueError("baseline suite must be included in suites")

    state_dimension = trace.state.shape[1]
    per_sensor: dict[str, tuple[list[FloatArray], FloatArray]] = {}
    for sensor in SENSOR_ORDER:
        model = _measurement_model(sensor, trace.config.wheel_radius)
        covariance = _sensor_covariance(trace, sensor, covariance_floor)
        per_sensor[sensor] = (
            [finite_difference_jacobian(model, state, step=finite_difference_step) for state in trace.state],
            covariance,
        )

    gramians: dict[str, FloatArray] = {}
    for suite in normalized:
        sensitivities: list[FloatArray] = []
        covariances: list[FloatArray] = []
        # Stack independent observations at each time step so mixed-size
        # covariances (3-axis IMU + 2-D GNSS, etc.) remain well-defined.
        for index in range(trace.state.shape[0]):
            time_sensitivities = [per_sensor[sensor][0][index] for sensor in suite.sensors]
            time_covariances = [per_sensor[sensor][1] for sensor in suite.sensors]
            sensitivities.append(np.vstack(time_sensitivities))
            covariances.append(_block_diagonal(time_covariances))
        gramian = empirical_observability_gramian(sensitivities, np.asarray(covariances))
        if gramian.shape != (state_dimension, state_dimension):
            raise ValueError("trace state dimension is incompatible with sensor models")
        gramians[suite.name] = gramian

    baseline_gramian = gramians[baseline_suite.name]
    speed = float(np.mean(np.hypot(trace.state[:, 0], trace.state[:, 1])))
    results: list[AblationResult] = []
    for suite in normalized:
        report = observability_report(
            gramians[suite.name],
            speed=speed,
            low_speed_threshold=low_speed_threshold,
            regularization=regularization,
            condition_limit=condition_limit,
        )
        results.append(
            AblationResult(
                maneuver=trace.config.kind,
                suite=suite.name,
                sensors=suite.sensors,
                gramian=report.gramian,
                effective_rank=effective_rank(report.gramian),
                condition_number=regularized_condition_number(
                    report.gramian, regularization=regularization
                ),
                information_gain=information_gain(
                    report.gramian, baseline_gramian, regularization=regularization
                ),
                crlb=cramer_rao_lower_bound(report.gramian, regularization=regularization),
                ill_conditioned=bool(report.ill_conditioned),
                low_speed=bool(report.low_speed),
            )
        )
    return results


__all__ = ["AblationResult", "SENSOR_ORDER", "SensorSuite", "run_sensor_ablation"]
