from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .discretization import StateDiscretizer

SENSOR_NAMES = ("imu", "wheel_speed", "gnss")
ACTION_MASKS = (1, 2, 4, 3, 5, 6, 7)


@dataclass(frozen=True)
class RewardWeights:
    """Weights for the environment's individually reported reward terms."""

    uncertainty: float = 4.0
    sensor_cost: float = 1.0
    switching: float = 0.25
    information_gain: float = 2.0
    unobservable: float = 3.0
    invalid_action: float = 5.0
    safety_failure: float = 20.0

    def __post_init__(self) -> None:
        values = np.asarray(list(self.__dict__.values()), dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("reward weights must be finite and non-negative")


class SensorSelectionEnv:
    """Deterministic covariance proxy for interpretable sensor-selection research.

    ``sensor_information[t, sensor]`` is the positive-semidefinite information
    matrix contributed by one sensor in one trajectory window. Observability is
    deliberately computed from their sum, not from the full-rank prior.
    """

    def __init__(
        self,
        *,
        speeds: Sequence[float],
        excitations: Sequence[float],
        sensor_information: np.ndarray,
        health_masks: Sequence[int],
        initial_covariance: np.ndarray,
        process_covariance: np.ndarray,
        sensor_costs: Sequence[float] = (2.0, 1.0, 5.0),
        danger_variance: float = 100.0,
        rank_tolerance: float = 1e-9,
        reward_weights: RewardWeights | None = None,
        discretizer: StateDiscretizer | None = None,
    ) -> None:
        self._speeds = _finite_vector(speeds, "speeds")
        self._excitations = _finite_vector(excitations, "excitations")
        self._health_masks = _health_vector(health_masks)
        self._information = np.asarray(sensor_information, dtype=float)
        self._initial_covariance = _covariance(initial_covariance, "initial_covariance", True)
        self._process_covariance = _covariance(process_covariance, "process_covariance", False)

        horizon = self._speeds.size
        state_size = self._initial_covariance.shape[0]
        if horizon == 0 or self._excitations.size != horizon or self._health_masks.size != horizon:
            raise ValueError("speeds, excitations, and health_masks must have the same non-zero length")
        if self._information.shape != (horizon, 3, state_size, state_size):
            raise ValueError("sensor_information must have shape (steps, 3, state_size, state_size)")
        if self._process_covariance.shape != self._initial_covariance.shape:
            raise ValueError("process_covariance must match initial_covariance")
        if not np.all(np.isfinite(self._information)):
            raise ValueError("sensor_information must contain only finite values")
        for matrix in self._information.reshape(-1, state_size, state_size):
            _require_symmetric_psd(matrix, "sensor_information", positive_definite=False)

        self._sensor_costs = _finite_vector(sensor_costs, "sensor_costs")
        if self._sensor_costs.shape != (3,) or np.any(self._sensor_costs < 0):
            raise ValueError("sensor_costs must contain three non-negative values")
        if not np.isfinite(danger_variance) or danger_variance <= 0:
            raise ValueError("danger_variance must be finite and positive")
        if not np.isfinite(rank_tolerance) or rank_tolerance <= 0:
            raise ValueError("rank_tolerance must be finite and positive")

        self.danger_variance = float(danger_variance)
        self.rank_tolerance = float(rank_tolerance)
        self.reward_weights = reward_weights or RewardWeights()
        self.discretizer = discretizer or StateDiscretizer()
        if self.discretizer.sensor_count != 3 or self.discretizer.action_count != len(ACTION_MASKS):
            raise ValueError("discretizer must use three sensors and seven actions")

        self._step = 0
        self._covariance = self._initial_covariance.copy()
        self._previous_action: int | None = None
        self._terminated = False

    @property
    def horizon(self) -> int:
        return int(self._speeds.size)

    def reset(self) -> tuple[int, dict[str, Any]]:
        self._step = 0
        self._covariance = self._initial_covariance.copy()
        self._previous_action = None
        self._terminated = False
        return self._state(0), {
            "covariance": self._covariance.copy(),
            "health_mask": int(self._health_masks[0]),
        }

    def step(self, action: int) -> tuple[int, float, bool, dict[str, Any]]:
        if self._terminated:
            raise RuntimeError("episode has terminated; call reset before step")
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer in [0, 6]")
        action = int(action)
        if not 0 <= action < len(ACTION_MASKS):
            raise ValueError("action must be an integer in [0, 6]")

        index = self._step
        health_mask = int(self._health_masks[index])
        if health_mask == 0:
            return self._safety_termination(action)

        requested_mask = ACTION_MASKS[action]
        fallback_used = requested_mask & health_mask != requested_mask
        effective_mask = health_mask if fallback_used else requested_mask
        effective_action = ACTION_MASKS.index(effective_mask)

        prior = _symmetrize(self._covariance + self._process_covariance)
        information = np.zeros_like(prior)
        for sensor_index in range(3):
            if effective_mask & (1 << sensor_index):
                information += self._information[index, sensor_index]
        information = _symmetrize(information)
        sensor_rank = int(np.linalg.matrix_rank(information, tol=self.rank_tolerance))

        prior_precision = np.linalg.pinv(prior, hermitian=True, rcond=self.rank_tolerance)
        posterior_precision = _symmetrize(prior_precision + information)
        precision_rank = int(np.linalg.matrix_rank(posterior_precision, tol=self.rank_tolerance))
        self._covariance = self._bounded_inverse(posterior_precision)

        prior_trace = float(np.trace(prior))
        posterior_trace = float(np.trace(self._covariance))
        uncertainty = min(1.0, posterior_trace / (prior.shape[0] * self.danger_variance))
        information_gain = max(0.0, (prior_trace - posterior_trace) / max(prior_trace, 1e-12))
        sensor_cost = float(
            sum(self._sensor_costs[i] for i in range(3) if effective_mask & (1 << i))
        ) / max(float(np.sum(self._sensor_costs)), 1e-12)
        switched = self._previous_action is not None and effective_action != self._previous_action
        weights = self.reward_weights
        reward_components = {
            "uncertainty": -weights.uncertainty * uncertainty,
            "sensor_cost": -weights.sensor_cost * sensor_cost,
            "switching": -weights.switching * float(switched),
            "information_gain": weights.information_gain * information_gain,
            "unobservable": -weights.unobservable * float(sensor_rank < prior.shape[0]),
            "invalid_action": -weights.invalid_action * float(fallback_used),
        }
        reward = float(sum(reward_components.values()))
        self._previous_action = effective_action
        self._step += 1
        self._terminated = self._step >= self.horizon
        next_index = min(self._step, self.horizon - 1)
        next_state = self._state(next_index)
        info = {
            "requested_action": action,
            "requested_sensor_mask": requested_mask,
            "effective_action": effective_action,
            "effective_sensor_mask": effective_mask,
            "fallback_used": fallback_used,
            "health_mask": health_mask,
            "sensor_information_rank": sensor_rank,
            "posterior_precision_rank": precision_rank,
            "observable": sensor_rank == prior.shape[0],
            "prior_covariance": prior.copy(),
            "covariance": self._covariance.copy(),
            "reward_components": reward_components,
        }
        return next_state, reward, self._terminated, info

    def _safety_termination(self, requested_action: int) -> tuple[int, float, bool, dict[str, Any]]:
        self._terminated = True
        components = {
            "uncertainty": 0.0,
            "sensor_cost": 0.0,
            "switching": 0.0,
            "information_gain": 0.0,
            "unobservable": -self.reward_weights.unobservable,
            "invalid_action": -self.reward_weights.safety_failure,
        }
        info = {
            "requested_action": requested_action,
            "requested_sensor_mask": ACTION_MASKS[requested_action],
            "effective_action": None,
            "effective_sensor_mask": 0,
            "fallback_used": True,
            "health_mask": 0,
            "sensor_information_rank": 0,
            "posterior_precision_rank": int(self._covariance.shape[0]),
            "observable": False,
            "prior_covariance": self._covariance.copy(),
            "covariance": self._covariance.copy(),
            "reward_components": components,
        }
        return self._state(self._step), float(sum(components.values())), True, info

    def _state(self, index: int) -> int:
        uncertainty = min(
            1.0,
            float(np.trace(self._covariance))
            / (self._covariance.shape[0] * self.danger_variance),
        )
        return self.discretizer.encode(
            float(self._speeds[index]),
            float(self._excitations[index]),
            uncertainty,
            int(self._health_masks[index]),
            self._previous_action,
        )

    def _bounded_inverse(self, precision: np.ndarray) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(_symmetrize(precision))
        minimum_precision = 1.0 / self.danger_variance
        bounded = np.maximum(eigenvalues, minimum_precision)
        covariance = (eigenvectors * (1.0 / bounded)) @ eigenvectors.T
        covariance = _symmetrize(covariance)
        if not np.all(np.isfinite(covariance)):
            raise FloatingPointError("posterior covariance became non-finite")
        return covariance


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional sequence")
    return array


def _health_vector(values: Sequence[int]) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("health_masks must be a one-dimensional integer sequence")
    if np.any(array < 0) or np.any(array > 7):
        raise ValueError("health_masks must be between zero and seven")
    return array.astype(int)


def _covariance(value: np.ndarray, name: str, positive_definite: bool) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a non-empty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    _require_symmetric_psd(matrix, name, positive_definite)
    return _symmetrize(matrix)


def _require_symmetric_psd(matrix: np.ndarray, name: str, positive_definite: bool) -> None:
    if not np.allclose(matrix, matrix.T, rtol=1e-9, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(_symmetrize(matrix))
    threshold = 1e-12 if positive_definite else -1e-12
    if (positive_definite and np.min(eigenvalues) <= threshold) or (
        not positive_definite and np.min(eigenvalues) < threshold
    ):
        qualifier = "positive definite" if positive_definite else "positive semidefinite"
        raise ValueError(f"{name} must be {qualifier}")


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.T) * 0.5
