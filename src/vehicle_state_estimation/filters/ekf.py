from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate_square(matrix: np.ndarray, size: int, name: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite square matrix of size {size}")
    return matrix


@dataclass
class ExtendedKalmanFilter:
    state: np.ndarray
    covariance: np.ndarray

    @classmethod
    def identity(cls, dim: int) -> ExtendedKalmanFilter:
        if dim <= 0:
            raise ValueError("filter dimension must be positive")
        return cls(np.zeros(dim, dtype=float), np.eye(dim, dtype=float))

    def predict(self, transition: np.ndarray, process_noise: np.ndarray) -> np.ndarray:
        n = self.state.size
        transition = _validate_square(transition, n, "transition")
        process_noise = _validate_square(process_noise, n, "process_noise")
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process_noise
        self._stabilize_covariance()
        return self.state.copy()

    def update(self, measurement: np.ndarray, observation_matrix: np.ndarray,
               measurement_noise: np.ndarray) -> np.ndarray:
        z = np.asarray(measurement, dtype=float)
        h = np.asarray(observation_matrix, dtype=float)
        r = np.asarray(measurement_noise, dtype=float)
        if z.ndim != 1 or h.shape != (z.size, self.state.size):
            raise ValueError("measurement and observation matrix shapes are inconsistent")
        if r.shape != (z.size, z.size) or not np.all(np.isfinite(z)):
            raise ValueError("measurement noise or measurement is invalid")
        innovation = z - h @ self.state
        s = h @ self.covariance @ h.T + r
        try:
            gain = np.linalg.solve(s, (self.covariance @ h.T).T).T
        except np.linalg.LinAlgError as exc:
            raise ValueError("innovation covariance is singular") from exc
        identity = np.eye(self.state.size)
        self.state = self.state + gain @ innovation
        self.covariance = (identity - gain @ h) @ self.covariance @ (identity - gain @ h).T + gain @ r @ gain.T
        self._stabilize_covariance()
        return self.state.copy()

    def _stabilize_covariance(self) -> None:
        self.covariance = (self.covariance + self.covariance.T) * 0.5
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
        self.covariance = eigenvectors @ np.diag(np.maximum(eigenvalues, 1e-12)) @ eigenvectors.T
