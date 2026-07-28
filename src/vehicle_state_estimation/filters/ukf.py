from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class UnscentedKalmanFilter:
    state: np.ndarray
    covariance: np.ndarray
    alpha: float = 1e-1
    beta: float = 2.0
    kappa: float = 0.0

    @classmethod
    def identity(cls, dim: int) -> UnscentedKalmanFilter:
        if dim <= 0:
            raise ValueError("filter dimension must be positive")
        return cls(np.zeros(dim), np.eye(dim))

    def _sigma_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.state.size
        lam = self.alpha**2 * (n + self.kappa) - n
        scale = n + lam
        try:
            root = np.linalg.cholesky(scale * self.covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("covariance is not positive definite") from exc
        points = [self.state]
        for i in range(n):
            points.extend([self.state + root[:, i], self.state - root[:, i]])
        wm = np.full(2 * n + 1, 1.0 / (2 * scale))
        wc = wm.copy()
        wm[0] = lam / scale
        wc[0] = lam / scale + (1 - self.alpha**2 + self.beta)
        return np.asarray(points), wm, wc

    def predict(self, transition: Callable[[np.ndarray], np.ndarray], process_noise: np.ndarray) -> np.ndarray:
        points, wm, wc = self._sigma_points()
        propagated = np.asarray([transition(point) for point in points], dtype=float)
        if propagated.shape != points.shape:
            raise ValueError("transition returned an invalid state shape")
        self.state = wm @ propagated
        diff = propagated - self.state
        self.covariance = sum(w * np.outer(d, d) for w, d in zip(wc, diff, strict=True))
        self.covariance += np.asarray(process_noise, dtype=float)
        self.covariance = (self.covariance + self.covariance.T) * 0.5
        return self.state.copy()
