from __future__ import annotations

import numpy as np


def rmse(predicted: np.ndarray, truth: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if predicted.shape != truth.shape or predicted.size == 0:
        raise ValueError("predicted and truth must have equal non-empty shapes")
    return float(np.sqrt(np.mean((predicted - truth) ** 2)))


def nis(innovation: np.ndarray, covariance: np.ndarray) -> float:
    innovation = np.asarray(innovation, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if innovation.ndim != 1 or covariance.shape != (innovation.size, innovation.size):
        raise ValueError("innovation and covariance shapes are inconsistent")
    return float(innovation @ np.linalg.solve(covariance, innovation))


def nees(error: np.ndarray, covariance: np.ndarray) -> float:
    return nis(error, covariance)
