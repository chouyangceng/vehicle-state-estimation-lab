"""Numerically robust local observability and information metrics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ObservabilityReport:
    """Summary of local information quality at one operating point."""

    gramian: FloatArray
    effective_rank: int
    condition_number: float
    crlb: FloatArray
    low_speed: bool
    ill_conditioned: bool


def _finite_array(value: ArrayLike, *, name: str, ndim: int | None = None) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        dimension = "one-dimensional" if ndim == 1 else f"{ndim}-dimensional"
        raise ValueError(f"{name} must be {dimension}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _symmetric_matrix(value: ArrayLike, *, name: str) -> FloatArray:
    matrix = _finite_array(value, name=name, ndim=2)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    return 0.5 * (matrix + matrix.T)


def _positive_semidefinite(value: ArrayLike, *, name: str) -> FloatArray:
    matrix = _symmetric_matrix(value, name=name)
    eigenvalues = np.linalg.eigvalsh(matrix)
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    if eigenvalues.min(initial=0.0) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    return matrix


def _validate_non_negative(value: float, *, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def finite_difference_jacobian(
    function: Callable[[FloatArray], ArrayLike],
    point: ArrayLike,
    *,
    step: float = 1e-6,
) -> FloatArray:
    """Evaluate a central finite-difference Jacobian at ``point``."""

    point_array = _finite_array(point, name="point", ndim=1)
    step = float(step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be finite and positive")

    baseline = _finite_array(function(point_array.copy()), name="function output", ndim=1)
    jacobian = np.empty((baseline.size, point_array.size), dtype=float)
    for index in range(point_array.size):
        delta = np.zeros_like(point_array)
        delta[index] = step
        positive = _finite_array(
            function(point_array + delta), name="function output", ndim=1
        )
        negative = _finite_array(
            function(point_array - delta), name="function output", ndim=1
        )
        if positive.shape != baseline.shape or negative.shape != baseline.shape:
            raise ValueError("function output shape must remain constant")
        jacobian[:, index] = (positive - negative) / (2.0 * step)
    return jacobian


def empirical_observability_gramian(
    sensitivities: Sequence[ArrayLike],
    covariance: ArrayLike | Sequence[ArrayLike],
) -> FloatArray:
    """Accumulate ``H.T @ pinv(R) @ H`` over a sensitivity sequence."""

    if len(sensitivities) == 0:
        raise ValueError("sensitivities must contain at least one matrix")
    matrices = [_finite_array(item, name="sensitivity", ndim=2) for item in sensitivities]
    state_dimension = matrices[0].shape[1]
    if state_dimension == 0 or any(item.shape[1] != state_dimension for item in matrices):
        raise ValueError("all sensitivities must share a non-zero state dimension")

    covariance_array = np.asarray(covariance, dtype=float)
    if covariance_array.ndim == 2:
        covariances = [covariance_array] * len(matrices)
    elif covariance_array.ndim == 3 and covariance_array.shape[0] == len(matrices):
        covariances = [covariance_array[index] for index in range(len(matrices))]
    else:
        raise ValueError("covariance must be one matrix or one matrix per sensitivity")

    gramian = np.zeros((state_dimension, state_dimension), dtype=float)
    for sensitivity, noise in zip(matrices, covariances, strict=True):
        covariance_matrix = _positive_semidefinite(noise, name="covariance")
        if covariance_matrix.shape[0] != sensitivity.shape[0]:
            raise ValueError("covariance measurement dimension must match sensitivity rows")
        gramian += sensitivity.T @ np.linalg.pinv(covariance_matrix) @ sensitivity
    return 0.5 * (gramian + gramian.T)


def effective_rank(information: ArrayLike, *, relative_tolerance: float = 1e-9) -> int:
    """Count eigen-directions above a scale-relative threshold."""

    matrix = _positive_semidefinite(information, name="information")
    tolerance = _validate_non_negative(relative_tolerance, name="relative_tolerance")
    eigenvalues = np.linalg.eigvalsh(matrix)
    largest = float(eigenvalues.max(initial=0.0))
    if largest == 0.0:
        return 0
    return int(np.count_nonzero(eigenvalues > largest * tolerance))


def regularized_condition_number(
    information: ArrayLike, *, regularization: float = 1e-9
) -> float:
    """Return the spectral condition number of ``information + lambda I``."""

    matrix = _positive_semidefinite(information, name="information")
    regularization = _validate_non_negative(regularization, name="regularization")
    eigenvalues = np.linalg.eigvalsh(matrix) + regularization
    smallest = float(eigenvalues.min(initial=np.inf))
    largest = float(eigenvalues.max(initial=0.0))
    if smallest <= 0.0:
        return float("inf")
    return largest / smallest


def information_gain(
    candidate: ArrayLike,
    baseline: ArrayLike,
    *,
    regularization: float = 1e-9,
) -> float:
    """Compute baseline-relative log-determinant information gain."""

    candidate_matrix = _positive_semidefinite(candidate, name="candidate")
    baseline_matrix = _positive_semidefinite(baseline, name="baseline")
    if candidate_matrix.shape != baseline_matrix.shape:
        raise ValueError("candidate and baseline must have the same shape")
    regularization = _validate_non_negative(regularization, name="regularization")

    candidate_eigenvalues = np.linalg.eigvalsh(candidate_matrix) + regularization
    baseline_eigenvalues = np.linalg.eigvalsh(baseline_matrix) + regularization
    if np.any(candidate_eigenvalues <= 0.0) or np.any(baseline_eigenvalues <= 0.0):
        raise ValueError("regularized information matrices must be positive definite")
    return float(np.log(candidate_eigenvalues).sum() - np.log(baseline_eigenvalues).sum())


def cramer_rao_lower_bound(
    information: ArrayLike, *, regularization: float = 1e-9
) -> FloatArray:
    """Return per-state standard-deviation bounds from a stable pseudoinverse."""

    matrix = _positive_semidefinite(information, name="information")
    regularization = _validate_non_negative(regularization, name="regularization")
    regularized = matrix + regularization * np.eye(matrix.shape[0])
    covariance_bound = np.linalg.pinv(regularized)
    variances = np.maximum(np.diag(covariance_bound), 0.0)
    return np.sqrt(variances)


def observability_report(
    information: ArrayLike,
    *,
    speed: float,
    low_speed_threshold: float = 0.5,
    regularization: float = 1e-9,
    relative_tolerance: float = 1e-9,
    condition_limit: float = 1e10,
) -> ObservabilityReport:
    """Summarize rank and numerical warnings without hiding low-speed ambiguity."""

    matrix = _positive_semidefinite(information, name="information")
    speed = float(speed)
    if not np.isfinite(speed) or speed < 0.0:
        raise ValueError("speed must be finite and non-negative")
    low_speed_threshold = _validate_non_negative(
        low_speed_threshold, name="low_speed_threshold"
    )
    condition_limit = float(condition_limit)
    if not np.isfinite(condition_limit) or condition_limit <= 0.0:
        raise ValueError("condition_limit must be finite and positive")

    rank = effective_rank(matrix, relative_tolerance=relative_tolerance)
    condition = regularized_condition_number(matrix, regularization=regularization)
    low_speed = speed < low_speed_threshold
    ill_conditioned = low_speed or rank < matrix.shape[0] or condition > condition_limit
    return ObservabilityReport(
        gramian=matrix.copy(),
        effective_rank=rank,
        condition_number=condition,
        crlb=cramer_rao_lower_bound(matrix, regularization=regularization),
        low_speed=low_speed,
        ill_conditioned=ill_conditioned,
    )
