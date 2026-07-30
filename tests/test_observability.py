import numpy as np
import pytest

from vehicle_state_estimation.metrics.observability import (
    ObservabilityReport,
    cramer_rao_lower_bound,
    effective_rank,
    empirical_observability_gramian,
    finite_difference_jacobian,
    information_gain,
    observability_report,
    regularized_condition_number,
)


def test_finite_difference_jacobian_matches_nonlinear_map() -> None:
    def observation(state: np.ndarray) -> np.ndarray:
        return np.array([state[0] ** 2 + state[1], np.sin(state[1])])

    state = np.array([2.0, 0.3])

    jacobian = finite_difference_jacobian(observation, state, step=1e-6)

    expected = np.array([[4.0, 1.0], [0.0, np.cos(0.3)]])
    np.testing.assert_allclose(jacobian, expected, rtol=1e-6, atol=1e-7)


def test_empirical_gramian_is_symmetric_positive_semidefinite() -> None:
    sensitivities = [
        np.array([[1.0, 2.0], [0.0, 1.0]]),
        np.array([[2.0, -1.0], [1.0, 0.5]]),
    ]
    covariance = np.diag([0.5, 2.0])

    gramian = empirical_observability_gramian(sensitivities, covariance)

    np.testing.assert_allclose(gramian, gramian.T, atol=1e-12)
    assert np.linalg.eigvalsh(gramian).min() >= -1e-12


def test_effective_rank_uses_relative_eigenvalue_threshold() -> None:
    information = np.diag([10.0, 1.0, 1e-10])

    assert effective_rank(information, relative_tolerance=1e-6) == 2


def test_regularized_condition_number_remains_finite_for_singular_matrix() -> None:
    information = np.diag([4.0, 0.0])

    condition = regularized_condition_number(information, regularization=1e-3)

    assert condition == pytest.approx(4001.0)


def test_information_gain_is_logdet_difference_from_baseline() -> None:
    baseline = np.diag([1.0, 2.0])
    candidate = np.diag([2.0, 8.0])

    gain = information_gain(candidate, baseline, regularization=0.0)

    assert gain == pytest.approx(np.log(8.0))


def test_crlb_uses_pseudoinverse_and_returns_standard_deviation_bounds() -> None:
    information = np.diag([4.0, 9.0, 0.0])

    bounds = cramer_rao_lower_bound(information, regularization=0.0)

    np.testing.assert_allclose(bounds, np.array([0.5, 1.0 / 3.0, 0.0]))


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: finite_difference_jacobian(lambda x: x, np.ones((2, 1))), "one-dimensional"),
        (lambda: finite_difference_jacobian(lambda x: x, np.ones(2), step=0.0), "positive"),
        (lambda: empirical_observability_gramian([], np.eye(1)), "at least one"),
        (
            lambda: empirical_observability_gramian([np.ones((2, 2))], np.eye(3)),
            "measurement dimension",
        ),
        (lambda: effective_rank(np.ones((2, 3))), "square"),
        (lambda: regularized_condition_number(np.eye(2), regularization=-1.0), "non-negative"),
        (lambda: information_gain(np.eye(2), np.eye(3)), "same shape"),
        (lambda: cramer_rao_lower_bound(np.array([[1.0, np.nan], [0.0, 1.0]])), "finite"),
    ],
)
def test_invalid_inputs_are_rejected(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()


def test_report_flags_low_speed_even_when_regularization_makes_metrics_finite() -> None:
    information = np.diag([3.0, 1e-12])

    report = observability_report(
        information,
        speed=0.05,
        low_speed_threshold=0.5,
        regularization=1e-6,
        condition_limit=1e5,
    )

    assert isinstance(report, ObservabilityReport)
    assert report.ill_conditioned is True
    assert report.low_speed is True
    assert np.isfinite(report.condition_number)
    assert report.effective_rank == 1
    assert report.crlb.shape == (2,)


def test_report_flags_excessive_condition_number_at_normal_speed() -> None:
    report = observability_report(
        np.diag([1.0, 1e-12]),
        speed=10.0,
        regularization=1e-12,
        condition_limit=1e8,
    )

    assert report.ill_conditioned is True
    assert report.low_speed is False
