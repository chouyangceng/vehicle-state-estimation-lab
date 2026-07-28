import numpy as np

from vehicle_state_estimation.filters.ekf import ExtendedKalmanFilter


def test_ekf_covariance_is_symmetric_psd():
    f = ExtendedKalmanFilter.identity(dim=4)
    f.predict(np.eye(4), np.eye(4) * 0.01)
    f.update(np.zeros(2), np.zeros((2, 4)), np.eye(2))
    assert np.allclose(f.covariance, f.covariance.T, atol=1e-10)
    assert np.linalg.eigvalsh(f.covariance).min() >= -1e-9


def test_sensor_fault_marks_missing_measurement():
    from vehicle_state_estimation.simulation.sensors import SensorFault

    result = SensorFault(drop_probability=1.0, seed=1).apply(np.array([1.0]))
    assert result.is_valid is False


def test_ukf_predict_preserves_finite_state():
    from vehicle_state_estimation.filters.ukf import UnscentedKalmanFilter

    f = UnscentedKalmanFilter.identity(dim=2)
    f.predict(lambda x: x + np.array([1.0, -1.0]), np.eye(2) * 0.01)
    assert np.all(np.isfinite(f.state))
    assert np.all(np.linalg.eigvalsh(f.covariance) > 0)


def test_sensor_fault_adds_bias_and_delay():
    from vehicle_state_estimation.simulation.sensors import SensorFault

    result = SensorFault(bias=0.2, delay_steps=3, seed=2).apply(np.array([1.0]), timestamp=5.0)
    assert result.is_valid is True
    assert np.allclose(result.value, [1.2])
    assert result.timestamp == 8.0
