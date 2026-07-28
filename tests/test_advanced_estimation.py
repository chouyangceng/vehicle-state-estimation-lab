import numpy as np


def test_vehicle_simulator_generates_reproducible_sensor_log():
    from vehicle_state_estimation.simulation.vehicle_simulator import (
        SimulatorConfig,
        VehicleSimulator,
    )

    config = SimulatorConfig(steps=40, seed=11)
    first = VehicleSimulator(config).run()
    second = VehicleSimulator(config).run()
    assert first.truth.shape == (40, 4)
    assert first.imu.shape == (40, 3)
    assert np.allclose(first.imu, second.imu)
    assert np.all(np.isfinite(first.truth))


def test_adaptive_friction_estimator_tracks_step_change():
    from vehicle_state_estimation.estimators.friction import AdaptiveFrictionEstimator

    estimator = AdaptiveFrictionEstimator(initial=0.9, learning_rate=0.35)
    estimates = [estimator.update(lateral_force=force, normal_load=3500.0) for force in [3000.0] * 6]
    assert estimates[-1] < 0.9
    assert 0.1 <= estimates[-1] <= 1.2


def test_imm_keeps_probabilities_normalized():
    from vehicle_state_estimation.filters.imm import IMMEstimator

    imm = IMMEstimator([0.7, 0.3])
    state, probabilities = imm.update([0.2, 0.8])
    assert state == 0.38
    assert np.isclose(sum(probabilities), 1.0)


def test_fault_manager_reports_sensor_health():
    from vehicle_state_estimation.simulation.faults import FaultManager

    manager = FaultManager(["imu", "gnss"])
    manager.inject("imu", "dropout")
    assert manager.is_healthy("imu") is False
    assert manager.health_report() == {"imu": False, "gnss": True}


def test_monte_carlo_benchmark_returns_summary():
    from vehicle_state_estimation.metrics.monte_carlo import run_monte_carlo

    summary = run_monte_carlo(trials=4, steps=20, seed=4)
    assert summary.trials == 4
    assert summary.rmse_mean >= 0.0
    assert summary.rmse_std >= 0.0
