from vehicle_state_estimation.estimators import AdaptiveFrictionEstimator
from vehicle_state_estimation.metrics import run_monte_carlo
from vehicle_state_estimation.simulation import FaultManager, SimulatorConfig, VehicleSimulator

if __name__ == "__main__":
    log = VehicleSimulator(SimulatorConfig(steps=400, seed=21)).run()
    friction = AdaptiveFrictionEstimator(initial=0.9)
    for lateral_acceleration in log.imu[:, 1]:
        friction.update(abs(float(lateral_acceleration)) * 1500.0, 1500.0 * 9.81)
    faults = FaultManager(["imu", "gnss", "wheel_speed"])
    faults.inject("gnss", "dropout")
    summary = run_monte_carlo(trials=30, steps=200, seed=21)
    print({"friction_estimate": friction.estimate, "faults": faults.health_report(), "monte_carlo": summary})
