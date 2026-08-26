from __future__ import annotations

import numpy as np

from vehicle_state_estimation.simulation.maneuvers import ManeuverConfig, generate_maneuver

from .environment import SensorSelectionEnv

MANEUVERS = ("straight", "sine_steer", "double_lane_change", "low_friction_switch")


def make_research_environment(
    episode: int,
    *,
    seed: int = 7,
    windows: int = 24,
) -> SensorSelectionEnv:
    """Build one reproducible manoeuvre/fault episode from the vehicle simulator."""

    if not isinstance(episode, int) or isinstance(episode, bool) or episode < 0:
        raise ValueError("episode must be a non-negative integer")
    if not isinstance(windows, int) or isinstance(windows, bool) or windows < 4:
        raise ValueError("windows must be an integer of at least four")
    samples_per_window = 8
    kind = MANEUVERS[episode % len(MANEUVERS)]
    trace = generate_maneuver(
        ManeuverConfig(
            kind=kind,
            steps=windows * samples_per_window,
            dt=0.05,
            speed=4.0 + 4.0 * (episode % 4),
        )
    )
    state_windows = trace.state.reshape(windows, samples_per_window, 4)
    speeds = np.mean(state_windows[:, :, 0], axis=1)
    excitations = np.mean(np.abs(state_windows[:, :, 2]), axis=1)
    information = _sensor_information(speeds, excitations)
    health_masks = _health_schedule(episode, windows, seed)
    return SensorSelectionEnv(
        speeds=speeds,
        excitations=excitations,
        sensor_information=information,
        health_masks=health_masks,
        initial_covariance=np.diag([1.0, 0.8, 0.25, 0.5]),
        process_covariance=np.diag([0.04, 0.06, 0.025, 0.035]),
        danger_variance=8.0,
    )


def _sensor_information(speeds: np.ndarray, excitations: np.ndarray) -> np.ndarray:
    matrices = np.zeros((speeds.size, 3, 4, 4), dtype=float)
    for index, (speed, excitation) in enumerate(zip(speeds, excitations, strict=True)):
        lateral_scale = 1.0 + min(float(excitation) / 0.12, 2.0)
        speed_scale = 0.7 + min(float(speed) / 15.0, 1.5)
        imu_rows = (
            np.array([0.0, 0.35 * lateral_scale, 1.0, 0.0]),
            np.array([0.0, lateral_scale, 0.0, 0.0]),
        )
        wheel_rows = (np.array([speed_scale, 0.0, 0.0, 0.0]),)
        gnss_rows = (
            np.array([0.6, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.8, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.7]),
        )
        for sensor, (rows, precision) in enumerate(
            ((imu_rows, 5.0), (wheel_rows, 8.0), (gnss_rows, 2.5))
        ):
            matrices[index, sensor] = precision * sum(
                (np.outer(row, row) for row in rows), start=np.zeros((4, 4))
            )
    return matrices


def _health_schedule(episode: int, windows: int, seed: int) -> np.ndarray:
    masks = np.full(windows, 7, dtype=int)
    failure_mode = episode % 4
    if failure_mode == 0:
        return masks
    rng = np.random.default_rng(seed + 104729 * episode)
    duration = max(2, windows // 3)
    start = int(rng.integers(max(1, windows // 5), max(2, windows - duration)))
    failed_bit = 1 << (failure_mode - 1)
    masks[start : start + duration] &= ~failed_bit
    return masks
