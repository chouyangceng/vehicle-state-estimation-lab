import numpy as np
import pytest

from vehicle_state_estimation.rl import SensorSelectionEnv


def _environment(*, health_masks=(7, 7, 7)) -> SensorSelectionEnv:
    # Each sensor observes a different state direction. Only all three are
    # instantaneously full-rank, which makes observability assertions explicit.
    information = np.zeros((3, 3, 3, 3), dtype=float)
    for step in range(3):
        information[step, 0, 0, 0] = 4.0
        information[step, 1, 1, 1] = 3.0
        information[step, 2, 2, 2] = 2.0
    return SensorSelectionEnv(
        speeds=[4.0, 8.0, 12.0],
        excitations=[0.01, 0.08, 0.16],
        sensor_information=information,
        health_masks=health_masks,
        initial_covariance=np.eye(3),
        process_covariance=np.eye(3) * 0.05,
        danger_variance=10.0,
    )


def test_reset_is_reproducible_and_validates_inputs():
    env = _environment()
    first_state, first_info = env.reset()
    env.step(6)
    second_state, second_info = env.reset()
    assert first_state == second_state
    np.testing.assert_array_equal(first_info["covariance"], second_info["covariance"])

    with pytest.raises(ValueError, match="same non-zero length"):
        SensorSelectionEnv(
            speeds=[1.0],
            excitations=[0.1, 0.2],
            sensor_information=np.zeros((1, 3, 2, 2)),
            health_masks=[7],
            initial_covariance=np.eye(2),
            process_covariance=np.eye(2),
        )
    invalid = np.zeros((1, 3, 2, 2))
    invalid[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        SensorSelectionEnv(
            speeds=[1.0],
            excitations=[0.1],
            sensor_information=invalid,
            health_masks=[7],
            initial_covariance=np.eye(2),
            process_covariance=np.eye(2),
        )


def test_effective_information_reduces_covariance_and_rank_uses_sensor_information():
    env = _environment()
    env.reset()
    _, _, _, info = env.step(0)
    assert np.trace(info["covariance"]) < np.trace(np.eye(3) * 1.05)
    assert info["sensor_information_rank"] == 1
    assert info["posterior_precision_rank"] == 3
    assert info["observable"] is False


def test_faulty_action_has_deterministic_fallback_and_reward_breakdown_sums():
    env = _environment(health_masks=(3, 3, 3))
    env.reset()
    _, reward, terminated, info = env.step(6)  # Requests unavailable GNSS.
    assert terminated is False
    assert info["requested_action"] == 6
    assert info["effective_action"] == 3  # IMU + wheel speed.
    assert info["effective_sensor_mask"] == 3
    assert info["fallback_used"] is True
    assert info["reward_components"]["invalid_action"] < 0
    assert reward == pytest.approx(sum(info["reward_components"].values()))


def test_no_healthy_sensor_terminates_safely_without_fake_observation():
    env = _environment(health_masks=(0, 7, 7))
    env.reset()
    _, reward, terminated, info = env.step(6)
    assert terminated is True
    assert reward < 0
    assert info["effective_action"] is None
    assert info["effective_sensor_mask"] == 0
    assert info["sensor_information_rank"] == 0
    np.testing.assert_array_equal(info["covariance"], np.eye(3))


@pytest.mark.parametrize("action", [-1, 7, 1.5])
def test_step_rejects_invalid_action(action):
    env = _environment()
    env.reset()
    expected_error = TypeError if isinstance(action, float) else ValueError
    with pytest.raises(expected_error, match="action"):
        env.step(action)
