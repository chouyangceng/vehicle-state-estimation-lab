import numpy as np

from vehicle_state_estimation.rl import (
    AlwaysAllPolicy,
    GreedyInformationPolicy,
    LowestCostPolicy,
    RandomPolicy,
    SensorSelectionEnv,
    evaluate_policy,
    train_q_learning,
)


def _factory(episode: int) -> SensorSelectionEnv:
    steps = 4
    information = np.zeros((steps, 3, 2, 2), dtype=float)
    information[:, 0, 0, 0] = 5.0
    information[:, 1, 1, 1] = 4.0
    information[:, 2] = np.eye(2) * 2.0
    health_schedules = ((7, 7, 7, 7), (7, 3, 3, 7), (7, 6, 6, 7))
    return SensorSelectionEnv(
        speeds=[6.0, 8.0, 10.0, 12.0],
        excitations=[0.01, 0.05, 0.1, 0.15],
        sensor_information=information,
        health_masks=health_schedules[episode % len(health_schedules)],
        initial_covariance=np.eye(2),
        process_covariance=np.eye(2) * 0.1,
        danger_variance=10.0,
    )


def test_seeded_training_is_numerically_reproducible():
    first_agent, first_history = train_q_learning(_factory, episodes=12, seed=19)
    second_agent, second_history = train_q_learning(_factory, episodes=12, seed=19)
    np.testing.assert_array_equal(first_agent.q_table, second_agent.q_table)
    assert first_history == second_history
    assert len(first_history) == 12
    assert all(item["steps"] == 4 for item in first_history)


def test_evaluation_reports_cost_uncertainty_observability_and_fallbacks():
    result = evaluate_policy(_factory, AlwaysAllPolicy(), episodes=3)
    assert result["episodes"] == 3
    assert result["steps"] == 12
    assert result["mean_uncertainty"] >= 0
    assert result["p95_uncertainty"] >= result["mean_uncertainty"]
    assert 0 <= result["mean_sensor_cost"] <= 1
    assert result["fallback_steps"] == 4
    assert sum(result["action_counts"].values()) == 12
    assert set(result["reward_components"]) == {
        "information_gain",
        "invalid_action",
        "sensor_cost",
        "switching",
        "uncertainty",
        "unobservable",
    }


def test_baselines_are_deterministic_or_seed_reproducible():
    assert AlwaysAllPolicy()(123) == 6
    assert LowestCostPolicy()(123) == 1
    first = RandomPolicy(seed=8)
    second = RandomPolicy(seed=8)
    assert [first(0) for _ in range(10)] == [second(999) for _ in range(10)]
    environment = _factory(1)
    environment.reset()
    greedy_action = GreedyInformationPolicy()(0, environment)
    assert 0 <= greedy_action < 7
    assert environment.greedy_information_action() == greedy_action
