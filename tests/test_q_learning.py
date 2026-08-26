import numpy as np
import pytest

from vehicle_state_estimation.rl import QLearningAgent, StateDiscretizer


def test_default_discretizer_matches_research_state_space():
    discretizer = StateDiscretizer()
    assert discretizer.shape == (3, 3, 4, 8, 8)
    assert discretizer.state_count == 2304
    below = discretizer.encode(1.99, 0.019, 0.24, 7, None)
    boundary = discretizer.encode(2.0, 0.02, 0.25, 7, None)
    assert below != boundary
    assert boundary == discretizer.encode(2.0, 0.02, 0.25, 7, None)


def test_discretizer_rejects_ambiguous_boundaries_and_invalid_masks():
    with pytest.raises(ValueError, match="strictly increasing"):
        StateDiscretizer(speed_edges=(2.0, 2.0))
    with pytest.raises(ValueError, match="health_mask"):
        StateDiscretizer().encode(5.0, 0.1, 0.5, 8, None)


def test_q_update_matches_bellman_equation_and_terminal_has_no_bootstrap():
    agent = QLearningAgent(4, 2, learning_rate=0.5, discount=0.9)
    agent.q_table[1] = [2.0, 4.0]
    assert agent.update(0, 1, reward=1.0, next_state=1, done=False) == pytest.approx(2.3)
    assert agent.update(2, 0, reward=2.0, next_state=1, done=True) == pytest.approx(1.0)


def test_greedy_ties_are_stable_and_seeded_exploration_is_reproducible():
    first = QLearningAgent(3, 4, seed=12)
    second = QLearningAgent(3, 4, seed=12)
    assert first.select_action(0) == 0
    assert [first.select_action(0, epsilon=1.0) for _ in range(8)] == [
        second.select_action(0, epsilon=1.0) for _ in range(8)
    ]


def test_policy_payload_round_trip_and_validation():
    agent = QLearningAgent(3, 2, learning_rate=0.2, discount=0.8)
    agent.q_table[:] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    payload = agent.to_policy_dict(metadata={"seed": 7, "purpose": "unit-test"})
    restored = QLearningAgent.from_policy_dict(payload)
    np.testing.assert_array_equal(restored.q_table, agent.q_table)
    assert restored.learning_rate == 0.2
    assert restored.discount == 0.8
    assert payload["metadata"]["seed"] == 7

    broken = {**payload, "format_version": 99}
    with pytest.raises(ValueError, match="format_version"):
        QLearningAgent.from_policy_dict(broken)
    broken = {**payload, "q_table": [[1.0, float("nan")], [3.0, 4.0], [5.0, 6.0]]}
    with pytest.raises(ValueError, match="finite"):
        QLearningAgent.from_policy_dict(broken)
