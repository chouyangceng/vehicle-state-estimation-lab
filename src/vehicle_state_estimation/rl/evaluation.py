from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np

from .environment import SensorSelectionEnv
from .q_learning import QLearningAgent

EnvironmentFactory = Callable[[int], SensorSelectionEnv]
Policy = Callable[[int, SensorSelectionEnv], int]


def train_q_learning(
    environment_factory: EnvironmentFactory,
    *,
    episodes: int,
    seed: int = 0,
    learning_rate: float = 0.1,
    discount: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
) -> tuple[QLearningAgent, list[dict[str, float | int]]]:
    """Train on deterministic episode factories and return transparent history."""

    if not isinstance(episodes, int) or isinstance(episodes, bool) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if not 0 <= epsilon_end <= epsilon_start <= 1:
        raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")

    first_environment = environment_factory(0)
    if not isinstance(first_environment, SensorSelectionEnv):
        raise TypeError("environment_factory must return SensorSelectionEnv")
    agent = QLearningAgent(
        first_environment.discretizer.state_count,
        7,
        learning_rate=learning_rate,
        discount=discount,
        seed=seed,
    )
    history: list[dict[str, float | int]] = []
    for episode in range(episodes):
        environment = first_environment if episode == 0 else environment_factory(episode)
        if not isinstance(environment, SensorSelectionEnv):
            raise TypeError("environment_factory must return SensorSelectionEnv")
        epsilon_fraction = episode / max(episodes - 1, 1)
        epsilon = epsilon_start + epsilon_fraction * (epsilon_end - epsilon_start)
        state, _ = environment.reset()
        terminated = False
        total_reward = 0.0
        total_uncertainty = 0.0
        total_cost = 0.0
        fallback_steps = 0
        steps = 0
        while not terminated:
            action = agent.select_action(state, epsilon=epsilon)
            next_state, reward, terminated, info = environment.step(action)
            agent.update(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward
            total_uncertainty += float(info["normalized_uncertainty"])
            total_cost += float(info["normalized_sensor_cost"])
            fallback_steps += int(info["fallback_used"])
            steps += 1
        history.append(
            {
                "episode": episode,
                "epsilon": float(epsilon),
                "return": float(total_reward),
                "mean_uncertainty": total_uncertainty / steps,
                "mean_sensor_cost": total_cost / steps,
                "fallback_steps": fallback_steps,
                "steps": steps,
            }
        )
    return agent, history


def evaluate_policy(
    environment_factory: EnvironmentFactory,
    policy: Policy,
    *,
    episodes: int,
) -> dict[str, Any]:
    """Evaluate one policy with metrics shared by learned and fixed baselines."""

    if not isinstance(episodes, int) or isinstance(episodes, bool) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    uncertainties: list[float] = []
    costs: list[float] = []
    returns: list[float] = []
    reward_components: defaultdict[str, float] = defaultdict(float)
    action_counts: Counter[int] = Counter()
    fallback_steps = 0
    unobservable_steps = 0
    switching_steps = 0
    episode_metrics: list[dict[str, float | int]] = []

    for episode in range(episodes):
        environment = environment_factory(episode)
        if not isinstance(environment, SensorSelectionEnv):
            raise TypeError("environment_factory must return SensorSelectionEnv")
        state, _ = environment.reset()
        terminated = False
        episode_return = 0.0
        episode_uncertainties: list[float] = []
        episode_costs: list[float] = []
        while not terminated:
            action = policy(state, environment)
            state, reward, terminated, info = environment.step(action)
            episode_return += reward
            uncertainties.append(float(info["normalized_uncertainty"]))
            costs.append(float(info["normalized_sensor_cost"]))
            episode_uncertainties.append(float(info["normalized_uncertainty"]))
            episode_costs.append(float(info["normalized_sensor_cost"]))
            fallback_steps += int(info["fallback_used"])
            unobservable_steps += int(not info["observable"])
            switching_steps += int(info["switched"])
            effective_action = info["effective_action"]
            if effective_action is not None:
                action_counts[int(effective_action)] += 1
            for name, value in info["reward_components"].items():
                reward_components[str(name)] += float(value)
        returns.append(episode_return)
        episode_metrics.append(
            {
                "episode": episode,
                "return": float(episode_return),
                "mean_uncertainty": float(np.mean(episode_uncertainties)),
                "mean_sensor_cost": float(np.mean(episode_costs)),
            }
        )

    step_count = len(uncertainties)
    return {
        "episodes": episodes,
        "steps": step_count,
        "mean_return": float(np.mean(returns)),
        "mean_uncertainty": float(np.mean(uncertainties)),
        "p95_uncertainty": float(np.percentile(uncertainties, 95)),
        "mean_sensor_cost": float(np.mean(costs)),
        "fallback_steps": fallback_steps,
        "unobservable_steps": unobservable_steps,
        "switching_steps": switching_steps,
        "action_counts": {str(action): action_counts[action] for action in range(7)},
        "reward_components": dict(sorted(reward_components.items())),
        "episode_metrics": episode_metrics,
    }
