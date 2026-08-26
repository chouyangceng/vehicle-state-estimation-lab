from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


class QLearningAgent:
    """Small seeded tabular Q-learning agent with deterministic inference."""

    def __init__(
        self,
        state_count: int,
        action_count: int,
        learning_rate: float = 0.1,
        discount: float = 0.95,
        seed: int = 0,
    ) -> None:
        if state_count <= 0 or action_count <= 0:
            raise ValueError("state_count and action_count must be positive")
        if not 0 < learning_rate <= 1 or not 0 <= discount <= 1:
            raise ValueError("learning_rate or discount is invalid")
        self.learning_rate = float(learning_rate)
        self.discount = float(discount)
        self.q_table = np.zeros((state_count, action_count), dtype=float)
        self._rng = np.random.default_rng(seed)

    def select_action(self, state: int, epsilon: float = 0.0) -> int:
        self._validate_state(state)
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between zero and one")
        if epsilon > 0 and self._rng.random() < epsilon:
            return int(self._rng.integers(self.q_table.shape[1]))
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        self._validate_state(state)
        self._validate_state(next_state)
        if not 0 <= action < self.q_table.shape[1] or not np.isfinite(reward):
            raise ValueError("action or reward is invalid")
        bootstrap = 0.0 if done else float(np.max(self.q_table[next_state]))
        target = float(reward) + self.discount * bootstrap
        current = self.q_table[state, action]
        updated = current + self.learning_rate * (target - current)
        self.q_table[state, action] = updated
        return float(updated)

    def _validate_state(self, state: int) -> None:
        if not isinstance(state, (int, np.integer)) or not 0 <= state < self.q_table.shape[0]:
            raise ValueError("state is outside the Q table")

    def to_policy_dict(self, *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Return a strict-JSON-compatible, versioned inference payload."""

        if not np.all(np.isfinite(self.q_table)):
            raise ValueError("Q table must contain only finite values")
        return {
            "format_version": 1,
            "algorithm": "tabular_q_learning",
            "state_count": int(self.q_table.shape[0]),
            "action_count": int(self.q_table.shape[1]),
            "learning_rate": self.learning_rate,
            "discount": self.discount,
            "q_table": self.q_table.tolist(),
            "metadata": dict(metadata or {}),
        }

    @classmethod
    def from_policy_dict(cls, payload: Mapping[str, Any]) -> QLearningAgent:
        """Load a validated payload; unknown versions fail closed."""

        if not isinstance(payload, Mapping):
            raise TypeError("policy payload must be a mapping")
        if payload.get("format_version") != 1:
            raise ValueError("unsupported policy format_version")
        if payload.get("algorithm") != "tabular_q_learning":
            raise ValueError("unsupported policy algorithm")
        try:
            state_count = int(payload["state_count"])
            action_count = int(payload["action_count"])
            learning_rate = float(payload["learning_rate"])
            discount = float(payload["discount"])
            q_table = np.asarray(payload["q_table"], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("policy payload is incomplete or malformed") from error
        if q_table.shape != (state_count, action_count):
            raise ValueError("Q table shape does not match policy dimensions")
        if not np.all(np.isfinite(q_table)):
            raise ValueError("Q table must contain only finite values")
        agent = cls(state_count, action_count, learning_rate, discount)
        agent.q_table[:] = q_table
        return agent
