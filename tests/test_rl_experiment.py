from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.rl_sensor_selection import run_study
from vehicle_state_estimation.rl import QLearningAgent
from vehicle_state_estimation.rl.scenarios import make_research_environment


def test_research_scenario_is_seeded_and_executable():
    first = make_research_environment(5, seed=13, windows=8)
    second = make_research_environment(5, seed=13, windows=8)
    first_state, first_info = first.reset()
    second_state, second_info = second.reset()
    assert first_state == second_state
    assert first_info["health_mask"] == second_info["health_mask"]
    assert first.horizon == 8


def test_fast_rl_study_generates_strict_complete_artifacts(tmp_path: Path):
    output = run_study(output_dir=tmp_path, seed=17, fast=True)
    expected = {
        "policy.json",
        "training_history.csv",
        "evaluation.json",
        "policy_comparison.csv",
        "paired_comparison.csv",
        "training_curve.png",
        "tradeoff.png",
        "action_usage.png",
        "summary.md",
    }
    assert expected == {item.name for item in output.iterdir()}
    policy_text = (output / "policy.json").read_text(encoding="utf-8")
    assert "NaN" not in policy_text and "Infinity" not in policy_text
    agent = QLearningAgent.from_policy_dict(json.loads(policy_text))
    assert agent.q_table.shape == (2304, 7)
    evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert set(evaluation["policies"]) == {
        "q_learning",
        "always_all",
        "lowest_cost",
        "random",
        "greedy_information",
    }
    assert set(evaluation["paired_comparisons"]) == {
        "always_all", "lowest_cost", "random", "greedy_information"
    }
    assert evaluation["paired_comparisons"]["always_all"]["mean_sensor_cost"]["ci95_low"] < 0
    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert "指标边界" in summary
    assert "不能替代道路和硬件在环试验" in summary
    assert all((output / name).stat().st_size > 1000 for name in expected if name.endswith(".png"))

    with pytest.raises(FileExistsError, match="not empty"):
        run_study(output_dir=tmp_path, seed=17, fast=True)
