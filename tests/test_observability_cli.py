from __future__ import annotations

import json
from pathlib import Path

from experiments.observability_study import main, run_study


def test_fast_study_writes_strict_machine_readable_outputs(tmp_path: Path) -> None:
    output = run_study(output_dir=tmp_path, seed=11, fast=True)

    expected = {"results.json", "ranking.csv", "observability.png", "summary.md"}
    assert expected.issubset({path.name for path in output.iterdir()})
    payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["seed"] == 11
    assert len(payload["results"]) == 4 * 4
    # JSON must not rely on JavaScript's non-standard Infinity/NaN literals.
    text = (output / "results.json").read_text(encoding="utf-8")
    assert "Infinity" not in text
    assert "NaN" not in text
    assert (output / "ranking.csv").read_text(encoding="utf-8").splitlines()[0].startswith(
        "rank,maneuver,suite"
    )
    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert "结论" in summary
    assert "局限" in summary


def test_cli_accepts_config_seed_and_output_arguments(tmp_path: Path) -> None:
    output = tmp_path / "custom"
    returned = main(
        [
            "--config",
            "configs/observability.yaml",
            "--output",
            str(output),
            "--seed",
            "23",
            "--fast",
        ]
    )

    assert returned == output
    payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["seed"] == 23
    assert payload["metadata"]["fast"] is True


def test_default_config_parser_is_dependency_free(tmp_path: Path) -> None:
    output = run_study(output_dir=tmp_path, config_path="does-not-exist.yaml", fast=True)
    payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["config_source"] == "built-in defaults"

