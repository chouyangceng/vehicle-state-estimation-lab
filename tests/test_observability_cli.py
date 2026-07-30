from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.observability_study import _read_config, _write_ranking, main, run_study


def test_fast_study_writes_strict_machine_readable_outputs(tmp_path: Path) -> None:
    output = run_study(output_dir=tmp_path, seed=11, fast=True)

    expected = {"results.json", "ranking.csv", "observability.png", "summary.md"}
    assert expected.issubset({path.name for path in output.iterdir()})
    payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["seed"] == 11
    assert payload["metadata"]["seed_semantics"] == "metadata_only_deterministic_replay"
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


@pytest.mark.parametrize("key", ["maneuvers", "sensor_suites"])
def test_config_rejects_explicitly_empty_collections(tmp_path: Path, key: str) -> None:
    config = tmp_path / "empty.yaml"
    config.write_text(f"{key}:\n", encoding="utf-8")

    with pytest.raises(ValueError, match=f"{key} must contain at least one"):
        run_study(config_path=config, output_dir=tmp_path / "out", fast=True)


def test_config_strips_utf8_bom_and_rejects_unknown_key(tmp_path: Path) -> None:
    config = tmp_path / "bom.yaml"
    config.write_text("\ufeffseed: 23\n", encoding="utf-8")
    parsed, source = _read_config(config)
    assert parsed["seed"] == 23
    assert source == str(config)

    config.write_text("seed: 23\nnot_a_supported_key: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config key 'not_a_supported_key'"):
        _read_config(config)


def test_config_rejects_malformed_lines_with_clear_error(tmp_path: Path) -> None:
    config = tmp_path / "malformed.yaml"
    config.write_text("seed: 23\nthis is not yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="malformed config line 2"):
        _read_config(config)


def test_ranking_never_promotes_partially_unobservable_result(tmp_path: Path) -> None:
    results = [
        {
            "maneuver": "sine_steer",
            "suite": "bounded",
            "sensors": ["imu", "gnss"],
            "effective_rank": 3,
            "condition_number": 10.0,
            "information_gain": 1.0,
            "crlb": [0.1, 0.2, 0.3, 0.4],
            "ill_conditioned": False,
            "low_speed": False,
        },
        {
            "maneuver": "sine_steer",
            "suite": "partial",
            "sensors": ["imu"],
            "effective_rank": 4,
            "condition_number": 1.0,
            "information_gain": 100.0,
            "crlb": [None, 0.01, 0.01, 0.01],
            "ill_conditioned": False,
            "low_speed": False,
        },
    ]
    _write_ranking(results, tmp_path / "ranking.csv")
    rows = (tmp_path / "ranking.csv").read_text(encoding="utf-8").splitlines()
    assert rows[1].split(",")[2] == "bounded"
    assert rows[2].split(",")[2] == "partial"
    assert rows[2].endswith("-inf")
