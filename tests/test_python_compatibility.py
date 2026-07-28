"""Regression checks for the Python version shipped with ROS 2 Humble/Iron."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_project_metadata_and_ci_cover_python_310() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert re.search(r'requires-python\s*=\s*">=3\.10"', pyproject)
    assert re.search(r'target-version\s*=\s*"py310"', pyproject)
    assert 'python-version: ["3.10", "3.11"]' in workflow


def test_python_sources_parse_with_python_310_grammar() -> None:
    source_roots = (ROOT / "src", ROOT / "ros2", ROOT / "tests")
    source_files = [path for source_root in source_roots for path in source_root.rglob("*.py")]

    for source_file in source_files:
        ast.parse(
            source_file.read_text(encoding="utf-8"),
            filename=str(source_file),
            feature_version=(3, 10),
        )
