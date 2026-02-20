"""Smoke tests: run each example script to ensure it doesn't crash."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

EXAMPLES = [
    "examples/quickstart.py",
    "examples/custom_network.py",
    "examples/max_pressure.py",
    "examples/multi_agent.py",
    "examples/json_network.py",
]


@pytest.mark.parametrize("script", EXAMPLES)
def test_example_runs(script):
    """Each example script should exit cleanly."""
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"{script} failed with:\n"
        f"STDOUT: {result.stdout[-500:]}\n"
        f"STDERR: {result.stderr[-500:]}"
    )
