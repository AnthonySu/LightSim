"""Tests for the RL baselines module."""

import pytest

from lightsim.benchmarks.rl_baselines import (
    evaluate_controller,
    BaselineResult,
)
from lightsim.core.signal import FixedTimeController, MaxPressureController


class TestRLBaselines:
    def test_evaluate_fixed_time(self):
        """Evaluate FixedTimeController."""
        result = evaluate_controller(
            "single-intersection-v0",
            FixedTimeController(),
            episodes=1,
            episode_steps=200,
        )
        assert isinstance(result, BaselineResult)
        assert result.policy == "FixedTimeController"
        assert result.episodes == 1
        assert result.avg_throughput > 0
        assert result.wall_time > 0

    def test_evaluate_max_pressure(self):
        """Evaluate MaxPressureController."""
        result = evaluate_controller(
            "single-intersection-v0",
            MaxPressureController(min_green=5.0),
            episodes=1,
            episode_steps=200,
        )
        assert isinstance(result, BaselineResult)
        assert result.policy == "MaxPressureController"
        assert result.avg_throughput > 0

    def test_compare_controllers(self):
        """Both controllers should produce valid results."""
        results = {}
        for name, ctrl in [
            ("fixed", FixedTimeController()),
            ("maxp", MaxPressureController(min_green=5.0)),
        ]:
            r = evaluate_controller(
                "single-intersection-v0", ctrl,
                episodes=1, episode_steps=300,
            )
            results[name] = r

        # Both should have non-zero throughput
        assert results["fixed"].avg_throughput > 0
        assert results["maxp"].avg_throughput > 0
