"""Tests for reward functions."""

from __future__ import annotations

import pytest
import numpy as np

from lightsim.envs.rewards import (
    get_reward_function,
    QueueReward,
    PressureReward,
    DelayReward,
    WaitingTimeReward,
    ThroughputReward,
    NormalizedThroughputReward,
)
from lightsim.core.engine import SimulationEngine
from lightsim.core.signal import FixedTimeController
from lightsim.benchmarks.scenarios import get_scenario


@pytest.fixture
def single_intersection_engine():
    """Create a single-intersection engine with some vehicles."""
    network, demand = get_scenario("single-intersection-v0")()
    engine = SimulationEngine(
        network=network, dt=1.0,
        controller=FixedTimeController(),
        demand_profiles=demand,
    )
    engine.reset(seed=42)
    # Run a few steps to inject some vehicles
    for _ in range(100):
        engine.step()
    return engine


@pytest.fixture
def agent_node(single_intersection_engine):
    """Get the signalized node ID."""
    from lightsim.core.types import NodeType
    for node_id, node in single_intersection_engine.network.nodes.items():
        if node.node_type == NodeType.SIGNALIZED:
            return node_id
    pytest.fail("No signalized node found")


class TestRegistry:
    def test_get_known_reward(self):
        r = get_reward_function("queue")
        assert isinstance(r, QueueReward)

    def test_get_all_rewards(self):
        for name in ["queue", "pressure", "delay", "waiting_time",
                      "throughput", "normalized_throughput"]:
            r = get_reward_function(name)
            assert r is not None

    def test_get_unknown_reward_raises(self):
        with pytest.raises(KeyError, match="Unknown reward function"):
            get_reward_function("nonexistent_reward")


class TestQueueReward:
    def test_returns_float(self, single_intersection_engine, agent_node):
        r = QueueReward()
        val = r.compute(single_intersection_engine, agent_node)
        assert isinstance(val, float)

    def test_is_non_positive(self, single_intersection_engine, agent_node):
        """Queue reward is always <= 0 (negative queue)."""
        r = QueueReward()
        val = r.compute(single_intersection_engine, agent_node)
        assert val <= 0.0

    def test_empty_network_is_zero(self):
        """With no vehicles, queue reward should be 0."""
        network, demand = get_scenario("single-intersection-v0")()
        engine = SimulationEngine(
            network=network, dt=1.0,
            controller=FixedTimeController(),
            demand_profiles=[],  # no demand
        )
        engine.reset()
        from lightsim.core.types import NodeType
        node_id = None
        for nid, node in engine.network.nodes.items():
            if node.node_type == NodeType.SIGNALIZED:
                node_id = nid
                break
        r = QueueReward()
        val = r.compute(engine, node_id)
        assert val == 0.0


class TestPressureReward:
    def test_returns_float(self, single_intersection_engine, agent_node):
        r = PressureReward()
        val = r.compute(single_intersection_engine, agent_node)
        assert isinstance(val, float)

    def test_is_non_positive(self, single_intersection_engine, agent_node):
        """Pressure reward = -|pressure|, always <= 0."""
        r = PressureReward()
        val = r.compute(single_intersection_engine, agent_node)
        assert val <= 0.0


class TestDelayReward:
    def test_returns_float(self, single_intersection_engine, agent_node):
        r = DelayReward()
        val = r.compute(single_intersection_engine, agent_node)
        assert isinstance(val, float)

    def test_is_non_positive(self, single_intersection_engine, agent_node):
        r = DelayReward()
        val = r.compute(single_intersection_engine, agent_node)
        assert val <= 0.0


class TestWaitingTimeReward:
    def test_returns_float(self, single_intersection_engine, agent_node):
        r = WaitingTimeReward()
        val = r.compute(single_intersection_engine, agent_node)
        assert isinstance(val, float)

    def test_is_non_positive(self, single_intersection_engine, agent_node):
        r = WaitingTimeReward()
        val = r.compute(single_intersection_engine, agent_node)
        assert val <= 0.0


class TestThroughputReward:
    def test_returns_float(self, single_intersection_engine, agent_node):
        r = ThroughputReward()
        val = r.compute(single_intersection_engine, agent_node)
        assert isinstance(val, float)

    def test_is_non_negative(self, single_intersection_engine, agent_node):
        """Throughput is a flow, always >= 0."""
        r = ThroughputReward()
        val = r.compute(single_intersection_engine, agent_node)
        assert val >= 0.0


class TestNormalizedThroughputReward:
    def test_returns_float(self, single_intersection_engine, agent_node):
        r = NormalizedThroughputReward()
        val = r.compute(single_intersection_engine, agent_node)
        assert isinstance(val, float)

    def test_in_unit_range(self, single_intersection_engine, agent_node):
        """Normalized throughput should be in [0, 1]."""
        r = NormalizedThroughputReward()
        val = r.compute(single_intersection_engine, agent_node)
        assert 0.0 <= val <= 1.0
