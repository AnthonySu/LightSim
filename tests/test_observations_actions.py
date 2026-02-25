"""Tests for observation builders and action handlers."""

from __future__ import annotations

import pytest
import numpy as np

from lightsim.envs.observations import (
    get_obs_builder,
    DefaultObservation,
    PressureObservation,
    FullDensityObservation,
)
from lightsim.envs.actions import (
    get_action_handler,
    PhaseSelectAction,
    NextOrStayAction,
)
from lightsim.core.engine import SimulationEngine
from lightsim.core.signal import RLController
from lightsim.core.types import NodeType
from lightsim.benchmarks.scenarios import get_scenario


@pytest.fixture
def engine_and_node():
    """Create an engine with RLController and return (engine, node_id)."""
    network, demand = get_scenario("single-intersection-v0")()
    ctrl = RLController()
    engine = SimulationEngine(
        network=network, dt=1.0,
        controller=ctrl,
        demand_profiles=demand,
    )
    engine.reset(seed=42)
    node_id = None
    for nid, node in network.nodes.items():
        if node.node_type == NodeType.SIGNALIZED:
            node_id = nid
            break
    return engine, node_id


# ---------------------------------------------------------------------------
# Observation registry
# ---------------------------------------------------------------------------


class TestObsRegistry:
    def test_get_known_builders(self):
        for name in ["default", "pressure", "full_density"]:
            b = get_obs_builder(name)
            assert b is not None

    def test_get_unknown_builder_raises(self):
        with pytest.raises(KeyError, match="Unknown observation builder"):
            get_obs_builder("nonexistent_obs")


# ---------------------------------------------------------------------------
# DefaultObservation
# ---------------------------------------------------------------------------


class TestDefaultObservation:
    def test_observation_space_shape(self, engine_and_node):
        engine, node_id = engine_and_node
        builder = DefaultObservation()
        space = builder.observation_space(engine, node_id)
        assert space.shape[0] > 0

    def test_observe_returns_array(self, engine_and_node):
        engine, node_id = engine_and_node
        builder = DefaultObservation()
        obs = builder.observe(engine, node_id)
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32

    def test_observe_within_bounds(self, engine_and_node):
        engine, node_id = engine_and_node
        builder = DefaultObservation()
        space = builder.observation_space(engine, node_id)
        # Run a few steps to have some density
        for _ in range(50):
            engine.step()
        obs = builder.observe(engine, node_id)
        assert obs.shape == space.shape
        assert np.all(obs >= space.low)
        assert np.all(obs <= space.high)

    def test_observe_shape_matches_space(self, engine_and_node):
        engine, node_id = engine_and_node
        builder = DefaultObservation()
        space = builder.observation_space(engine, node_id)
        obs = builder.observe(engine, node_id)
        assert obs.shape == space.shape

    def test_phase_one_hot_in_obs(self, engine_and_node):
        """First N elements should be a one-hot phase vector."""
        engine, node_id = engine_and_node
        builder = DefaultObservation()
        obs = builder.observe(engine, node_id)
        n_phases = engine.net.n_phases_per_node.get(node_id, 1)
        phase_vec = obs[:n_phases]
        # Exactly one element should be 1.0
        assert np.sum(phase_vec) == pytest.approx(1.0)
        assert np.max(phase_vec) == 1.0


# ---------------------------------------------------------------------------
# PressureObservation
# ---------------------------------------------------------------------------


class TestPressureObservation:
    def test_observation_space_bounds(self, engine_and_node):
        engine, node_id = engine_and_node
        builder = PressureObservation()
        space = builder.observation_space(engine, node_id)
        assert np.all(space.low == -1.0)
        assert np.all(space.high == 1.0)

    def test_observe_within_bounds(self, engine_and_node):
        engine, node_id = engine_and_node
        builder = PressureObservation()
        for _ in range(50):
            engine.step()
        obs = builder.observe(engine, node_id)
        assert np.all(obs >= -1.0)
        assert np.all(obs <= 1.0)


# ---------------------------------------------------------------------------
# FullDensityObservation
# ---------------------------------------------------------------------------


class TestFullDensityObservation:
    def test_includes_all_cells(self, engine_and_node):
        engine, node_id = engine_and_node
        builder = FullDensityObservation()
        space = builder.observation_space(engine, node_id)
        n_phases = engine.net.n_phases_per_node.get(node_id, 1)
        # Should include one-hot phase + all cells
        assert space.shape[0] == n_phases + engine.net.n_cells

    def test_observe_within_bounds(self, engine_and_node):
        engine, node_id = engine_and_node
        builder = FullDensityObservation()
        for _ in range(50):
            engine.step()
        obs = builder.observe(engine, node_id)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)


# ---------------------------------------------------------------------------
# Action registry
# ---------------------------------------------------------------------------


class TestActionRegistry:
    def test_get_known_handlers(self):
        for name in ["phase_select", "next_or_stay"]:
            h = get_action_handler(name)
            assert h is not None

    def test_get_unknown_handler_raises(self):
        with pytest.raises(KeyError, match="Unknown action handler"):
            get_action_handler("nonexistent_action")


# ---------------------------------------------------------------------------
# PhaseSelectAction
# ---------------------------------------------------------------------------


class TestPhaseSelectAction:
    def test_action_space_size(self, engine_and_node):
        engine, node_id = engine_and_node
        handler = PhaseSelectAction()
        space = handler.action_space(engine, node_id)
        n_phases = engine.net.n_phases_per_node.get(node_id, 1)
        assert space.n == n_phases

    def test_apply_changes_phase(self, engine_and_node):
        engine, node_id = engine_and_node
        handler = PhaseSelectAction()
        n_phases = engine.net.n_phases_per_node.get(node_id, 1)
        if n_phases < 2:
            pytest.skip("Need at least 2 phases")
        # Set to phase 1 and run enough steps to pass yellow+all_red
        handler.apply(1, engine, node_id)
        for _ in range(10):
            engine.step()
        state = engine.signal_manager.states[node_id]
        assert state.current_phase_idx == 1


# ---------------------------------------------------------------------------
# NextOrStayAction
# ---------------------------------------------------------------------------


class TestNextOrStayAction:
    def test_action_space_is_binary(self, engine_and_node):
        engine, node_id = engine_and_node
        handler = NextOrStayAction()
        space = handler.action_space(engine, node_id)
        assert space.n == 2

    def test_stay_keeps_phase(self, engine_and_node):
        engine, node_id = engine_and_node
        handler = NextOrStayAction()
        # Get current phase
        state = engine.signal_manager.states[node_id]
        current = state.current_phase_idx
        # Apply stay (action 0)
        handler.apply(0, engine, node_id)
        engine.step()
        state = engine.signal_manager.states[node_id]
        assert state.current_phase_idx == current

    def test_next_advances_phase(self, engine_and_node):
        engine, node_id = engine_and_node
        handler = NextOrStayAction()
        n_phases = engine.net.n_phases_per_node.get(node_id, 1)
        if n_phases < 2:
            pytest.skip("Need at least 2 phases")
        state = engine.signal_manager.states[node_id]
        current = state.current_phase_idx
        # Apply next (action 1) and run enough steps to pass yellow+all_red
        handler.apply(1, engine, node_id)
        for _ in range(10):
            engine.step()
        state = engine.signal_manager.states[node_id]
        expected = (current + 1) % n_phases
        assert state.current_phase_idx == expected
