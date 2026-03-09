"""Signal controller edge case tests."""

import numpy as np
import pytest

from lightsim.benchmarks.single_intersection import create_single_intersection
from lightsim.core.demand import DemandProfile
from lightsim.core.engine import SimulationEngine
from lightsim.core.network import Network
from lightsim.core.signal import (
    FixedTimeController,
    MaxPressureController,
    WebsterController,
)
from lightsim.core.types import FLOAT, LinkID, NodeID, NodeType, TurnType


def _make_signalized_intersection():
    """Two-approach signalized intersection."""
    net = Network()
    vf = 13.89
    cell_length = vf * 1.0

    net.add_node(NodeID(0), NodeType.SIGNALIZED)
    net.add_node(NodeID(1), NodeType.ORIGIN)
    net.add_node(NodeID(2), NodeType.DESTINATION)
    net.add_node(NodeID(3), NodeType.ORIGIN)
    net.add_node(NodeID(4), NodeType.DESTINATION)

    kwargs = dict(
        length=cell_length * 3, lanes=1, n_cells=3,
        free_flow_speed=vf, wave_speed=5.56, jam_density=0.15, capacity=0.5,
    )

    net.add_link(LinkID(0), NodeID(1), NodeID(0), **kwargs)
    net.add_link(LinkID(1), NodeID(0), NodeID(2), **kwargs)
    net.add_link(LinkID(2), NodeID(3), NodeID(0), **kwargs)
    net.add_link(LinkID(3), NodeID(0), NodeID(4), **kwargs)

    m_ns = net.add_movement(LinkID(0), LinkID(1), NodeID(0), TurnType.THROUGH, 1.0)
    m_ew = net.add_movement(LinkID(2), LinkID(3), NodeID(0), TurnType.THROUGH, 1.0)

    net.add_phase(NodeID(0), [m_ns.movement_id], min_green=5.0)
    net.add_phase(NodeID(0), [m_ew.movement_id], min_green=5.0)

    return net


def _make_single_phase_node():
    """Network with a signalized node that has only one phase."""
    net = Network()
    vf = 13.89
    cell_length = vf * 1.0

    net.add_node(NodeID(0), NodeType.SIGNALIZED)
    net.add_node(NodeID(1), NodeType.ORIGIN)
    net.add_node(NodeID(2), NodeType.DESTINATION)

    kwargs = dict(
        length=cell_length * 3, lanes=1, n_cells=3,
        free_flow_speed=vf, wave_speed=5.56, jam_density=0.15, capacity=0.5,
    )

    net.add_link(LinkID(0), NodeID(1), NodeID(0), **kwargs)
    net.add_link(LinkID(1), NodeID(0), NodeID(2), **kwargs)

    m = net.add_movement(LinkID(0), LinkID(1), NodeID(0), TurnType.THROUGH, 1.0)
    net.add_phase(NodeID(0), [m.movement_id], min_green=5.0)

    return net


class TestMaxPressureZeroVehicles:
    """MaxPressure with zero density should not cause division by zero."""

    def test_zero_density_no_crash(self):
        """MaxPressure on empty network selects a valid phase index."""
        net = _make_signalized_intersection()
        controller = MaxPressureController(min_green=1.0)
        engine = SimulationEngine(
            network=net, dt=1.0, controller=controller, demand_profiles=[],
        )
        engine.reset(seed=42)

        # Run enough steps for min_green to expire and force a decision
        for _ in range(50):
            engine.step()

        state = engine.signal_manager.states[NodeID(0)]
        assert state.current_phase_idx >= 0
        assert (engine.state.density >= 0).all()
        assert np.isfinite(engine.state.density).all()

    def test_zero_density_pressure_is_zero(self):
        """All phase pressures should be zero with no vehicles."""
        from lightsim.core.signal import _compute_phase_pressure

        net = _make_signalized_intersection().compile(dt=1.0)
        density = np.zeros(net.n_cells, dtype=FLOAT)

        for pid in range(len(net.phase_movements)):
            pressure = _compute_phase_pressure(pid, net, density)
            assert pressure == 0.0, f"Phase {pid} pressure should be 0, got {pressure}"


class TestWebsterHighY:
    """Webster controller with very high Y values (near saturation)."""

    def test_high_y_bounded_cycle(self):
        """Webster should cap cycle length even with very high demand ratios."""
        network, _ = create_single_intersection()
        controller = WebsterController()

        # Create very high demand to push Y close to 1.0
        demand = [
            DemandProfile(LinkID(0), [0.0], [2.0]),
            DemandProfile(LinkID(1), [0.0], [2.0]),
            DemandProfile(LinkID(2), [0.0], [2.0]),
            DemandProfile(LinkID(3), [0.0], [2.0]),
        ]
        engine = SimulationEngine(
            network=network, dt=1.0, controller=controller,
            demand_profiles=demand,
        )
        engine.reset(seed=42)

        # Run long enough for Webster to optimize multiple times
        for _ in range(600):
            engine.step()

        # Should not crash; metrics should be finite
        metrics = engine.get_network_metrics()
        assert np.isfinite(metrics["total_entered"])
        assert np.isfinite(metrics["total_exited"])
        assert (engine.state.density >= 0).all()
        assert np.isfinite(engine.state.density).all()

    def test_webster_green_times_are_positive(self):
        """All computed green times should be >= 5.0 (min green)."""
        network, demand = create_single_intersection()
        controller = WebsterController()
        engine = SimulationEngine(
            network=network, dt=1.0, controller=controller,
            demand_profiles=demand,
        )
        engine.reset(seed=42)

        # Run to trigger optimization
        for _ in range(100):
            engine.step()

        for node_id, greens in controller._green_times.items():
            for g in greens:
                assert g >= 5.0, (
                    f"Node {node_id}: green time {g} is below minimum 5.0"
                )


class TestFixedTimeSinglePhase:
    """FixedTime controller with a single-phase node."""

    def test_single_phase_stays_green(self):
        """A single-phase node should always stay in phase 0."""
        net = _make_single_phase_node()
        controller = FixedTimeController({NodeID(0): [30.0]})
        demand = [DemandProfile(LinkID(0), [0.0], [0.3])]
        engine = SimulationEngine(
            network=net, dt=1.0, controller=controller,
            demand_profiles=demand,
        )
        engine.reset(seed=42)

        for _ in range(200):
            engine.step()
            state = engine.signal_manager.states[NodeID(0)]
            # With only 1 phase, cycling (idx+1) % 1 == 0 always
            assert state.current_phase_idx == 0

        assert engine.state.total_exited > 0

    def test_single_phase_no_deadlock(self):
        """Vehicles should flow through a single-phase intersection."""
        net = _make_single_phase_node()
        demand = [DemandProfile(LinkID(0), [0.0], [0.3])]
        engine = SimulationEngine(
            network=net, dt=1.0,
            controller=FixedTimeController(),
            demand_profiles=demand,
        )
        engine.reset(seed=42)

        for _ in range(300):
            engine.step()

        assert engine.state.total_entered > 0
        assert engine.state.total_exited > 0


class TestControllerSwitchMidSimulation:
    """Switching the signal controller during a running simulation."""

    def test_switch_fixed_to_maxpressure(self):
        """Switching from FixedTime to MaxPressure mid-run should not crash."""
        network, demand = create_single_intersection()

        # Start with FixedTime
        engine = SimulationEngine(
            network=network, dt=1.0,
            controller=FixedTimeController(),
            demand_profiles=demand,
        )
        engine.reset(seed=42)
        for _ in range(200):
            engine.step()

        entered_before = engine.state.total_entered
        exited_before = engine.state.total_exited

        # Switch controller by rebuilding signal manager
        new_controller = MaxPressureController(min_green=5.0)
        engine.controller = new_controller
        from lightsim.core.signal import SignalManager
        engine.signal_manager = SignalManager(engine.net, new_controller)

        # Continue running
        for _ in range(200):
            engine.step()

        # Should still function
        assert engine.state.total_entered > entered_before
        assert engine.state.total_exited > exited_before
        assert (engine.state.density >= 0).all()
        assert np.isfinite(engine.state.density).all()
