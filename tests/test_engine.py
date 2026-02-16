"""Tests for the simulation engine: single intersection queue/discharge."""

import numpy as np
import pytest

from lightsim.core.demand import DemandProfile
from lightsim.core.engine import SimulationEngine
from lightsim.core.network import Network
from lightsim.core.signal import FixedTimeController
from lightsim.core.types import LinkID, NodeID, NodeType, TurnType


def _make_simple_intersection() -> tuple[Network, list[DemandProfile]]:
    """Two-approach intersection: NB and EB, with through movements."""
    net = Network()
    vf = 13.89
    dt = 1.0
    cell_length = vf * dt  # satisfy CFL

    # Nodes
    net.add_node(NodeID(0), NodeType.SIGNALIZED)
    net.add_node(NodeID(1), NodeType.ORIGIN)     # North origin
    net.add_node(NodeID(2), NodeType.DESTINATION) # South dest
    net.add_node(NodeID(3), NodeType.ORIGIN)     # East origin
    net.add_node(NodeID(4), NodeType.DESTINATION) # West dest

    kwargs = dict(
        length=cell_length * 3, lanes=1, n_cells=3,
        free_flow_speed=vf, wave_speed=5.56, jam_density=0.15, capacity=0.5,
    )

    # Links
    net.add_link(LinkID(0), NodeID(1), NodeID(0), **kwargs)  # NB in
    net.add_link(LinkID(1), NodeID(0), NodeID(2), **kwargs)  # SB out
    net.add_link(LinkID(2), NodeID(3), NodeID(0), **kwargs)  # EB in
    net.add_link(LinkID(3), NodeID(0), NodeID(4), **kwargs)  # WB out

    # Movements
    m_ns = net.add_movement(LinkID(0), LinkID(1), NodeID(0), TurnType.THROUGH, 1.0)
    m_ew = net.add_movement(LinkID(2), LinkID(3), NodeID(0), TurnType.THROUGH, 1.0)

    # Two phases
    net.add_phase(NodeID(0), [m_ns.movement_id], min_green=5.0)
    net.add_phase(NodeID(0), [m_ew.movement_id], min_green=5.0)

    demand = [
        DemandProfile(LinkID(0), [0.0], [0.3]),
        DemandProfile(LinkID(2), [0.0], [0.2]),
    ]

    return net, demand


class TestSimpleIntersection:
    """Test queue formation and discharge at a signalised intersection."""

    def test_queue_forms_during_red(self):
        """Vehicles should accumulate on a red approach."""
        net, demand = _make_simple_intersection()
        controller = FixedTimeController({NodeID(0): [30.0, 30.0]})
        engine = SimulationEngine(
            network=net, dt=1.0, controller=controller,
            demand_profiles=demand,
        )
        engine.reset(seed=42)

        # Run until phase 1 starts (EB gets red during phase 0)
        for _ in range(20):
            engine.step()

        # NB should be green (phase 0), EB should have some queue
        eb_vehicles = engine.get_link_vehicles(LinkID(2))
        assert eb_vehicles > 0, "EB approach should have vehicles during red"

    def test_vehicles_discharge_during_green(self):
        """Vehicles should discharge when their phase is green."""
        net, demand = _make_simple_intersection()
        controller = FixedTimeController({NodeID(0): [30.0, 30.0]})
        engine = SimulationEngine(
            network=net, dt=1.0, controller=controller,
            demand_profiles=demand,
        )
        engine.reset(seed=42)

        # Run for a full cycle
        for _ in range(100):
            engine.step()

        # Total exited should be positive
        assert engine.state.total_exited > 0, "No vehicles exited the network"

    def test_metrics_reasonable(self):
        """Network metrics should be non-negative and reasonable."""
        net, demand = _make_simple_intersection()
        engine = SimulationEngine(
            network=net, dt=1.0, demand_profiles=demand,
        )
        engine.reset(seed=42)

        for _ in range(100):
            engine.step()

        metrics = engine.get_network_metrics()
        assert metrics["time"] == 100.0
        assert metrics["total_vehicles"] >= 0
        assert metrics["total_entered"] >= 0
        assert metrics["total_exited"] >= 0
        assert metrics["avg_density"] >= 0


class TestSeedReproducibility:
    """Test that same seed produces same trajectory."""

    def test_deterministic(self):
        """Two runs with the same seed should produce identical results."""
        net, demand = _make_simple_intersection()

        results = []
        for _ in range(2):
            engine = SimulationEngine(
                network=net, dt=1.0, demand_profiles=demand,
            )
            engine.reset(seed=42)
            for _ in range(50):
                engine.step()
            results.append(engine.state.density.copy())

        np.testing.assert_array_equal(results[0], results[1])
