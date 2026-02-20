"""Tests for traffic metrics: occupancy, spillback, queue length, delay."""

import numpy as np
import pytest

from lightsim.core.demand import DemandProfile
from lightsim.core.engine import SimulationEngine
from lightsim.core.network import Network
from lightsim.core.signal import FixedTimeController
from lightsim.core.types import LinkID, NodeID, NodeType, TurnType
from lightsim.utils.metrics import (
    compute_link_delay,
    compute_link_occupancy,
    compute_link_queue_length,
    compute_mfd,
    compute_movement_counts,
    compute_network_delay,
    compute_network_occupancy,
    compute_pressure,
    detect_spillback,
)


def _make_intersection():
    """Two-approach signalized intersection for metric tests."""
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

    demand = [
        DemandProfile(LinkID(0), [0.0], [0.3]),
        DemandProfile(LinkID(2), [0.0], [0.2]),
    ]
    return net, demand


class TestOccupancy:
    """Test occupancy metric."""

    def test_empty_network_zero_occupancy(self):
        """Occupancy should be 0 when network is empty."""
        net, _ = _make_intersection()
        engine = SimulationEngine(network=net, dt=1.0, demand_profiles=[])
        engine.reset(seed=42)
        assert compute_link_occupancy(engine, LinkID(0)) == 0.0
        assert compute_network_occupancy(engine) == 0.0

    def test_occupancy_increases_with_demand(self):
        """Occupancy should increase as vehicles enter."""
        net, demand = _make_intersection()
        controller = FixedTimeController({NodeID(0): [30.0, 30.0]})
        engine = SimulationEngine(
            network=net, dt=1.0, controller=controller, demand_profiles=demand,
        )
        engine.reset(seed=42)

        occ_start = compute_network_occupancy(engine)
        for _ in range(100):
            engine.step()
        occ_end = compute_network_occupancy(engine)

        assert occ_end > occ_start

    def test_occupancy_bounded(self):
        """Occupancy should be in [0, 1]."""
        net, demand = _make_intersection()
        engine = SimulationEngine(network=net, dt=1.0, demand_profiles=demand)
        engine.reset(seed=42)
        for _ in range(500):
            engine.step()
            occ = compute_network_occupancy(engine)
            assert 0.0 <= occ <= 1.0, f"Occupancy {occ} out of [0,1]"


class TestSpillback:
    """Test spillback detection."""

    def test_no_spillback_empty(self):
        """No spillback in empty network."""
        net, _ = _make_intersection()
        engine = SimulationEngine(network=net, dt=1.0, demand_profiles=[])
        engine.reset(seed=42)
        assert not detect_spillback(engine, LinkID(0))

    def test_spillback_under_heavy_demand(self):
        """Heavy demand with red signal should cause spillback."""
        net, _ = _make_intersection()
        heavy_demand = [
            DemandProfile(LinkID(0), [0.0], [1.5]),
            DemandProfile(LinkID(2), [0.0], [1.5]),
        ]
        controller = FixedTimeController({NodeID(0): [60.0, 60.0]})
        engine = SimulationEngine(
            network=net, dt=1.0, controller=controller, demand_profiles=heavy_demand,
        )
        engine.reset(seed=42)
        # Run long enough for EB to fill during its red phase
        for _ in range(200):
            engine.step()
        # At least one link should have spillback
        any_spillback = (
            detect_spillback(engine, LinkID(0)) or
            detect_spillback(engine, LinkID(2))
        )
        assert any_spillback, "Expected spillback under heavy demand"


class TestQueueLength:
    """Test queue length metric."""

    def test_zero_queue_empty(self):
        """No queue in empty network."""
        net, _ = _make_intersection()
        engine = SimulationEngine(network=net, dt=1.0, demand_profiles=[])
        engine.reset(seed=42)
        assert compute_link_queue_length(engine, LinkID(0)) == 0.0

    def test_queue_non_negative(self):
        """Queue length should always be non-negative."""
        net, demand = _make_intersection()
        engine = SimulationEngine(network=net, dt=1.0, demand_profiles=demand)
        engine.reset(seed=42)
        for _ in range(200):
            engine.step()
            for lid in [LinkID(0), LinkID(1), LinkID(2), LinkID(3)]:
                assert compute_link_queue_length(engine, lid) >= 0.0


class TestNetworkDelay:
    """Test network delay metric."""

    def test_zero_delay_empty(self):
        """No delay when network is empty."""
        net, _ = _make_intersection()
        engine = SimulationEngine(network=net, dt=1.0, demand_profiles=[])
        engine.reset(seed=42)
        assert compute_network_delay(engine) == 0.0

    def test_delay_non_negative(self):
        """Delay should be non-negative."""
        net, demand = _make_intersection()
        engine = SimulationEngine(network=net, dt=1.0, demand_profiles=demand)
        engine.reset(seed=42)
        for _ in range(200):
            engine.step()
            assert compute_network_delay(engine) >= 0.0


class TestMovementCounts:
    """Test turning movement counts."""

    def test_counts_zero_when_empty(self):
        """Movement counts should be zero when network is empty."""
        net, _ = _make_intersection()
        controller = FixedTimeController({NodeID(0): [30.0, 30.0]})
        engine = SimulationEngine(
            network=net, dt=1.0, controller=controller, demand_profiles=[],
        )
        engine.reset(seed=42)
        counts = compute_movement_counts(engine, NodeID(0))
        for mid, flow in counts.items():
            assert flow == 0.0

    def test_counts_non_negative(self):
        """Movement counts should be non-negative."""
        net, demand = _make_intersection()
        controller = FixedTimeController({NodeID(0): [30.0, 30.0]})
        engine = SimulationEngine(
            network=net, dt=1.0, controller=controller, demand_profiles=demand,
        )
        engine.reset(seed=42)
        for _ in range(100):
            engine.step()
            counts = compute_movement_counts(engine, NodeID(0))
            for mid, flow in counts.items():
                assert flow >= 0.0


class TestMFD:
    """Test MFD computation."""

    def test_mfd_empty(self):
        """MFD at zero demand: (0, 0)."""
        net, _ = _make_intersection()
        engine = SimulationEngine(network=net, dt=1.0, demand_profiles=[])
        engine.reset(seed=42)
        avg_k, avg_q = compute_mfd(engine)
        assert avg_k == 0.0
        assert avg_q == 0.0

    def test_mfd_positive_with_demand(self):
        """MFD should show positive density and flow with demand."""
        net, demand = _make_intersection()
        engine = SimulationEngine(network=net, dt=1.0, demand_profiles=demand)
        engine.reset(seed=42)
        for _ in range(100):
            engine.step()
        avg_k, avg_q = compute_mfd(engine)
        assert avg_k > 0
        assert avg_q > 0
