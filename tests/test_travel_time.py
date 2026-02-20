"""Tests for travel time estimation and tracking."""

import numpy as np
import pytest

from lightsim.core.demand import DemandProfile
from lightsim.core.engine import SimulationEngine
from lightsim.core.network import Network
from lightsim.core.types import LinkID, NodeID, NodeType
from lightsim.utils.travel_time import (
    TravelTimeTracker,
    estimate_link_free_flow_tt,
    estimate_link_travel_time,
)


def _make_single_link(n_cells=5, cell_length=20.0):
    net = Network()
    net.add_node(NodeID(0), NodeType.ORIGIN)
    net.add_node(NodeID(1), NodeType.DESTINATION)
    net.add_link(
        LinkID(0), NodeID(0), NodeID(1),
        length=cell_length * n_cells, lanes=1, n_cells=n_cells,
        free_flow_speed=13.89, wave_speed=5.56, jam_density=0.15, capacity=0.5,
    )
    return net


class TestEstimateTravelTime:
    """Test point-in-time travel time estimation."""

    def test_free_flow_travel_time(self):
        """Empty link: travel time should equal free-flow TT."""
        net = _make_single_link()
        engine = SimulationEngine(network=net, dt=1.0, demand_profiles=[])
        engine.reset(seed=42)

        tt = estimate_link_travel_time(engine, LinkID(0))
        ff = estimate_link_free_flow_tt(engine, LinkID(0))
        assert abs(tt - ff) < 1e-6, f"Expected free-flow TT, got {tt:.3f} vs {ff:.3f}"

    def test_congested_travel_time_higher(self):
        """Link with demand should have higher TT than free-flow."""
        net = _make_single_link()
        demand = [DemandProfile(LinkID(0), [0.0], [0.4])]
        engine = SimulationEngine(network=net, dt=1.0, demand_profiles=demand)
        engine.reset(seed=42)

        for _ in range(200):
            engine.step()

        tt = estimate_link_travel_time(engine, LinkID(0))
        ff = estimate_link_free_flow_tt(engine, LinkID(0))
        assert tt >= ff - 1e-6, f"Congested TT {tt:.3f} < free-flow {ff:.3f}"

    def test_travel_time_positive(self):
        """Travel time should always be positive."""
        net = _make_single_link()
        demand = [DemandProfile(LinkID(0), [0.0], [0.3])]
        engine = SimulationEngine(network=net, dt=1.0, demand_profiles=demand)
        engine.reset(seed=42)
        for _ in range(100):
            engine.step()
            tt = estimate_link_travel_time(engine, LinkID(0))
            assert tt > 0, "Travel time should be positive"


class TestTravelTimeTracker:
    """Test cumulative travel time tracking."""

    def test_tracker_records_history(self):
        """Tracker should accumulate samples."""
        net = _make_single_link()
        demand = [DemandProfile(LinkID(0), [0.0], [0.3])]
        engine = SimulationEngine(network=net, dt=1.0, demand_profiles=demand)
        engine.reset(seed=42)

        tracker = TravelTimeTracker(link_ids=[LinkID(0)])
        for _ in range(50):
            engine.step()
            tracker.update(engine)

        mean_tt = tracker.get_mean_travel_time(LinkID(0))
        assert mean_tt > 0

    def test_tracker_window(self):
        """Rolling window should limit history."""
        net = _make_single_link()
        engine = SimulationEngine(network=net, dt=1.0, demand_profiles=[])
        engine.reset(seed=42)

        tracker = TravelTimeTracker(link_ids=[LinkID(0)], window=10)
        for _ in range(50):
            engine.step()
            tracker.update(engine)

        # History should be at most 10 entries
        assert len(tracker._history[LinkID(0)]) == 10

    def test_tracker_reset(self):
        """reset() should clear all history."""
        net = _make_single_link()
        engine = SimulationEngine(network=net, dt=1.0, demand_profiles=[])
        engine.reset(seed=42)

        tracker = TravelTimeTracker()
        for _ in range(10):
            engine.step()
            tracker.update(engine)

        tracker.reset()
        assert tracker._history == {}
        assert tracker._count == 0

    def test_travel_time_index_free_flow(self):
        """TTI should be ~1.0 under free flow."""
        net = _make_single_link()
        engine = SimulationEngine(network=net, dt=1.0, demand_profiles=[])
        engine.reset(seed=42)

        tracker = TravelTimeTracker(link_ids=[LinkID(0)])
        engine.step()
        tracker.update(engine)

        tti = tracker.get_travel_time_index(engine, LinkID(0))
        assert abs(tti - 1.0) < 0.01, f"TTI should be ~1.0, got {tti:.3f}"

    def test_network_mean_travel_time(self):
        """Network mean TT should be positive with demand."""
        net = _make_single_link()
        demand = [DemandProfile(LinkID(0), [0.0], [0.3])]
        engine = SimulationEngine(network=net, dt=1.0, demand_profiles=demand)
        engine.reset(seed=42)

        tracker = TravelTimeTracker()
        for _ in range(100):
            engine.step()
            tracker.update(engine)

        net_tt = tracker.get_network_mean_travel_time(engine)
        assert net_tt > 0
