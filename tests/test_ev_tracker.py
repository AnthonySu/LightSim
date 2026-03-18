"""Tests for the EVTracker emergency vehicle overlay."""

from __future__ import annotations

import numpy as np
import pytest

from lightsim.core import (
    SimulationEngine,
    EVTracker,
    EVState,
    FixedTimeController,
    MaxPressureController,
)
from lightsim.networks.grid import create_grid_network
from lightsim.networks.arterial import create_arterial_network


def _make_engine(rows=2, cols=2, dt=5.0, controller=None):
    """Create a small grid engine for testing."""
    net = create_grid_network(rows, cols)
    ctrl = controller or FixedTimeController()
    engine = SimulationEngine(net, dt=dt, controller=ctrl)
    engine.reset(seed=42)
    return engine


def _get_route(engine, start_link=None):
    """Get a simple route through the grid (first N links of any path)."""
    net = engine.net
    link_ids = sorted(net.link_cells.keys())
    if start_link is not None and start_link in link_ids:
        idx = link_ids.index(start_link)
        return link_ids[idx:idx + 3] if idx + 3 <= len(link_ids) else link_ids[idx:]
    # Just grab a few connected links
    return link_ids[:min(4, len(link_ids))]


class TestEVTrackerBasic:
    """Basic EVTracker functionality."""

    def test_init(self):
        engine = _make_engine()
        route = _get_route(engine)
        ev = EVTracker(engine, route)
        assert not ev.arrived
        assert ev.state.link_idx == 0
        assert ev.state.progress == 0.0
        assert ev.state.stops == 0

    def test_empty_route_raises(self):
        engine = _make_engine()
        with pytest.raises(ValueError, match="at least one link"):
            EVTracker(engine, [])

    def test_reset(self):
        engine = _make_engine()
        route = _get_route(engine)
        ev = EVTracker(engine, route)

        # Run a few steps
        for _ in range(5):
            engine.step()
            ev.step()

        assert ev.state.travel_time > 0

        # Reset should clear state
        ev.reset()
        assert ev.state.travel_time == 0.0
        assert ev.state.link_idx == 0
        assert ev.state.progress == 0.0
        assert not ev.arrived

    def test_ev_advances(self):
        engine = _make_engine()
        route = _get_route(engine)
        ev = EVTracker(engine, route, speed_factor=2.0)

        initial_progress = ev.state.progress
        for _ in range(10):
            engine.step()
            ev.step()

        # EV should have moved forward
        assert ev.state.travel_time > 0
        assert ev.state.distance_traveled > 0
        assert ev.state.link_idx > 0 or ev.state.progress > initial_progress

    def test_ev_eventually_arrives(self):
        """EV should reach destination within a reasonable number of steps."""
        engine = _make_engine(rows=2, cols=2, dt=5.0)
        route = _get_route(engine)
        ev = EVTracker(engine, route, speed_factor=3.0)

        for _ in range(200):
            engine.step()
            ev.step()
            if ev.arrived:
                break

        assert ev.arrived, f"EV did not arrive after 200 steps (link_idx={ev.state.link_idx}/{len(route)})"

    def test_speed_factor(self):
        """Higher speed factor should lead to faster arrival."""
        engine_slow = _make_engine(dt=5.0)
        engine_fast = _make_engine(dt=5.0)
        route_slow = _get_route(engine_slow)
        route_fast = _get_route(engine_fast)

        ev_slow = EVTracker(engine_slow, route_slow, speed_factor=1.0)
        ev_fast = EVTracker(engine_fast, route_fast, speed_factor=3.0)

        for _ in range(100):
            engine_slow.step()
            ev_slow.step()
            engine_fast.step()
            ev_fast.step()

        # Faster EV should arrive sooner (less travel time for same route)
        if ev_fast.arrived and ev_slow.arrived:
            assert ev_fast.state.travel_time <= ev_slow.state.travel_time
        else:
            # At least the fast one should have more progress
            assert ev_fast.fraction_completed >= ev_slow.fraction_completed - 1e-9


class TestEVTrackerObservation:
    """Test EV observation generation."""

    def test_get_ev_observation(self):
        engine = _make_engine()
        route = _get_route(engine)
        ev = EVTracker(engine, route)

        obs = ev.get_ev_observation()
        assert "link_idx" in obs
        assert "progress" in obs
        assert "speed" in obs
        assert "travel_time" in obs
        assert "distance_traveled" in obs
        assert "arrived" in obs
        assert "stops" in obs
        assert "fraction_completed" in obs

    def test_fraction_completed_increases(self):
        engine = _make_engine()
        route = _get_route(engine)
        ev = EVTracker(engine, route, speed_factor=2.0)

        fractions = []
        for _ in range(20):
            engine.step()
            ev.step()
            fractions.append(ev.fraction_completed)

        # Should be non-decreasing
        for i in range(1, len(fractions)):
            assert fractions[i] >= fractions[i - 1] - 1e-9


class TestEVTrackerProperties:
    """Test properties and edge cases."""

    def test_position_property(self):
        engine = _make_engine()
        route = _get_route(engine)
        ev = EVTracker(engine, route)

        link_id, progress = ev.position
        assert link_id == route[0]
        assert progress == 0.0

    def test_current_link_property(self):
        engine = _make_engine()
        route = _get_route(engine)
        ev = EVTracker(engine, route)
        assert ev.current_link == route[0]

    def test_ev_state_dataclass(self):
        state = EVState()
        assert state.link_idx == 0
        assert state.progress == 0.0
        assert not state.arrived
        assert state.stops == 0


class TestEVTrackerWithMaxPressure:
    """Test EV with MaxPressure controller (adaptive signals)."""

    def test_ev_with_maxpressure(self):
        engine = _make_engine(controller=MaxPressureController())
        route = _get_route(engine)
        ev = EVTracker(engine, route, speed_factor=1.5)

        for _ in range(50):
            engine.step()
            ev.step()

        # Should still work with adaptive signals
        assert ev.state.travel_time > 0


class TestEVTrackerArterial:
    """Test EV on arterial network."""

    def test_ev_on_arterial(self):
        net = create_arterial_network(n_intersections=4)
        engine = SimulationEngine(net, dt=5.0, controller=FixedTimeController())
        engine.reset(seed=42)

        route = sorted(engine.net.link_cells.keys())[:4]
        ev = EVTracker(engine, route, speed_factor=1.5)

        for _ in range(50):
            engine.step()
            ev.step()

        assert ev.state.travel_time > 0
        assert ev.state.distance_traveled > 0
