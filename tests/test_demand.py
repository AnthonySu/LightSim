"""Tests for DemandProfile and DemandManager."""

from __future__ import annotations

import numpy as np
import pytest

from lightsim.core.demand import DemandProfile, DemandManager
from lightsim.core.engine import SimulationEngine
from lightsim.core.signal import FixedTimeController
from lightsim.core.types import LinkID
from lightsim.benchmarks.scenarios import get_scenario


class TestDemandProfile:
    def test_constant_rate(self):
        p = DemandProfile(link_id=LinkID(0), time_points=[0.0], flow_rates=[0.5])
        assert p.get_rate(0.0) == pytest.approx(0.5)
        assert p.get_rate(100.0) == pytest.approx(0.5)
        assert p.get_rate(9999.0) == pytest.approx(0.5)

    def test_two_intervals(self):
        p = DemandProfile(
            link_id=LinkID(0),
            time_points=[0.0, 100.0],
            flow_rates=[0.3, 0.7],
        )
        assert p.get_rate(0.0) == pytest.approx(0.3)
        assert p.get_rate(50.0) == pytest.approx(0.3)
        assert p.get_rate(99.9) == pytest.approx(0.3)
        assert p.get_rate(100.0) == pytest.approx(0.7)
        assert p.get_rate(200.0) == pytest.approx(0.7)

    def test_three_intervals(self):
        p = DemandProfile(
            link_id=LinkID(0),
            time_points=[0.0, 60.0, 120.0],
            flow_rates=[0.2, 0.5, 0.1],
        )
        assert p.get_rate(30.0) == pytest.approx(0.2)
        assert p.get_rate(90.0) == pytest.approx(0.5)
        assert p.get_rate(150.0) == pytest.approx(0.1)

    def test_boundary_time_point(self):
        """At exact breakpoint, should use the new interval's rate."""
        p = DemandProfile(
            link_id=LinkID(0),
            time_points=[0.0, 50.0],
            flow_rates=[1.0, 2.0],
        )
        assert p.get_rate(50.0) == pytest.approx(2.0)

    def test_time_zero(self):
        p = DemandProfile(
            link_id=LinkID(0),
            time_points=[0.0, 100.0],
            flow_rates=[0.4, 0.8],
        )
        assert p.get_rate(0.0) == pytest.approx(0.4)

    def test_default_profile_is_zero(self):
        p = DemandProfile(link_id=LinkID(0))
        assert p.get_rate(0.0) == pytest.approx(0.0)
        assert p.get_rate(100.0) == pytest.approx(0.0)


class TestDemandManager:
    def test_injection_shape(self):
        """Injection array should have n_cells elements."""
        network, demand = get_scenario("single-intersection-v0")()
        engine = SimulationEngine(
            network=network, dt=1.0,
            controller=FixedTimeController(),
            demand_profiles=demand,
        )
        engine.reset()
        dm = engine.demand_manager
        density = engine.state.density
        inj = dm.get_injection(t=0.0, dt=1.0, density=density)
        assert inj.shape == (engine.net.n_cells,)

    def test_no_demand_no_injection(self):
        """With no demand profiles, injection should be all zeros."""
        network, _ = get_scenario("single-intersection-v0")()
        engine = SimulationEngine(
            network=network, dt=1.0,
            controller=FixedTimeController(),
            demand_profiles=[],
        )
        engine.reset()
        dm = engine.demand_manager
        density = engine.state.density
        inj = dm.get_injection(t=0.0, dt=1.0, density=density)
        assert np.all(inj == 0.0)

    def test_injection_is_non_negative(self):
        """Injection should never be negative."""
        network, demand = get_scenario("single-intersection-v0")()
        engine = SimulationEngine(
            network=network, dt=1.0,
            controller=FixedTimeController(),
            demand_profiles=demand,
        )
        engine.reset()
        dm = engine.demand_manager
        for step in range(100):
            density = engine.state.density
            inj = dm.get_injection(t=step * 1.0, dt=1.0, density=density)
            assert np.all(inj >= 0.0)
            engine.step()

    def test_injection_respects_capacity(self):
        """Injection should not exceed cell capacity."""
        network, demand = get_scenario("single-intersection-v0")()
        engine = SimulationEngine(
            network=network, dt=1.0,
            controller=FixedTimeController(),
            demand_profiles=demand,
        )
        engine.reset()
        dm = engine.demand_manager
        # Fill network to near jam density
        engine.state.density[:] = engine.net.kj * 0.99
        density = engine.state.density
        inj = dm.get_injection(t=0.0, dt=1.0, density=density)
        # Injection should be very small or zero since cells are nearly full
        assert np.all(inj < 1.0)  # less than 1 vehicle per cell
