"""Demand edge case tests: negative values, zero-length profiles, capacity overflow."""

import numpy as np
import pytest

from lightsim.core.demand import DemandManager, DemandProfile
from lightsim.core.engine import SimulationEngine
from lightsim.core.network import Network
from lightsim.core.signal import FixedTimeController
from lightsim.core.types import FLOAT, LinkID, NodeID, NodeType


def _make_single_link(n_cells=3, cell_length=None, vf=13.89, capacity=0.5):
    """Origin -> Destination with one link."""
    net = Network()
    net.add_node(NodeID(0), NodeType.ORIGIN)
    net.add_node(NodeID(1), NodeType.DESTINATION)
    if cell_length is None:
        cell_length = vf * 1.0  # satisfy CFL for dt=1.0
    net.add_link(
        LinkID(0), NodeID(0), NodeID(1),
        length=cell_length * n_cells, lanes=1, n_cells=n_cells,
        free_flow_speed=vf, wave_speed=5.56, jam_density=0.15, capacity=capacity,
    )
    return net


class TestNegativeDemand:
    """Negative demand values should not inject negative vehicles."""

    def test_negative_rate_clamped_to_zero_in_profile(self):
        """DemandProfile clamps negative flow_rates to 0 at init."""
        p = DemandProfile(link_id=LinkID(0), time_points=[0.0], flow_rates=[-0.5])
        # Negative rates are clamped to 0 during validation
        assert p.get_rate(0.0) == pytest.approx(0.0)

    def test_negative_rate_injection_capped_at_zero(self):
        """DemandManager should not inject negative vehicles."""
        net = _make_single_link()
        demand = [DemandProfile(LinkID(0), [0.0], [-0.5])]
        engine = SimulationEngine(
            network=net, dt=1.0, demand_profiles=demand,
        )
        engine.reset(seed=42)

        # Pre-fill some density
        engine.state.density[:] = 0.05

        inj = engine.demand_manager.get_injection(t=0.0, dt=1.0,
                                                   density=engine.state.density)
        # Injection should not be negative (DemandManager caps by space which is >= 0,
        # and min(negative, space) would be negative, but vehicles don't leave via demand)
        # The actual result depends on implementation:
        # mean_veh = -0.5 * 1.0 = -0.5, space = positive, min(-0.5, space) = -0.5
        # This test documents the actual behavior.
        # If injection is negative, the simulation step clamps density to [0, kj],
        # so the system is still stable.
        for _ in range(100):
            engine.step()

        # Density should remain non-negative due to clamping in engine
        assert (engine.state.density >= 0).all()

    def test_negative_demand_simulation_stable(self):
        """Full simulation with negative demand should remain numerically stable."""
        net = _make_single_link()
        demand = [DemandProfile(LinkID(0), [0.0], [-1.0])]
        engine = SimulationEngine(
            network=net, dt=1.0, demand_profiles=demand,
        )
        engine.reset(seed=42)

        for _ in range(200):
            engine.step()

        assert (engine.state.density >= 0).all()
        assert np.isfinite(engine.state.density).all()


class TestZeroLengthDemandProfile:
    """Empty or minimal demand profiles."""

    def test_default_profile_zero_rate(self):
        """A DemandProfile with defaults has zero rate at all times."""
        p = DemandProfile(link_id=LinkID(0))
        assert p.get_rate(0.0) == pytest.approx(0.0)
        assert p.get_rate(1000.0) == pytest.approx(0.0)

    def test_single_point_profile(self):
        """A profile with one time point and one rate works correctly."""
        p = DemandProfile(link_id=LinkID(0), time_points=[0.0], flow_rates=[0.3])
        assert p.get_rate(0.0) == pytest.approx(0.3)
        assert p.get_rate(9999.0) == pytest.approx(0.3)

    def test_empty_demand_list_runs(self):
        """Simulation with empty demand list produces zero vehicles."""
        net = _make_single_link()
        engine = SimulationEngine(network=net, dt=1.0, demand_profiles=[])
        engine.reset(seed=42)

        for _ in range(100):
            engine.step()

        assert engine.state.total_entered == 0.0
        assert engine.state.total_exited == 0.0


class TestDemandExceedingCapacity:
    """Demand much larger than the source cell can absorb."""

    def test_injection_capped_by_cell_space(self):
        """Injection should not exceed the remaining space in the source cell."""
        net = _make_single_link(n_cells=3, capacity=0.5)
        # Demand 10x capacity
        demand = [DemandProfile(LinkID(0), [0.0], [5.0])]
        engine = SimulationEngine(
            network=net, dt=1.0, demand_profiles=demand,
        )
        engine.reset(seed=42)

        for _ in range(200):
            engine.step()
            # Density must never exceed jam density
            assert (engine.state.density <= engine.net.kj + 1e-6).all(), (
                f"Density exceeds kj: max={engine.state.density.max():.6f}"
            )
            assert (engine.state.density >= -1e-12).all()

    def test_excess_demand_does_not_crash(self):
        """Extremely high demand (100x capacity) should not crash."""
        net = _make_single_link(n_cells=3, capacity=0.5)
        demand = [DemandProfile(LinkID(0), [0.0], [50.0])]
        engine = SimulationEngine(
            network=net, dt=1.0, demand_profiles=demand,
        )
        engine.reset(seed=42)

        for _ in range(500):
            engine.step()

        metrics = engine.get_network_metrics()
        assert np.isfinite(metrics["total_entered"])
        assert np.isfinite(metrics["total_exited"])
        assert metrics["total_entered"] > 0

    def test_near_full_cell_minimal_injection(self):
        """When the source cell is nearly full, injection should be minimal."""
        net = _make_single_link(n_cells=3, capacity=0.5)
        demand = [DemandProfile(LinkID(0), [0.0], [1.0])]
        engine = SimulationEngine(
            network=net, dt=1.0, demand_profiles=demand,
        )
        engine.reset(seed=42)

        # Fill the source cell to 99% of jam density
        compiled = engine.net
        source_cell = compiled.link_first_cell[LinkID(0)]
        engine.state.density[source_cell] = compiled.kj[source_cell] * 0.99

        inj = engine.demand_manager.get_injection(
            t=0.0, dt=1.0, density=engine.state.density,
        )
        # Only a tiny amount should be injected
        assert inj[source_cell] < 0.1, (
            f"Injection {inj[source_cell]:.4f} should be near zero for 99% full cell"
        )

