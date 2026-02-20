"""Numerical edge case tests: boundary densities, zero demand, saturation, single-cell links."""

import numpy as np
import pytest

from lightsim.core.demand import DemandProfile
from lightsim.core.engine import SimulationEngine
from lightsim.core.flow_model import CTMFlowModel
from lightsim.core.network import Network
from lightsim.core.signal import FixedTimeController
from lightsim.core.types import FLOAT, CellID, LinkID, NodeID, NodeType, TurnType


def _make_single_link(n_cells=5, cell_length=20.0, lanes=1, vf=13.89, w=5.56,
                       kj=0.15, capacity=0.5):
    """Helper: origin → destination with one link."""
    net = Network()
    net.add_node(NodeID(0), NodeType.ORIGIN)
    net.add_node(NodeID(1), NodeType.DESTINATION)
    net.add_link(
        LinkID(0), NodeID(0), NodeID(1),
        length=cell_length * n_cells, lanes=lanes, n_cells=n_cells,
        free_flow_speed=vf, wave_speed=w, jam_density=kj, capacity=capacity,
    )
    return net


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


# ---------------------------------------------------------------------------
# Boundary density tests (sending/receiving at k=0, k=k_crit, k=kj)
# ---------------------------------------------------------------------------

class TestBoundaryDensities:
    """Verify sending/receiving at exact boundary densities."""

    def test_sending_at_zero_density(self):
        """S(0) = 0."""
        model = CTMFlowModel()
        net = _make_single_link(n_cells=5, cell_length=20.0).compile(dt=1.0)
        density = np.zeros(net.n_cells, dtype=FLOAT)
        sending = model.compute_sending_flow(density, net)
        np.testing.assert_allclose(sending, 0.0, atol=1e-15)

    def test_receiving_at_zero_density(self):
        """R(0) = min(Q, w*kj) * lanes = Q * lanes (for standard params)."""
        model = CTMFlowModel()
        net = _make_single_link(n_cells=5, cell_length=20.0).compile(dt=1.0)
        density = np.zeros(net.n_cells, dtype=FLOAT)
        receiving = model.compute_receiving_flow(density, net)
        expected = np.minimum(net.Q, net.w * net.kj) * net.lanes
        np.testing.assert_allclose(receiving, expected, rtol=1e-12)

    def test_sending_at_critical_density(self):
        """S(k_crit) = Q * lanes (exactly at capacity)."""
        model = CTMFlowModel()
        net = _make_single_link(n_cells=5, cell_length=20.0).compile(dt=1.0)
        k_crit = net.Q / net.vf
        density = k_crit.copy()
        sending = model.compute_sending_flow(density, net)
        expected = net.Q * net.lanes
        np.testing.assert_allclose(sending, expected, rtol=1e-10)

    def test_receiving_at_critical_density(self):
        """R(k_crit) = Q * lanes (capacity at critical density)."""
        model = CTMFlowModel()
        net = _make_single_link(n_cells=5, cell_length=20.0).compile(dt=1.0)
        k_crit = net.Q / net.vf
        density = k_crit.copy()
        receiving = model.compute_receiving_flow(density, net)
        expected = np.minimum(net.Q, net.w * (net.kj - density)) * net.lanes
        np.testing.assert_allclose(receiving, expected, rtol=1e-10)

    def test_sending_at_jam_density(self):
        """S(kj) = Q * lanes (congested branch still sends at capacity)."""
        model = CTMFlowModel()
        net = _make_single_link(n_cells=5, cell_length=20.0).compile(dt=1.0)
        density = net.kj.copy()
        sending = model.compute_sending_flow(density, net)
        expected = net.Q * net.lanes
        np.testing.assert_allclose(sending, expected, rtol=1e-10)

    def test_receiving_at_jam_density(self):
        """R(kj) = 0 (no space)."""
        model = CTMFlowModel()
        net = _make_single_link(n_cells=5, cell_length=20.0).compile(dt=1.0)
        density = net.kj.copy()
        receiving = model.compute_receiving_flow(density, net)
        np.testing.assert_allclose(receiving, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Zero demand
# ---------------------------------------------------------------------------

class TestZeroDemand:
    """Simulation with zero or no demand should not crash or create vehicles."""

    def test_no_demand_profiles(self):
        """Engine runs fine with no demand profiles at all."""
        net = _make_single_link()
        engine = SimulationEngine(network=net, dt=1.0, demand_profiles=[])
        engine.reset(seed=42)
        for _ in range(100):
            engine.step()
        assert engine.state.total_entered == 0.0
        assert engine.state.total_exited == 0.0
        assert (engine.state.density == 0.0).all()

    def test_zero_rate_demand(self):
        """Demand profile with rate=0 should inject nothing."""
        net = _make_single_link()
        demand = [DemandProfile(LinkID(0), [0.0], [0.0])]
        engine = SimulationEngine(network=net, dt=1.0, demand_profiles=demand)
        engine.reset(seed=42)
        for _ in range(100):
            engine.step()
        assert engine.state.total_entered == 0.0

    def test_signalized_intersection_zero_demand(self):
        """Signalized intersection with no demand shouldn't crash."""
        net = _make_signalized_intersection()
        controller = FixedTimeController({NodeID(0): [30.0, 30.0]})
        engine = SimulationEngine(
            network=net, dt=1.0, controller=controller, demand_profiles=[],
        )
        engine.reset(seed=42)
        for _ in range(200):
            engine.step()
        assert engine.state.total_entered == 0.0
        assert (engine.state.density >= 0).all()


# ---------------------------------------------------------------------------
# Saturation: density should never exceed jam density
# ---------------------------------------------------------------------------

class TestDensitySaturation:
    """Density should be bounded by [0, kj] even under extreme demand."""

    def test_high_demand_bounded(self):
        """Very high demand should not push density above kj."""
        net = _make_single_link(n_cells=3, cell_length=20.0, capacity=0.5)
        # Inject at 5x capacity
        demand = [DemandProfile(LinkID(0), [0.0], [2.5])]
        engine = SimulationEngine(network=net, dt=1.0, demand_profiles=demand)
        engine.reset(seed=42)
        for _ in range(500):
            engine.step()
            assert (engine.state.density >= -1e-12).all(), "Negative density"
            assert (engine.state.density <= engine.net.kj + 1e-6).all(), (
                f"Density exceeds kj: max={engine.state.density.max():.6f}, "
                f"kj={engine.net.kj[0]:.6f}"
            )

    def test_intersection_high_demand_bounded(self):
        """Signalized intersection under oversaturation stays bounded."""
        net = _make_signalized_intersection()
        demand = [
            DemandProfile(LinkID(0), [0.0], [1.5]),
            DemandProfile(LinkID(2), [0.0], [1.5]),
        ]
        controller = FixedTimeController({NodeID(0): [30.0, 30.0]})
        engine = SimulationEngine(
            network=net, dt=1.0, controller=controller, demand_profiles=demand,
        )
        engine.reset(seed=42)
        for _ in range(500):
            engine.step()
            assert (engine.state.density >= -1e-12).all(), "Negative density"
            assert (engine.state.density <= engine.net.kj + 1e-6).all(), (
                f"Density exceeds kj at step {engine.state.step_count}"
            )


# ---------------------------------------------------------------------------
# Single-cell links
# ---------------------------------------------------------------------------

class TestSingleCellLink:
    """Links with exactly one cell should work correctly."""

    def test_single_cell_propagation(self):
        """Vehicles enter and exit a single-cell link."""
        net = _make_single_link(n_cells=1, cell_length=20.0)
        demand = [DemandProfile(LinkID(0), [0.0], [0.3])]
        engine = SimulationEngine(network=net, dt=1.0, demand_profiles=demand)
        engine.reset(seed=42)
        for _ in range(200):
            engine.step()
        # Should enter and exit vehicles normally
        assert engine.state.total_entered > 0
        assert engine.state.total_exited > 0

    def test_single_cell_conservation(self):
        """Flow conservation on a single-cell link."""
        net = _make_single_link(n_cells=1, cell_length=20.0)
        demand = [DemandProfile(LinkID(0), [0.0], [0.2])]
        engine = SimulationEngine(network=net, dt=1.0, demand_profiles=demand)
        engine.reset(seed=42)
        for _ in range(300):
            engine.step()
        in_net = engine.get_total_vehicles()
        np.testing.assert_allclose(
            engine.state.total_entered,
            engine.state.total_exited + in_net,
            rtol=0.05,
        )


# ---------------------------------------------------------------------------
# Multi-lane consistency
# ---------------------------------------------------------------------------

class TestMultiLane:
    """Multi-lane links should carry proportionally more flow."""

    def test_two_lane_higher_throughput(self):
        """A 2-lane link should discharge roughly 2x a 1-lane link."""
        results = {}
        for lanes in [1, 2]:
            net = _make_single_link(n_cells=5, cell_length=20.0, lanes=lanes)
            demand = [DemandProfile(LinkID(0), [0.0], [0.3 * lanes])]
            engine = SimulationEngine(network=net, dt=1.0, demand_profiles=demand)
            engine.reset(seed=42)
            for _ in range(300):
                engine.step()
            results[lanes] = engine.state.total_exited

        # 2-lane should have roughly 2x throughput
        ratio = results[2] / max(results[1], 1e-9)
        assert 1.5 < ratio < 2.5, (
            f"2-lane/1-lane ratio {ratio:.2f} out of expected range [1.5, 2.5]"
        )


# ---------------------------------------------------------------------------
# Reset state isolation
# ---------------------------------------------------------------------------

class TestResetIsolation:
    """Engine.reset() should fully clean state."""

    def test_reset_clears_density(self):
        """After reset, density is all zeros."""
        net = _make_single_link()
        demand = [DemandProfile(LinkID(0), [0.0], [0.3])]
        engine = SimulationEngine(network=net, dt=1.0, demand_profiles=demand)
        engine.reset(seed=42)
        for _ in range(100):
            engine.step()
        assert engine.state.density.sum() > 0

        engine.reset(seed=99)
        assert (engine.state.density == 0.0).all()
        assert engine.state.total_entered == 0.0
        assert engine.state.total_exited == 0.0
        assert engine.state.time == 0.0
        assert engine.state.step_count == 0

    def test_different_seed_different_trajectory(self):
        """Different seeds on stochastic engine produce different results."""
        net = _make_single_link()
        demand = [DemandProfile(LinkID(0), [0.0], [0.3])]

        results = []
        for seed in [42, 99]:
            engine = SimulationEngine(
                network=net, dt=1.0, demand_profiles=demand, stochastic=True,
            )
            engine.reset(seed=seed)
            for _ in range(100):
                engine.step()
            results.append(engine.state.density.copy())

        assert not np.array_equal(results[0], results[1]), (
            "Different seeds should produce different trajectories"
        )


# ---------------------------------------------------------------------------
# Movement flow with all-red signal
# ---------------------------------------------------------------------------

class TestAllRedSignal:
    """When all signals are red, no movement flow should occur."""

    def test_no_movement_flow_during_yellow(self):
        """During yellow/all-red, movement flow is zero."""
        model = CTMFlowModel()
        net = _make_signalized_intersection().compile(dt=1.0)

        # Set some density on inbound cells
        density = np.zeros(net.n_cells, dtype=FLOAT)
        for lid in [LinkID(0), LinkID(2)]:
            for cid in net.link_cells[lid]:
                density[cid] = 0.05

        sending = model.compute_sending_flow(density, net)
        receiving = model.compute_receiving_flow(density, net)
        # All movements red
        signal_mask = np.zeros(net.n_movements, dtype=FLOAT)

        _, mov_flow = model.compute_flow(
            density, sending, receiving, signal_mask, net, dt=1.0
        )
        np.testing.assert_allclose(mov_flow, 0.0, atol=1e-15,
                                    err_msg="Flow should be zero during all-red")


# ---------------------------------------------------------------------------
# Intra-link flow direction: flow should be non-negative
# ---------------------------------------------------------------------------

class TestFlowNonNegative:
    """All flow values should be non-negative."""

    def test_intra_flow_non_negative(self):
        """Intra-link cell-to-cell flow should always be >= 0."""
        model = CTMFlowModel()
        net = _make_single_link(n_cells=5, cell_length=20.0).compile(dt=1.0)

        # Random density profile
        rng = np.random.default_rng(42)
        for _ in range(100):
            density = rng.uniform(0, net.kj[0], size=net.n_cells).astype(FLOAT)
            sending = model.compute_sending_flow(density, net)
            receiving = model.compute_receiving_flow(density, net)
            signal_mask = np.ones(net.n_movements, dtype=FLOAT)
            intra_flow, _ = model.compute_flow(
                density, sending, receiving, signal_mask, net, dt=1.0
            )
            assert (intra_flow >= -1e-12).all(), (
                f"Negative intra-link flow: min={intra_flow.min()}"
            )

    def test_movement_flow_non_negative(self):
        """Movement flow should always be >= 0."""
        model = CTMFlowModel()
        net = _make_signalized_intersection().compile(dt=1.0)

        rng = np.random.default_rng(42)
        for _ in range(100):
            density = rng.uniform(0, net.kj.min(), size=net.n_cells).astype(FLOAT)
            sending = model.compute_sending_flow(density, net)
            receiving = model.compute_receiving_flow(density, net)
            signal_mask = np.ones(net.n_movements, dtype=FLOAT)
            _, mov_flow = model.compute_flow(
                density, sending, receiving, signal_mask, net, dt=1.0
            )
            assert (mov_flow >= -1e-12).all(), (
                f"Negative movement flow: min={mov_flow.min()}"
            )
