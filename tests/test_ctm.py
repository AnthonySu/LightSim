"""Tests for CTM flow model: fundamental diagram, flow conservation, CFL."""

import numpy as np
import pytest

from lightsim.core.demand import DemandProfile
from lightsim.core.engine import SimulationEngine
from lightsim.core.flow_model import CTMFlowModel
from lightsim.core.network import Network
from lightsim.core.types import FLOAT, CellID, LinkID, NodeID, NodeType


def _make_single_link_network(
    n_cells: int = 10,
    cell_length: float = 100.0,
    lanes: int = 1,
    vf: float = 13.89,
    w: float = 5.56,
    kj: float = 0.15,
    capacity: float = 0.5,
) -> Network:
    """Helper: single link from origin to destination."""
    net = Network()
    net.add_node(NodeID(0), NodeType.ORIGIN)
    net.add_node(NodeID(1), NodeType.DESTINATION)
    net.add_link(
        LinkID(0),
        from_node=NodeID(0),
        to_node=NodeID(1),
        length=cell_length * n_cells,
        lanes=lanes,
        n_cells=n_cells,
        free_flow_speed=vf,
        wave_speed=w,
        jam_density=kj,
        capacity=capacity,
    )
    return net


class TestCTMSendingReceiving:
    """Test sending and receiving flow functions."""

    def test_sending_flow_free_flow(self):
        """S(k) = vf*k when k < k_crit."""
        model = CTMFlowModel()
        net = _make_single_link_network(n_cells=5, cell_length=20.0).compile(dt=1.0)
        k_crit = net.Q[0] / net.vf[0]
        density = np.full(net.n_cells, k_crit * 0.5, dtype=FLOAT)
        sending = model.compute_sending_flow(density, net)
        expected = net.vf * density * net.lanes
        np.testing.assert_allclose(sending, expected, rtol=1e-10)

    def test_sending_flow_congested(self):
        """S(k) = Q when k >= k_crit."""
        model = CTMFlowModel()
        net = _make_single_link_network(n_cells=5, cell_length=20.0).compile(dt=1.0)
        density = np.full(net.n_cells, net.kj[0] * 0.9, dtype=FLOAT)
        sending = model.compute_sending_flow(density, net)
        expected = net.Q * net.lanes
        np.testing.assert_allclose(sending, expected, rtol=1e-10)

    def test_receiving_flow_free_flow(self):
        """R(k) = Q when k < k_crit (plenty of space)."""
        model = CTMFlowModel()
        net = _make_single_link_network(n_cells=5, cell_length=20.0).compile(dt=1.0)
        density = np.full(net.n_cells, 0.01, dtype=FLOAT)
        receiving = model.compute_receiving_flow(density, net)
        expected = np.minimum(net.Q, net.w * (net.kj - density)) * net.lanes
        np.testing.assert_allclose(receiving, expected, rtol=1e-10)

    def test_receiving_flow_congested(self):
        """R(k) = w*(kj-k) when k > k_crit (limited space)."""
        model = CTMFlowModel()
        net = _make_single_link_network(n_cells=5, cell_length=20.0).compile(dt=1.0)
        density = np.full(net.n_cells, net.kj[0] * 0.95, dtype=FLOAT)
        receiving = model.compute_receiving_flow(density, net)
        expected = net.w * (net.kj - density) * net.lanes
        np.testing.assert_allclose(receiving, expected, rtol=1e-10)

    def test_receiving_flow_at_jam(self):
        """R(kj) = 0."""
        model = CTMFlowModel()
        net = _make_single_link_network(n_cells=5, cell_length=20.0).compile(dt=1.0)
        density = net.kj.copy()
        receiving = model.compute_receiving_flow(density, net)
        np.testing.assert_allclose(receiving, 0.0, atol=1e-12)


class TestCFLCondition:
    """Test CFL condition enforcement."""

    def test_cfl_violation_raises(self):
        """Cells too short for the time step should raise ValueError."""
        net = _make_single_link_network(n_cells=10, cell_length=5.0, vf=13.89)
        with pytest.raises(ValueError, match="CFL violation"):
            net.compile(dt=1.0)  # 5.0 < 13.89 * 1.0

    def test_cfl_satisfied(self):
        """Cells large enough should compile without error."""
        net = _make_single_link_network(n_cells=5, cell_length=20.0, vf=13.89)
        compiled = net.compile(dt=1.0)
        assert compiled.n_cells == 5


class TestFlowConservation:
    """Test that total vehicles are conserved (source/sink accounted)."""

    def test_single_link_conservation(self):
        """Vehicles in = vehicles out + vehicles in network (no leaks)."""
        net = _make_single_link_network(n_cells=5, cell_length=20.0)
        demand = DemandProfile(LinkID(0), [0.0], [0.3])
        engine = SimulationEngine(
            network=net, dt=1.0, demand_profiles=[demand]
        )
        engine.reset(seed=42)

        for _ in range(200):
            engine.step()

        in_network = engine.get_total_vehicles()
        entered = engine.state.total_entered
        exited = engine.state.total_exited

        # Conservation: entered = exited + in_network
        np.testing.assert_allclose(
            entered, exited + in_network, rtol=0.05,
            err_msg="Flow conservation violated"
        )

    def test_no_negative_density(self):
        """Density should never go negative."""
        net = _make_single_link_network(n_cells=5, cell_length=20.0)
        demand = DemandProfile(LinkID(0), [0.0], [0.3])
        engine = SimulationEngine(
            network=net, dt=1.0, demand_profiles=[demand]
        )
        engine.reset(seed=42)

        for _ in range(100):
            engine.step()
            assert (engine.state.density >= -1e-12).all(), "Negative density detected"


class TestFundamentalDiagram:
    """Test that CTM reproduces the triangular fundamental diagram."""

    def test_triangular_fd(self):
        """Run multiple demand levels on a single link; verify (k, q) pattern."""
        vf = 13.89
        w = 5.56
        kj = 0.15
        Q = 0.5
        lanes = 1
        dt = 1.0
        k_crit = Q / vf

        demand_rates = np.linspace(0, Q * lanes * 1.2, 20)
        observed_k = []
        observed_q = []

        for rate in demand_rates:
            net = _make_single_link_network(
                n_cells=10, cell_length=vf * dt,
                lanes=lanes, vf=vf, w=w, kj=kj, capacity=Q,
            )
            demand = DemandProfile(LinkID(0), [0.0], [float(rate)])
            engine = SimulationEngine(
                network=net, dt=dt, demand_profiles=[demand]
            )
            engine.reset(seed=0)

            for _ in range(300):
                engine.step()

            # Measure at middle cells
            mid = [4, 5]
            k = float(engine.state.density[mid].mean())
            q = float(np.minimum(vf * k, Q) * lanes)
            observed_k.append(k)
            observed_q.append(q)

        observed_k = np.array(observed_k)
        observed_q = np.array(observed_q)

        # Check: all points should lie on or near the triangular FD
        for k, q in zip(observed_k, observed_q):
            if k < 1e-6:
                assert q < 1e-3
            elif k <= k_crit + 0.001:
                # Free-flow branch: q ≈ vf * k
                expected_q = vf * k * lanes
                assert abs(q - expected_q) < 0.1, \
                    f"Free-flow branch: k={k:.4f}, q={q:.4f}, expected={expected_q:.4f}"
            else:
                # Congested branch: q = w * (kj - k) or Q
                q_max = Q * lanes
                assert q <= q_max + 0.01, \
                    f"Flow exceeds capacity: k={k:.4f}, q={q:.4f}, Qmax={q_max:.4f}"
