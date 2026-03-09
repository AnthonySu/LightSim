"""Engine numerical stability tests: high density, zero capacity, stochastic variance."""

import numpy as np
import pytest

from lightsim.core.demand import DemandProfile
from lightsim.core.engine import SimulationEngine
from lightsim.core.flow_model import CTMFlowModel
from lightsim.core.network import Network
from lightsim.core.signal import FixedTimeController
from lightsim.core.types import FLOAT, LinkID, NodeID, NodeType, TurnType


def _make_single_link(n_cells=5, vf=13.89, w=5.56, kj=0.15, capacity=0.5,
                       lanes=1):
    """Origin -> Destination with one link."""
    net = Network()
    net.add_node(NodeID(0), NodeType.ORIGIN)
    net.add_node(NodeID(1), NodeType.DESTINATION)
    cell_length = vf * 1.0  # satisfy CFL for dt=1.0
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


class TestHighDensityCells:
    """Density near jam density (kj) should remain stable."""

    def test_density_at_kj_stays_bounded(self):
        """Starting at jam density, simulation should not produce NaN or overflow."""
        net = _make_single_link(n_cells=5)
        engine = SimulationEngine(network=net, dt=1.0, demand_profiles=[])
        engine.reset(seed=42)

        # Set all cells to jam density
        engine.state.density[:] = engine.net.kj

        for _ in range(100):
            engine.step()

        assert np.isfinite(engine.state.density).all()
        assert (engine.state.density >= 0).all()
        assert (engine.state.density <= engine.net.kj + 1e-6).all()

    def test_density_just_below_kj(self):
        """Density at 99.9% of kj with incoming demand stays stable."""
        net = _make_single_link(n_cells=5)
        demand = [DemandProfile(LinkID(0), [0.0], [0.5])]
        engine = SimulationEngine(
            network=net, dt=1.0, demand_profiles=demand,
        )
        engine.reset(seed=42)

        # Fill to near capacity
        engine.state.density[:] = engine.net.kj * 0.999

        for _ in range(200):
            engine.step()
            assert np.isfinite(engine.state.density).all()
            assert (engine.state.density >= -1e-12).all()
            assert (engine.state.density <= engine.net.kj + 1e-6).all()

    def test_high_density_intersection(self):
        """Intersection at near-jam density should remain stable."""
        net = _make_signalized_intersection()
        controller = FixedTimeController({NodeID(0): [30.0, 30.0]})
        demand = [
            DemandProfile(LinkID(0), [0.0], [2.0]),
            DemandProfile(LinkID(2), [0.0], [2.0]),
        ]
        engine = SimulationEngine(
            network=net, dt=1.0, controller=controller,
            demand_profiles=demand,
        )
        engine.reset(seed=42)

        # Pre-fill to high density
        engine.state.density[:] = engine.net.kj * 0.9

        for _ in range(300):
            engine.step()

        assert np.isfinite(engine.state.density).all()
        assert (engine.state.density >= 0).all()


class TestZeroCapacityLinks:
    """Links with very low (near-zero) capacity should not cause division errors."""

    def test_very_low_capacity_no_crash(self):
        """A link with capacity near zero should still run without crashing."""
        net = _make_single_link(n_cells=3, capacity=1e-6)
        demand = [DemandProfile(LinkID(0), [0.0], [0.01])]
        engine = SimulationEngine(
            network=net, dt=1.0, demand_profiles=demand,
        )
        engine.reset(seed=42)

        for _ in range(100):
            engine.step()

        assert np.isfinite(engine.state.density).all()
        assert (engine.state.density >= 0).all()

    def test_low_capacity_flow_model(self):
        """CTM flow model with very low capacity produces finite flows."""
        model = CTMFlowModel()
        net = _make_single_link(n_cells=3, capacity=1e-6).compile(dt=1.0)
        density = np.full(net.n_cells, 0.01, dtype=FLOAT)

        sending = model.compute_sending_flow(density, net)
        receiving = model.compute_receiving_flow(density, net)

        assert np.isfinite(sending).all()
        assert np.isfinite(receiving).all()
        assert (sending >= 0).all()
        assert (receiving >= 0).all()


class TestStochasticVariance:
    """Stochastic mode should produce reasonable variance over many steps."""

    def test_stochastic_produces_different_trajectories(self):
        """Different seeds in stochastic mode should yield different densities."""
        net = _make_single_link(n_cells=5)
        demand = [DemandProfile(LinkID(0), [0.0], [0.3])]

        trajectories = []
        for seed in [1, 2, 3]:
            engine = SimulationEngine(
                network=net, dt=1.0, demand_profiles=demand, stochastic=True,
            )
            engine.reset(seed=seed)
            for _ in range(200):
                engine.step()
            trajectories.append(engine.state.density.copy())

        # At least some pairs should differ
        any_different = False
        for i in range(len(trajectories)):
            for j in range(i + 1, len(trajectories)):
                if not np.array_equal(trajectories[i], trajectories[j]):
                    any_different = True
                    break

        assert any_different, "Stochastic mode with different seeds should differ"

    def test_stochastic_mean_close_to_deterministic(self):
        """Average of many stochastic runs should approximate deterministic result."""
        net = _make_single_link(n_cells=5)
        demand = [DemandProfile(LinkID(0), [0.0], [0.3])]
        n_steps = 300

        # Deterministic run
        engine_det = SimulationEngine(
            network=net, dt=1.0, demand_profiles=demand, stochastic=False,
        )
        engine_det.reset(seed=42)
        for _ in range(n_steps):
            engine_det.step()
        det_exited = engine_det.state.total_exited

        # Average of stochastic runs
        n_runs = 20
        stoch_exited = []
        for seed in range(n_runs):
            engine_s = SimulationEngine(
                network=net, dt=1.0, demand_profiles=demand, stochastic=True,
            )
            engine_s.reset(seed=seed)
            for _ in range(n_steps):
                engine_s.step()
            stoch_exited.append(engine_s.state.total_exited)

        mean_stoch = np.mean(stoch_exited)
        # Mean of stochastic runs should be within 20% of deterministic
        assert abs(mean_stoch - det_exited) / max(det_exited, 1.0) < 0.2, (
            f"Stochastic mean {mean_stoch:.1f} too far from deterministic {det_exited:.1f}"
        )

    def test_stochastic_variance_positive(self):
        """Stochastic mode should produce non-zero variance in total exited."""
        net = _make_single_link(n_cells=5)
        demand = [DemandProfile(LinkID(0), [0.0], [0.3])]

        exited_values = []
        for seed in range(15):
            engine = SimulationEngine(
                network=net, dt=1.0, demand_profiles=demand, stochastic=True,
            )
            engine.reset(seed=seed)
            for _ in range(200):
                engine.step()
            exited_values.append(engine.state.total_exited)

        variance = np.var(exited_values)
        assert variance > 0, "Stochastic mode should produce non-zero variance"

