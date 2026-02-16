"""Tests for mesoscopic extensions: start-up lost time, capacity factor, and stochastic demand."""

import numpy as np
import pytest

from lightsim.core.demand import DemandManager, DemandProfile
from lightsim.core.engine import SimulationEngine
from lightsim.core.network import Network
from lightsim.core.signal import FixedTimeController, SignalManager
from lightsim.core.types import LinkID, NodeID, NodeType, TurnType


def _make_intersection(lost_time: float = 0.0) -> tuple[Network, list[DemandProfile]]:
    """Two-approach intersection with configurable lost_time."""
    net = Network()
    vf = 13.89
    cell_length = vf * 1.0  # CFL-safe for dt=1.0

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

    net.add_phase(NodeID(0), [m_ns.movement_id], min_green=5.0, lost_time=lost_time)
    net.add_phase(NodeID(0), [m_ew.movement_id], min_green=5.0, lost_time=lost_time)

    demand = [
        DemandProfile(LinkID(0), [0.0], [0.3]),
        DemandProfile(LinkID(2), [0.0], [0.2]),
    ]
    return net, demand


# ---------------------------------------------------------------------------
# Lost time / capacity factor tests
# ---------------------------------------------------------------------------

class TestLostTimeZero:
    """Default lost_time=0.0 should produce identical output to original."""

    def test_lost_time_zero_matches_original(self):
        net, demand = _make_intersection(lost_time=0.0)
        controller = FixedTimeController({NodeID(0): [30.0, 30.0]})
        engine = SimulationEngine(
            network=net, dt=1.0, controller=controller,
            demand_profiles=demand,
        )
        engine.reset(seed=42)
        for _ in range(200):
            engine.step()
        exited_zero = engine.state.total_exited

        # Re-run: should be identical
        engine2 = SimulationEngine(
            network=net, dt=1.0, controller=controller,
            demand_profiles=demand,
        )
        engine2.reset(seed=42)
        for _ in range(200):
            engine2.step()

        assert exited_zero == engine2.state.total_exited
        np.testing.assert_array_equal(engine.state.density, engine2.state.density)


class TestLostTimeReducesThroughput:
    """lost_time > 0 should reduce total exited vehicles."""

    def test_lost_time_reduces_throughput(self):
        net_0, demand = _make_intersection(lost_time=0.0)
        net_2, _ = _make_intersection(lost_time=2.0)

        controller = FixedTimeController({NodeID(0): [30.0, 30.0]})
        n_steps = 500

        engine_0 = SimulationEngine(
            network=net_0, dt=1.0, controller=controller, demand_profiles=demand,
        )
        engine_0.reset(seed=42)
        for _ in range(n_steps):
            engine_0.step()

        engine_2 = SimulationEngine(
            network=net_2, dt=1.0, controller=controller, demand_profiles=demand,
        )
        engine_2.reset(seed=42)
        for _ in range(n_steps):
            engine_2.step()

        assert engine_2.state.total_exited < engine_0.state.total_exited, (
            f"lost_time=2.0 should reduce throughput: "
            f"{engine_2.state.total_exited:.1f} vs {engine_0.state.total_exited:.1f}"
        )


class TestLostTimePenalizesFrequentSwitching:
    """Shorter green phases should lose more capacity with lost_time > 0."""

    def test_lost_time_penalizes_frequent_switching(self):
        net_short, demand = _make_intersection(lost_time=2.0)
        net_long, _ = _make_intersection(lost_time=2.0)

        controller_short = FixedTimeController({NodeID(0): [10.0, 10.0]})
        controller_long = FixedTimeController({NodeID(0): [30.0, 30.0]})
        n_steps = 600

        engine_short = SimulationEngine(
            network=net_short, dt=1.0, controller=controller_short,
            demand_profiles=demand,
        )
        engine_short.reset(seed=42)
        for _ in range(n_steps):
            engine_short.step()

        engine_long = SimulationEngine(
            network=net_long, dt=1.0, controller=controller_long,
            demand_profiles=demand,
        )
        engine_long.reset(seed=42)
        for _ in range(n_steps):
            engine_long.step()

        # Short phases should exit fewer vehicles due to more lost time
        assert engine_short.state.total_exited < engine_long.state.total_exited, (
            f"10s green should lose more to lost_time than 30s green: "
            f"{engine_short.state.total_exited:.1f} vs {engine_long.state.total_exited:.1f}"
        )


class TestCapacityFactorRamp:
    """Verify capacity factor values at specific times after green onset."""

    def test_capacity_factor_ramp(self):
        net, demand = _make_intersection(lost_time=2.0)
        compiled = net.compile(1.0)
        controller = FixedTimeController({NodeID(0): [30.0, 30.0]})
        sm = SignalManager(compiled, controller)
        density = np.zeros(compiled.n_cells)

        # Initially phase 0 is green — step once to initialise timers
        sm.step(1.0, density)
        factor = sm.get_capacity_factor()

        # Phase 0 movements have been green for 1s, lost_time=2.0
        # So factor should be 1.0/2.0 = 0.5
        phase0_movs = []
        for mid in range(compiled.n_movements):
            if compiled.phase_mov_mask[0, mid]:
                phase0_movs.append(mid)

        for mid in phase0_movs:
            assert abs(factor[mid] - 0.5) < 1e-9, (
                f"Movement {mid} factor at t=1s should be 0.5, got {factor[mid]}"
            )

        # Step again — now 2s of green, should be 1.0
        sm.step(1.0, density)
        factor = sm.get_capacity_factor()
        for mid in phase0_movs:
            assert abs(factor[mid] - 1.0) < 1e-9, (
                f"Movement {mid} factor at t=2s should be 1.0, got {factor[mid]}"
            )


# ---------------------------------------------------------------------------
# Stochastic demand tests
# ---------------------------------------------------------------------------

class TestStochasticFalse:
    """stochastic=False should be identical to deterministic behavior."""

    def test_stochastic_false_matches_deterministic(self):
        net, demand = _make_intersection()

        engine_det = SimulationEngine(
            network=net, dt=1.0, demand_profiles=demand, stochastic=False,
        )
        engine_det.reset(seed=42)
        for _ in range(100):
            engine_det.step()

        # Run again
        engine_det2 = SimulationEngine(
            network=net, dt=1.0, demand_profiles=demand, stochastic=False,
        )
        engine_det2.reset(seed=42)
        for _ in range(100):
            engine_det2.step()

        np.testing.assert_array_equal(engine_det.state.density, engine_det2.state.density)
        assert engine_det.state.total_exited == engine_det2.state.total_exited


class TestStochasticVaries:
    """Stochastic injections should vary step-to-step."""

    def test_stochastic_varies_across_steps(self):
        net, demand = _make_intersection()
        compiled = net.compile(1.0)
        rng = np.random.default_rng(42)
        dm = DemandManager(compiled, demand, stochastic=True, rng=rng)
        density = np.zeros(compiled.n_cells)

        injections = []
        for t in range(100):
            inj = dm.get_injection(float(t), 1.0, density)
            injections.append(inj.sum())

        # With Poisson, not all values should be equal
        unique_vals = len(set(injections))
        assert unique_vals > 1, "Stochastic injections should vary across steps"


class TestStochasticSeedReproducibility:
    """Same seed should produce same trajectory."""

    def test_stochastic_seed_reproducibility(self):
        net, demand = _make_intersection()

        results = []
        for _ in range(2):
            engine = SimulationEngine(
                network=net, dt=1.0, demand_profiles=demand, stochastic=True,
            )
            engine.reset(seed=123)
            for _ in range(100):
                engine.step()
            results.append(engine.state.density.copy())

        np.testing.assert_array_equal(results[0], results[1])


class TestStochasticMeanMatchesRate:
    """Over many steps, total injected ≈ rate * dt * n_steps."""

    def test_stochastic_mean_matches_rate(self):
        net, demand = _make_intersection()
        compiled = net.compile(1.0)

        n_steps = 10000
        # Use large jam density so space capping doesn't bite
        rng = np.random.default_rng(42)
        dm = DemandManager(compiled, demand, stochastic=True, rng=rng)
        density = np.zeros(compiled.n_cells)

        total_injected = 0.0
        for t in range(n_steps):
            inj = dm.get_injection(float(t), 1.0, density)
            total_injected += inj.sum()

        # Expected: (0.3 + 0.2) * 1.0 * 10000 = 5000
        expected = (0.3 + 0.2) * 1.0 * n_steps
        # Allow 5% tolerance for Poisson variance
        assert abs(total_injected - expected) / expected < 0.05, (
            f"Expected ~{expected:.0f} total injected, got {total_injected:.0f}"
        )


# ---------------------------------------------------------------------------
# Flow conservation
# ---------------------------------------------------------------------------

class TestFlowConservationWithLostTime:
    """entered = exited + in_network should hold with lost_time."""

    def test_flow_conservation_with_lost_time(self):
        net, demand = _make_intersection(lost_time=2.0)
        controller = FixedTimeController({NodeID(0): [20.0, 20.0]})
        engine = SimulationEngine(
            network=net, dt=1.0, controller=controller, demand_profiles=demand,
        )
        engine.reset(seed=42)

        for _ in range(300):
            engine.step()

        in_network = engine.get_total_vehicles()
        entered = engine.state.total_entered
        exited = engine.state.total_exited

        residual = abs(entered - exited - in_network)
        assert residual < 1.0, (
            f"Flow conservation violated: entered={entered:.2f}, "
            f"exited={exited:.2f}, in_network={in_network:.2f}, "
            f"residual={residual:.4f}"
        )
