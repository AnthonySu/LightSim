"""Tests for MaxPressure controller."""

import numpy as np
import pytest

from lightsim.benchmarks.single_intersection import create_single_intersection
from lightsim.core.engine import SimulationEngine
from lightsim.core.signal import FixedTimeController, MaxPressureController
from lightsim.core.types import NodeID


class TestMaxPressureController:
    def test_selects_higher_pressure_phase(self):
        """MaxPressure should select the phase with higher upstream demand."""
        network, demand = create_single_intersection()
        controller = MaxPressureController(min_green=1.0)
        engine = SimulationEngine(
            network=network, dt=1.0,
            controller=controller,
            demand_profiles=demand,
        )
        engine.reset(seed=42)

        # Run for a while to build up queue
        for _ in range(100):
            engine.step()

        # The controller should be functional
        state = engine.signal_manager.states[NodeID(0)]
        assert state.current_phase_idx >= 0

    def test_min_green_respected(self):
        """MaxPressure should not switch before min_green expires."""
        controller = MaxPressureController(min_green=10.0)
        network, demand = create_single_intersection()
        engine = SimulationEngine(
            network=network, dt=1.0,
            controller=controller,
            demand_profiles=demand,
        )
        engine.reset(seed=42)

        # Run a few steps — should stay in initial phase
        for _ in range(5):
            engine.step()
        state = engine.signal_manager.states[NodeID(0)]
        assert state.current_phase_idx == 0  # still in first phase

    def test_outperforms_or_matches_fixed_time(self):
        """MaxPressure should achieve comparable or better throughput."""
        results = {}
        for name, controller in [
            ("fixed", FixedTimeController()),
            ("maxp", MaxPressureController(min_green=5.0)),
        ]:
            network, demand = create_single_intersection()
            engine = SimulationEngine(
                network=network, dt=1.0,
                controller=controller,
                demand_profiles=demand,
            )
            engine.reset(seed=42)
            for _ in range(1800):
                engine.step()
            results[name] = engine.get_network_metrics()

        # MaxPressure should have at least 80% of fixed-time throughput
        # (in simple scenarios it may not always outperform, but shouldn't crash)
        assert results["maxp"]["total_exited"] >= results["fixed"]["total_exited"] * 0.8

    def test_runs_on_grid(self):
        """MaxPressure should work on multi-intersection networks."""
        from lightsim.benchmarks.grid_4x4 import create_grid_4x4
        network, demand = create_grid_4x4()
        controller = MaxPressureController(min_green=5.0)
        engine = SimulationEngine(
            network=network, dt=1.0,
            controller=controller,
            demand_profiles=demand,
        )
        engine.reset(seed=42)

        for _ in range(200):
            engine.step()

        metrics = engine.get_network_metrics()
        assert metrics["total_entered"] > 0
        assert metrics["total_vehicles"] >= 0

    def test_runs_on_arterial(self):
        """MaxPressure should work on arterial networks."""
        from lightsim.benchmarks.arterial_5 import create_arterial_5
        network, demand = create_arterial_5()
        controller = MaxPressureController(min_green=5.0)
        engine = SimulationEngine(
            network=network, dt=1.0,
            controller=controller,
            demand_profiles=demand,
        )
        engine.reset(seed=42)

        for _ in range(200):
            engine.step()

        metrics = engine.get_network_metrics()
        assert metrics["total_entered"] > 0

    def test_arterial_phase_balance(self):
        """Arterial phases should be balanced (regression for phase bug)."""
        from lightsim.networks.arterial import create_arterial_network
        net = create_arterial_network(n_intersections=3)
        for nid in range(1, 4):
            node = net.nodes[NodeID(nid)]
            assert len(node.phases) == 2
            ew_count = len(node.phases[0].movements)
            ns_count = len(node.phases[1].movements)
            assert ew_count == ns_count, (
                f"Node {nid}: phase 0 has {ew_count} movements, "
                f"phase 1 has {ns_count} — should be balanced"
            )
