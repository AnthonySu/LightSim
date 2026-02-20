"""Smoke tests for all signal controllers on single intersection."""

import pytest

from lightsim.benchmarks.single_intersection import create_single_intersection
from lightsim.core.engine import SimulationEngine
from lightsim.core.signal import (
    EfficientMaxPressureController,
    FixedTimeController,
    LostTimeAwareMaxPressureController,
    MaxPressureController,
    SOTLController,
    WebsterController,
)

CONTROLLERS = [
    ("FixedTime", FixedTimeController()),
    ("Webster", WebsterController()),
    ("SOTL", SOTLController()),
    ("MaxPressure", MaxPressureController(min_green=5.0)),
    ("LTAwareMP", LostTimeAwareMaxPressureController(min_green=5.0)),
    ("EfficientMP", EfficientMaxPressureController(min_green=5.0)),
]


@pytest.mark.parametrize("name,controller", CONTROLLERS, ids=[c[0] for c in CONTROLLERS])
def test_controller_runs_without_error(name, controller):
    """Each controller should run 500 steps on single intersection without error."""
    network, demand = create_single_intersection()
    engine = SimulationEngine(
        network=network, dt=1.0,
        controller=controller,
        demand_profiles=demand,
    )
    engine.reset(seed=42)

    for _ in range(500):
        engine.step()

    metrics = engine.get_network_metrics()
    assert metrics["total_entered"] > 0
    assert metrics["total_exited"] > 0
    assert metrics["total_vehicles"] >= 0


@pytest.mark.parametrize("name,controller", CONTROLLERS, ids=[c[0] for c in CONTROLLERS])
def test_controller_produces_positive_throughput(name, controller):
    """Each controller should produce meaningful throughput over 1800 steps."""
    network, demand = create_single_intersection()
    engine = SimulationEngine(
        network=network, dt=1.0,
        controller=controller,
        demand_profiles=demand,
    )
    engine.reset(seed=42)

    for _ in range(1800):
        engine.step()

    metrics = engine.get_network_metrics()
    # All controllers should exit at least 1000 vehicles in 1800s
    assert metrics["total_exited"] > 1000, (
        f"{name} only exited {metrics['total_exited']:.0f} vehicles"
    )
