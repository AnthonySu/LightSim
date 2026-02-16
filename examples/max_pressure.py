"""Compare FixedTime vs MaxPressure control on a single intersection."""

from lightsim.benchmarks.single_intersection import create_single_intersection
from lightsim.core.engine import SimulationEngine
from lightsim.core.signal import FixedTimeController, MaxPressureController

STEPS = 3600  # 1 hour

for name, controller in [
    ("FixedTime(30s)", FixedTimeController()),
    ("MaxPressure",    MaxPressureController(min_green=5.0)),
]:
    network, demand = create_single_intersection()
    engine = SimulationEngine(
        network=network, dt=1.0,
        controller=controller,
        demand_profiles=demand,
    )
    engine.reset(seed=42)

    total_queue = 0.0
    for _ in range(STEPS):
        engine.step()
        # Accumulate queue across all links
        for link in network.links.values():
            total_queue += engine.get_link_queue(link.link_id)

    m = engine.get_network_metrics()
    avg_queue = total_queue / STEPS
    print(f"{name:<20}  throughput={m['total_exited']:>7.0f}  "
          f"avg_queue={avg_queue:>6.2f}  vehicles_remaining={m['total_vehicles']:.1f}")
