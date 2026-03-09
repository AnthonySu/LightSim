"""Mesoscopic mode: stochastic demand + start-up lost time.

Compares controller performance in default (deterministic) vs mesoscopic mode,
showing how start-up lost time penalizes frequent phase switching.
"""

try:
    from lightsim.benchmarks.single_intersection import create_single_intersection
    from lightsim.core.engine import SimulationEngine
    from lightsim.core.signal import (
        FixedTimeController,
        LostTimeAwareMaxPressureController,
        MaxPressureController,
    )
except ImportError:
    print("Please install lightsim: pip install -e '.[all]'")
    raise

STEPS = 3600  # 1 hour
LOST_TIME = 2.0  # seconds of capacity ramp-up after each green onset

controllers = [
    ("FixedTime(30s)", FixedTimeController()),
    ("MaxPressure(mg5)", MaxPressureController(min_green=5.0)),
    ("LT-Aware-MP", LostTimeAwareMaxPressureController()),
]


def run(mode: str, stochastic: bool, lost_time: float):
    print(f"\n{'=' * 60}")
    print(f"  Mode: {mode}  (stochastic={stochastic}, lost_time={lost_time}s)")
    print(f"{'=' * 60}")

    for name, controller in controllers:
        network, demand = create_single_intersection()

        # Enable lost time on all phases
        if lost_time > 0:
            for node in network.nodes.values():
                for phase in node.phases:
                    phase.lost_time = lost_time

        engine = SimulationEngine(
            network=network, dt=1.0,
            controller=controller,
            demand_profiles=demand,
            stochastic=stochastic,
        )
        engine.reset(seed=42)

        for _ in range(STEPS):
            engine.step()

        m = engine.get_network_metrics()
        print(f"  {name:<20}  throughput={m['total_exited']:>6.0f}"
              f"  vehicles_remaining={m['total_vehicles']:>5.1f}")


def main():
    # Default: deterministic, no lost time
    run("Default", stochastic=False, lost_time=0.0)

    # Mesoscopic: stochastic demand + start-up lost time
    run("Mesoscopic", stochastic=True, lost_time=LOST_TIME)


if __name__ == "__main__":
    main()
