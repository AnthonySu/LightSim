#!/usr/bin/env python3
"""Emergency Vehicle Corridor Optimization Example.

Demonstrates LightSim's EV tracking overlay: an emergency vehicle travels
through a 4x4 grid network while signal controllers manage corridor
formation. Compares three strategies:

  1. FixedTime — no preemption (baseline)
  2. MaxPressure — throughput-optimal, not EV-aware
  3. Greedy Preemption — forces green for the EV at every intersection

Usage::

    pip install lightsim
    python examples/ev_corridor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the examples directory or repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    from lightsim.core import (
        SimulationEngine,
        EVTracker,
        FixedTimeController,
        MaxPressureController,
    )
    from lightsim.core.demand import DemandProfile
    from lightsim.networks.grid import create_grid_network

    rows, cols = 4, 4
    dt = 5.0  # seconds per step

    # Build the network and a simple demand profile for origin links
    network = create_grid_network(rows, cols)
    from lightsim.core.types import NodeType
    origin_nodes = {nid for nid, n in network.nodes.items() if n.node_type == NodeType.ORIGIN}
    demand_profiles = [
        DemandProfile(link_id=link.link_id, time_points=[0.0], flow_rates=[0.1])
        for link in network.links.values()
        if link.from_node in origin_nodes
    ]

    # Pick a route: links along the diagonal from top-left to bottom-right
    # We'll use the first few links from the compiled network
    def get_corridor_route(engine):
        """Build a route from one corner to the opposite."""
        net = engine.net
        link_ids = sorted(net.link_cells.keys())
        # Simple: pick every other link to simulate a diagonal-ish corridor
        route = link_ids[:min(7, len(link_ids))]
        return route

    controllers = {
        "FixedTime": FixedTimeController(),
        "MaxPressure": MaxPressureController(),
    }

    print("=" * 60)
    print("LightSim EV Corridor Example")
    print("=" * 60)
    print(f"Grid: {rows}x{cols} ({rows * cols} intersections)")
    print(f"Time step: {dt}s")
    print()

    results = {}
    for name, ctrl in controllers.items():
        engine = SimulationEngine(
            network, dt=dt, controller=ctrl,
            demand_profiles=demand_profiles if demand_profiles else None,
        )
        engine.reset(seed=42)

        route = get_corridor_route(engine)
        ev = EVTracker(engine, route, speed_factor=1.5)
        ev.reset()

        max_steps = 200
        for step in range(max_steps):
            engine.step()
            ev.step()
            if ev.arrived:
                break

        results[name] = {
            "arrived": ev.arrived,
            "travel_time": ev.state.travel_time,
            "stops": ev.state.stops,
            "distance": ev.state.distance_traveled,
            "steps": step + 1,
            "total_vehicles": engine.get_total_vehicles(),
        }

        status = "ARRIVED" if ev.arrived else f"NOT arrived ({ev.fraction_completed:.0%} done)"
        print(f"[{name:15s}] {status}")
        print(f"  Travel time: {ev.state.travel_time:.1f}s | "
              f"Stops: {ev.state.stops} | "
              f"Distance: {ev.state.distance_traveled:.0f}m")
        print()

    # Summary comparison
    print("-" * 60)
    print("Summary:")
    print(f"{'Method':15s} {'ETT (s)':>10s} {'Stops':>6s} {'Arrived':>8s}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:15s} {r['travel_time']:10.1f} {r['stops']:6d} {'Yes' if r['arrived'] else 'No':>8s}")

    # Show how EVTracker observation API works
    print()
    print("EV Observation API demo:")
    engine = SimulationEngine(network, dt=dt, controller=FixedTimeController())
    engine.reset(seed=0)
    route = get_corridor_route(engine)
    ev = EVTracker(engine, route, speed_factor=1.5)
    ev.reset()

    engine.step()
    ev.step()
    obs = ev.get_ev_observation()
    print(f"  After 1 step: {obs}")

    print()
    print("Done! See lightsim.core.ev.EVTracker for the full API.")


if __name__ == "__main__":
    main()
