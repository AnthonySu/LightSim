"""Speed benchmark: measure LightSim throughput at various network sizes.

Measures simulation steps per second for different network configurations,
demonstrating the scalability of the vectorised CTM engine.

Usage::

    python -m lightsim.benchmarks.speed_benchmark
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..core.demand import DemandProfile
from ..core.engine import SimulationEngine
from ..core.signal import FixedTimeController
from ..core.types import LinkID, NodeID, NodeType
from ..networks.grid import create_grid_network
from ..networks.arterial import create_arterial_network


@dataclass
class BenchmarkResult:
    name: str
    n_cells: int
    n_intersections: int
    n_steps: int
    wall_time: float
    steps_per_sec: float
    sim_seconds: float
    speedup_vs_realtime: float


def benchmark_scenario(
    name: str,
    network,
    demand: list[DemandProfile],
    n_steps: int = 10000,
    dt: float = 1.0,
) -> BenchmarkResult:
    """Benchmark a single scenario."""
    engine = SimulationEngine(
        network=network,
        dt=dt,
        controller=FixedTimeController(),
        demand_profiles=demand,
    )
    engine.reset(seed=42)

    # Count signalised nodes
    n_int = sum(
        1 for n in network.nodes.values()
        if n.node_type == NodeType.SIGNALIZED
    )

    # Warm-up
    for _ in range(100):
        engine.step()

    engine.reset(seed=42)

    # Timed run
    t0 = time.perf_counter()
    for _ in range(n_steps):
        engine.step()
    t1 = time.perf_counter()

    wall = t1 - t0
    steps_per_sec = n_steps / wall
    sim_seconds = n_steps * dt
    speedup = sim_seconds / wall

    return BenchmarkResult(
        name=name,
        n_cells=engine.net.n_cells,
        n_intersections=n_int,
        n_steps=n_steps,
        wall_time=wall,
        steps_per_sec=steps_per_sec,
        sim_seconds=sim_seconds,
        speedup_vs_realtime=speedup,
    )


def run_all(n_steps: int = 10000) -> list[BenchmarkResult]:
    """Run benchmarks across various network sizes."""
    results = []

    # --- Single intersection ---
    from ..benchmarks.single_intersection import create_single_intersection
    net, demand = create_single_intersection()
    results.append(benchmark_scenario("single-intersection", net, demand, n_steps))

    # --- Grids ---
    for rows, cols in [(2, 2), (4, 4), (6, 6), (8, 8)]:
        net = create_grid_network(rows=rows, cols=cols, n_cells_per_link=3)
        demand = []
        for link in net.links.values():
            from_node = net.nodes.get(link.from_node)
            if from_node and from_node.node_type == NodeType.ORIGIN:
                demand.append(DemandProfile(link.link_id, [0.0], [0.1]))
        name = f"grid-{rows}x{cols}"
        results.append(benchmark_scenario(name, net, demand, n_steps))

    # --- Arterials ---
    for n in [3, 5, 10, 20]:
        net = create_arterial_network(n_intersections=n, n_cells_per_link=4)
        demand = []
        for link in net.links.values():
            from_node = net.nodes.get(link.from_node)
            if from_node and from_node.node_type == NodeType.ORIGIN:
                demand.append(DemandProfile(link.link_id, [0.0], [0.1]))
        name = f"arterial-{n}"
        results.append(benchmark_scenario(name, net, demand, n_steps))

    return results


def print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results as a formatted table."""
    header = (
        f"{'Scenario':<22} {'Cells':>6} {'Intx':>5} {'Steps':>7} "
        f"{'Wall(s)':>8} {'Steps/s':>10} {'Speedup':>8}"
    )
    print("=" * len(header))
    print("LightSim Speed Benchmark")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:<22} {r.n_cells:>6} {r.n_intersections:>5} {r.n_steps:>7} "
            f"{r.wall_time:>8.3f} {r.steps_per_sec:>10.0f} {r.speedup_vs_realtime:>7.0f}x"
        )
    print("-" * len(header))


def main():
    results = run_all()
    print_results(results)


if __name__ == "__main__":
    main()
