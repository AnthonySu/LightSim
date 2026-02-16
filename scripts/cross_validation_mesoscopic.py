"""Cross-validation with mesoscopic extensions: LightSim vs SUMO.

Compares two LightSim configurations against SUMO:
  1. Default: deterministic demand, no lost_time
  2. Mesoscopic: stochastic=True, lost_time=2.0

Checks whether mesoscopic extensions bring controller ranking
(FixedTime vs MaxPressure) and absolute metrics closer to SUMO.

Usage::

    python scripts/cross_validation_mesoscopic.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lightsim.core.demand import DemandProfile
from lightsim.core.engine import SimulationEngine
from lightsim.core.network import Network
from lightsim.core.signal import FixedTimeController, MaxPressureController
from lightsim.core.types import LinkID, NodeID, NodeType, TurnType
from lightsim.utils.metrics import compute_link_delay


@dataclass
class MesoResult:
    mode: str          # "default" or "mesoscopic"
    simulator: str     # "LightSim" or "SUMO"
    controller: str
    total_exited: float
    avg_delay: float
    total_queue: float
    wall_time: float


def _build_single_intersection(lost_time: float = 0.0):
    """Build single intersection network with configurable lost_time."""
    from lightsim.benchmarks.scenarios import get_scenario
    factory = get_scenario("single-intersection-v0")
    network, demand = factory()

    # Patch lost_time on all phases if non-zero
    if lost_time > 0.0:
        for node in network.nodes.values():
            for phase in node.phases:
                phase.lost_time = lost_time

    return network, demand


def _run_lightsim(
    mode: str,
    controller,
    lost_time: float = 0.0,
    stochastic: bool = False,
    n_steps: int = 3600,
    n_seeds: int = 5,
) -> MesoResult:
    """Run LightSim and average over seeds (only matters for stochastic)."""
    network, demand = _build_single_intersection(lost_time=lost_time)

    all_exited = []
    all_delay = []
    all_queue = []
    total_wall = 0.0

    for seed in range(n_seeds):
        engine = SimulationEngine(
            network=network, dt=1.0,
            controller=controller,
            demand_profiles=demand,
            stochastic=stochastic,
        )
        engine.reset(seed=seed)

        t0 = time.perf_counter()
        for _ in range(n_steps):
            engine.step()
        total_wall += time.perf_counter() - t0

        metrics = engine.get_network_metrics()
        all_exited.append(metrics["total_exited"])

        total_delay = 0.0
        total_queue = 0.0
        n_links = 0
        for link in network.links.values():
            to_node = network.nodes.get(link.to_node)
            if to_node and to_node.node_type == NodeType.SIGNALIZED:
                total_delay += compute_link_delay(engine, link.link_id)
                total_queue += engine.get_link_queue(link.link_id)
                n_links += 1
        all_delay.append(total_delay / max(n_links, 1))
        all_queue.append(total_queue)

    return MesoResult(
        mode=mode,
        simulator="LightSim",
        controller=type(controller).__name__,
        total_exited=float(np.mean(all_exited)),
        avg_delay=float(np.mean(all_delay)),
        total_queue=float(np.mean(all_queue)),
        wall_time=total_wall / n_seeds,
    )


def _try_run_sumo(controller_name: str) -> MesoResult | None:
    """Attempt to run SUMO cross-validation. Returns None if SUMO unavailable."""
    try:
        from lightsim.benchmarks.cross_validation import _run_sumo_with_controller
        r = _run_sumo_with_controller("single-intersection-v0", controller_name)
        return MesoResult(
            mode="sumo",
            simulator="SUMO",
            controller=controller_name,
            total_exited=r.total_exited,
            avg_delay=r.avg_delay,
            total_queue=r.total_queue,
            wall_time=r.wall_time,
        )
    except Exception as e:
        print(f"  SUMO unavailable: {e}")
        return None


def main():
    print("=" * 80)
    print("Mesoscopic Cross-Validation: LightSim vs SUMO")
    print("=" * 80)

    controllers = [
        ("FixedTimeController", FixedTimeController()),
        ("MaxPressureController", MaxPressureController(min_green=5.0)),
    ]

    results: list[MesoResult] = []

    # --- LightSim default mode ---
    print("\n[1/3] LightSim DEFAULT (deterministic, no lost_time)")
    for ctrl_name, ctrl_obj in controllers:
        print(f"  Running {ctrl_name}...")
        r = _run_lightsim("default", ctrl_obj, lost_time=0.0, stochastic=False)
        results.append(r)
        print(f"    exited={r.total_exited:.0f}, delay={r.avg_delay:.2f}, "
              f"queue={r.total_queue:.1f}")

    # --- LightSim mesoscopic mode ---
    print("\n[2/3] LightSim MESOSCOPIC (stochastic=True, lost_time=2.0)")
    for ctrl_name, ctrl_obj in controllers:
        print(f"  Running {ctrl_name} (5-seed average)...")
        r = _run_lightsim("mesoscopic", ctrl_obj, lost_time=2.0, stochastic=True)
        results.append(r)
        print(f"    exited={r.total_exited:.0f}, delay={r.avg_delay:.2f}, "
              f"queue={r.total_queue:.1f}")

    # --- SUMO ---
    print("\n[3/3] SUMO reference")
    for ctrl_name, _ in controllers:
        print(f"  Running {ctrl_name}...")
        r = _try_run_sumo(ctrl_name)
        if r is not None:
            results.append(r)
            print(f"    exited={r.total_exited:.0f}, delay={r.avg_delay:.2f}, "
                  f"queue={r.total_queue:.1f}")

    # --- Summary ---
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    header = (f"{'Mode':<14} {'Simulator':<10} {'Controller':<22} "
              f"{'Exited':>8} {'Delay':>8} {'Queue':>8}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.mode:<14} {r.simulator:<10} {r.controller:<22} "
              f"{r.total_exited:>8.0f} {r.avg_delay:>8.2f} {r.total_queue:>8.1f}")

    # --- Controller ranking analysis ---
    print("\n" + "-" * 80)
    print("Controller Ranking Analysis")
    print("-" * 80)

    for mode in ["default", "mesoscopic"]:
        mode_results = [r for r in results if r.mode == mode and r.simulator == "LightSim"]
        if len(mode_results) == 2:
            ft = next(r for r in mode_results if "Fixed" in r.controller)
            mp = next(r for r in mode_results if "MaxPressure" in r.controller)
            better = "MaxPressure" if mp.total_exited > ft.total_exited else "FixedTime"
            diff_pct = abs(mp.total_exited - ft.total_exited) / max(ft.total_exited, 1) * 100
            print(f"  LightSim {mode}: {better} wins by {diff_pct:.1f}% throughput")

    sumo_results = [r for r in results if r.simulator == "SUMO"]
    if len(sumo_results) == 2:
        ft = next(r for r in sumo_results if "Fixed" in r.controller)
        mp = next(r for r in sumo_results if "MaxPressure" in r.controller)
        better = "MaxPressure" if mp.total_exited > ft.total_exited else "FixedTime"
        diff_pct = abs(mp.total_exited - ft.total_exited) / max(ft.total_exited, 1) * 100
        print(f"  SUMO:             {better} wins by {diff_pct:.1f}% throughput")

    # --- Save results ---
    os.makedirs("results", exist_ok=True)
    out_path = "results/cross_validation_mesoscopic.json"
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
