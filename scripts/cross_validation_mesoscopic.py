"""Cross-validation with mesoscopic extensions: LightSim vs SUMO.

Runs a comprehensive ablation across:
  - Simulator modes: default (deterministic) vs mesoscopic (stochastic + lost_time)
  - Controllers: FixedTime, Webster, SOTL, MaxPressure (vanilla + tuned + lost-time-aware + EMP)
  - SUMO reference for FixedTime and actuated (MaxPressure proxy)

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

from lightsim.core.engine import SimulationEngine
from lightsim.core.signal import (
    EfficientMaxPressureController,
    FixedTimeController,
    LostTimeAwareMaxPressureController,
    MaxPressureController,
    SOTLController,
    WebsterController,
)
from lightsim.core.types import NodeID, NodeType
from lightsim.utils.metrics import compute_link_delay


@dataclass
class MesoResult:
    mode: str
    simulator: str
    controller: str
    total_exited: float
    avg_delay: float
    total_queue: float
    wall_time: float


def _build_single_intersection(lost_time: float = 0.0):
    from lightsim.benchmarks.scenarios import get_scenario
    factory = get_scenario("single-intersection-v0")
    network, demand = factory()
    if lost_time > 0.0:
        for node in network.nodes.values():
            for phase in node.phases:
                phase.lost_time = lost_time
    return network, demand


def _run_lightsim(
    mode: str,
    controller,
    ctrl_label: str,
    lost_time: float = 0.0,
    stochastic: bool = False,
    n_steps: int = 3600,
    n_seeds: int = 5,
) -> MesoResult:
    network, demand = _build_single_intersection(lost_time=lost_time)

    all_exited, all_delay, all_queue = [], [], []
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
        controller=ctrl_label,
        total_exited=float(np.mean(all_exited)),
        avg_delay=float(np.mean(all_delay)),
        total_queue=float(np.mean(all_queue)),
        wall_time=total_wall / n_seeds,
    )


def _try_run_sumo(controller_name: str) -> MesoResult | None:
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
        print(f"    SUMO unavailable: {e}")
        return None


def _print_table(results: list[MesoResult], title: str) -> None:
    print(f"\n{'=' * 90}")
    print(title)
    print("=" * 90)
    header = (f"{'Mode':<14} {'Sim':<10} {'Controller':<30} "
              f"{'Exited':>8} {'Delay':>8} {'Queue':>8}")
    print(header)
    print("-" * 90)
    for r in results:
        print(f"{r.mode:<14} {r.simulator:<10} {r.controller:<30} "
              f"{r.total_exited:>8.0f} {r.avg_delay:>8.2f} {r.total_queue:>8.1f}")


def main():
    print("=" * 90)
    print("Mesoscopic Cross-Validation + Controller Ablation")
    print("=" * 90)

    # ---------------------------------------------------------------
    # Define all controllers
    # ---------------------------------------------------------------
    controllers = [
        # --- Fixed-time family ---
        ("FixedTime-30s",           FixedTimeController()),
        ("Webster",                 WebsterController()),
        # --- Adaptive family ---
        ("SOTL",                    SOTLController()),
        ("MaxPressure-mg5",         MaxPressureController(min_green=5.0)),
        ("MaxPressure-mg10",        MaxPressureController(min_green=10.0)),
        ("MaxPressure-mg15",        MaxPressureController(min_green=15.0)),
        ("LT-Aware-MP-mg5",        LostTimeAwareMaxPressureController(min_green=5.0)),
        ("LT-Aware-MP-mg10",       LostTimeAwareMaxPressureController(min_green=10.0)),
        ("EfficientMP",            EfficientMaxPressureController()),
    ]

    results: list[MesoResult] = []

    # ---------------------------------------------------------------
    # 1. LightSim DEFAULT (deterministic, no lost_time)
    # ---------------------------------------------------------------
    print("\n[1/3] LightSim DEFAULT (deterministic, lost_time=0)")
    for label, ctrl in controllers:
        print(f"  {label}...", end="", flush=True)
        r = _run_lightsim("default", ctrl, label,
                          lost_time=0.0, stochastic=False, n_seeds=1)
        results.append(r)
        print(f"  exited={r.total_exited:.0f}  delay={r.avg_delay:.2f}  "
              f"queue={r.total_queue:.1f}")

    # ---------------------------------------------------------------
    # 2. LightSim MESOSCOPIC (stochastic, lost_time=2.0)
    # ---------------------------------------------------------------
    print("\n[2/3] LightSim MESOSCOPIC (stochastic=True, lost_time=2.0)")
    for label, ctrl in controllers:
        print(f"  {label} (5-seed avg)...", end="", flush=True)
        r = _run_lightsim("mesoscopic", ctrl, label,
                          lost_time=2.0, stochastic=True, n_seeds=5)
        results.append(r)
        print(f"  exited={r.total_exited:.0f}  delay={r.avg_delay:.2f}  "
              f"queue={r.total_queue:.1f}")

    # ---------------------------------------------------------------
    # 3. SUMO reference
    # ---------------------------------------------------------------
    print("\n[3/3] SUMO reference")
    for sumo_ctrl in ["FixedTimeController", "MaxPressureController"]:
        print(f"  {sumo_ctrl}...", end="", flush=True)
        r = _try_run_sumo(sumo_ctrl)
        if r is not None:
            results.append(r)
            print(f"  exited={r.total_exited:.0f}  delay={r.avg_delay:.2f}  "
                  f"queue={r.total_queue:.1f}")

    # ---------------------------------------------------------------
    # Print tables
    # ---------------------------------------------------------------
    _print_table(results, "All Results")

    # ---------------------------------------------------------------
    # Ranking analysis
    # ---------------------------------------------------------------
    print(f"\n{'=' * 90}")
    print("Controller Ranking (sorted by throughput within each mode)")
    print("=" * 90)

    for mode in ["default", "mesoscopic", "sumo"]:
        mode_r = sorted(
            [r for r in results if r.mode == mode],
            key=lambda r: r.total_exited, reverse=True,
        )
        if not mode_r:
            continue
        sim = mode_r[0].simulator
        print(f"\n  [{mode.upper()}] ({sim})")
        for i, r in enumerate(mode_r):
            marker = " <-- best" if i == 0 else ""
            print(f"    {i+1}. {r.controller:<30}  "
                  f"exited={r.total_exited:>7.0f}  "
                  f"delay={r.avg_delay:>7.2f}  "
                  f"queue={r.total_queue:>7.1f}{marker}")

    # ---------------------------------------------------------------
    # Key insight: does mesoscopic flip any rankings?
    # ---------------------------------------------------------------
    print(f"\n{'=' * 90}")
    print("Key Question: Does mesoscopic mode change the best adaptive controller?")
    print("=" * 90)

    adaptive_labels = [l for l, _ in controllers if l not in ("FixedTime-30s", "Webster")]
    for mode in ["default", "mesoscopic"]:
        mode_adaptive = [r for r in results if r.mode == mode and r.controller in adaptive_labels]
        if mode_adaptive:
            best = max(mode_adaptive, key=lambda r: r.total_exited)
            ft = next((r for r in results if r.mode == mode and r.controller == "FixedTime-30s"), None)
            if ft:
                gap = (best.total_exited - ft.total_exited) / ft.total_exited * 100
                direction = "beats" if gap > 0 else "loses to"
                print(f"  {mode}: best adaptive = {best.controller}  "
                      f"({direction} FixedTime by {abs(gap):.1f}%)")

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    out_path = "results/cross_validation_mesoscopic.json"
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
