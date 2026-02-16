"""Cross-validation with mesoscopic extensions: LightSim vs SUMO.

Runs a comprehensive ablation across:
  - Scenarios: single-intersection, grid-4x4
  - Simulator modes: default (deterministic) vs mesoscopic (stochastic + lost_time)
  - Controllers: FixedTime, Webster, SOTL, MaxPressure variants, LT-Aware-MP
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
from lightsim.core.types import NodeType
from lightsim.utils.metrics import compute_link_delay


@dataclass
class MesoResult:
    scenario: str
    mode: str
    simulator: str
    controller: str
    total_exited: float
    avg_delay: float
    total_queue: float
    wall_time: float


def _build_scenario(scenario: str, lost_time: float = 0.0):
    from lightsim.benchmarks.scenarios import get_scenario
    factory = get_scenario(scenario)
    network, demand = factory()
    if lost_time > 0.0:
        for node in network.nodes.values():
            for phase in node.phases:
                phase.lost_time = lost_time
    return network, demand


def _run_lightsim(
    scenario: str,
    mode: str,
    controller,
    ctrl_label: str,
    lost_time: float = 0.0,
    stochastic: bool = False,
    n_steps: int = 3600,
    n_seeds: int = 5,
) -> MesoResult:
    network, demand = _build_scenario(scenario, lost_time=lost_time)

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
        scenario=scenario,
        mode=mode,
        simulator="LightSim",
        controller=ctrl_label,
        total_exited=float(np.mean(all_exited)),
        avg_delay=float(np.mean(all_delay)),
        total_queue=float(np.mean(all_queue)),
        wall_time=total_wall / n_seeds,
    )


def _try_run_sumo(scenario: str, controller_name: str) -> MesoResult | None:
    try:
        from lightsim.benchmarks.cross_validation import _run_sumo_with_controller
        r = _run_sumo_with_controller(scenario, controller_name)
        return MesoResult(
            scenario=scenario,
            mode="sumo",
            simulator="SUMO",
            controller=controller_name,
            total_exited=r.total_exited,
            avg_delay=r.avg_delay,
            total_queue=r.total_queue,
            wall_time=r.wall_time,
        )
    except Exception as e:
        print(f" SUMO failed: {e}")
        return None


def _print_table(results: list[MesoResult], title: str) -> None:
    print(f"\n{'=' * 100}")
    print(title)
    print("=" * 100)
    header = (f"{'Scenario':<22} {'Mode':<12} {'Sim':<10} {'Controller':<26} "
              f"{'Exited':>8} {'Delay':>8} {'Queue':>8}")
    print(header)
    print("-" * 100)
    for r in results:
        print(f"{r.scenario:<22} {r.mode:<12} {r.simulator:<10} {r.controller:<26} "
              f"{r.total_exited:>8.0f} {r.avg_delay:>8.2f} {r.total_queue:>8.1f}")


def _ranking_analysis(results: list[MesoResult], scenario: str) -> None:
    scen_r = [r for r in results if r.scenario == scenario]
    if not scen_r:
        return

    print(f"\n  --- {scenario} ---")
    for mode in ["default", "mesoscopic", "sumo"]:
        mode_r = sorted(
            [r for r in scen_r if r.mode == mode],
            key=lambda r: r.total_exited, reverse=True,
        )
        if not mode_r:
            continue
        sim = mode_r[0].simulator
        print(f"\n  [{mode.upper()}] ({sim})")
        for i, r in enumerate(mode_r):
            marker = " <-- best" if i == 0 else ""
            print(f"    {i+1}. {r.controller:<26}  "
                  f"exited={r.total_exited:>7.0f}  "
                  f"delay={r.avg_delay:>7.2f}  "
                  f"queue={r.total_queue:>7.1f}{marker}")


def run_scenario(scenario: str, results: list[MesoResult]) -> None:
    """Run full ablation for one scenario."""

    # Focused controller set (drop EfficientMP which needs tuning)
    controllers = [
        ("FixedTime-30s",      FixedTimeController()),
        ("Webster",            WebsterController()),
        ("SOTL",               SOTLController()),
        ("MaxPressure-mg5",    MaxPressureController(min_green=5.0)),
        ("MaxPressure-mg10",   MaxPressureController(min_green=10.0)),
        ("MaxPressure-mg15",   MaxPressureController(min_green=15.0)),
        ("LT-Aware-MP-mg5",   LostTimeAwareMaxPressureController(min_green=5.0)),
        ("LT-Aware-MP-mg10",  LostTimeAwareMaxPressureController(min_green=10.0)),
    ]

    print(f"\n{'#' * 100}")
    print(f"  SCENARIO: {scenario}")
    print(f"{'#' * 100}")

    # 1. Default
    print(f"\n  [DEFAULT] deterministic, lost_time=0")
    for label, ctrl in controllers:
        print(f"    {label}...", end="", flush=True)
        r = _run_lightsim(scenario, "default", ctrl, label,
                          lost_time=0.0, stochastic=False, n_seeds=1)
        results.append(r)
        print(f"  exited={r.total_exited:.0f}  delay={r.avg_delay:.2f}  "
              f"queue={r.total_queue:.1f}  ({r.wall_time:.1f}s)")

    # 2. Mesoscopic
    print(f"\n  [MESOSCOPIC] stochastic=True, lost_time=2.0")
    for label, ctrl in controllers:
        print(f"    {label} (5-seed)...", end="", flush=True)
        r = _run_lightsim(scenario, "mesoscopic", ctrl, label,
                          lost_time=2.0, stochastic=True, n_seeds=5)
        results.append(r)
        print(f"  exited={r.total_exited:.0f}  delay={r.avg_delay:.2f}  "
              f"queue={r.total_queue:.1f}  ({r.wall_time:.1f}s)")

    # 3. SUMO
    print(f"\n  [SUMO]")
    for sumo_ctrl in ["FixedTimeController", "MaxPressureController"]:
        print(f"    {sumo_ctrl}...", end="", flush=True)
        r = _try_run_sumo(scenario, sumo_ctrl)
        if r is not None:
            results.append(r)
            print(f"  exited={r.total_exited:.0f}  delay={r.avg_delay:.2f}  "
                  f"queue={r.total_queue:.1f}")


def main():
    print("=" * 100)
    print("Mesoscopic Cross-Validation: LightSim vs SUMO")
    print("Single Intersection + Grid 4x4")
    print("=" * 100)

    results: list[MesoResult] = []

    for scenario in ["single-intersection-v0", "grid-4x4-v0"]:
        run_scenario(scenario, results)

    # ---------------------------------------------------------------
    # Summary tables
    # ---------------------------------------------------------------
    _print_table(
        [r for r in results if r.scenario == "single-intersection-v0"],
        "Single Intersection Results",
    )
    _print_table(
        [r for r in results if r.scenario == "grid-4x4-v0"],
        "Grid 4x4 Results",
    )

    # ---------------------------------------------------------------
    # Rankings
    # ---------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print("Controller Rankings (sorted by throughput)")
    print("=" * 100)
    for scenario in ["single-intersection-v0", "grid-4x4-v0"]:
        _ranking_analysis(results, scenario)

    # ---------------------------------------------------------------
    # Cross-scenario consistency
    # ---------------------------------------------------------------
    print(f"\n{'=' * 100}")
    print("Cross-Scenario Summary: Best controller per mode")
    print("=" * 100)
    for scenario in ["single-intersection-v0", "grid-4x4-v0"]:
        print(f"\n  {scenario}:")
        for mode in ["default", "mesoscopic", "sumo"]:
            mode_r = [r for r in results if r.scenario == scenario and r.mode == mode]
            if mode_r:
                best = max(mode_r, key=lambda r: r.total_exited)
                print(f"    {mode:<12} -> {best.controller:<26} "
                      f"(exited={best.total_exited:.0f}, delay={best.avg_delay:.2f})")

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
