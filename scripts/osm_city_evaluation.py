"""OSM City Scenario Evaluation: controller rankings across diverse real-world networks.

Tests 5 controllers on 6 OSM city scenarios to show consistent algorithmic
rankings across diverse topologies (17-131 signals, 4 continents).

Usage::
    python scripts/osm_city_evaluation.py
    python scripts/osm_city_evaluation.py --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lightsim.benchmarks.scenarios import get_scenario
from lightsim.core.engine import SimulationEngine
from lightsim.core.signal import (
    FixedTimeController,
    GreenWaveController,
    MaxPressureController,
    SOTLController,
    WebsterController,
)
from lightsim.core.types import NodeType
from lightsim.utils.metrics import compute_link_delay

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CITIES = [
    "osm-manhattan-v0",
    "osm-shanghai-v0",
    "osm-london-v0",
    "osm-sanfrancisco-v0",
    "osm-mumbai-v0",
    "osm-siouxfalls-v0",
]

CONTROLLERS = [
    ("FixedTime", lambda: FixedTimeController()),
    ("Webster", lambda: WebsterController()),
    ("MaxPressure", lambda: MaxPressureController(min_green=15.0)),
    ("SOTL", lambda: SOTLController()),
    ("GreenWave", lambda: GreenWaveController()),
]

EPISODE_STEPS = 3600
SEEDS = [0, 1, 2]

RESULTS_DIR = Path("results")
RESULTS_FILE = RESULTS_DIR / "osm_city_evaluation.json"
CHECKPOINT_FILE = RESULTS_DIR / "osm_city_eval_checkpoint.json"


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------
def run_single(scenario_name: str, controller, seed: int,
               episode_steps: int = EPISODE_STEPS) -> dict:
    """Run one controller on one city scenario with one seed."""
    factory = get_scenario(scenario_name)
    network, demand = factory()

    engine = SimulationEngine(
        network=network, dt=1.0,
        controller=controller,
        demand_profiles=demand,
        stochastic=True,
    )
    engine.reset(seed=seed)

    t0 = time.perf_counter()
    for _ in range(episode_steps):
        engine.step()
    wall = time.perf_counter() - t0

    metrics = engine.get_network_metrics()

    # Compute total queue across signalized links
    total_queue = 0.0
    n_signals = 0
    for node in network.nodes.values():
        if node.node_type == NodeType.SIGNALIZED:
            n_signals += 1
    for link in network.links.values():
        to_node = network.nodes.get(link.to_node)
        if to_node and to_node.node_type == NodeType.SIGNALIZED:
            total_queue += engine.get_link_queue(link.link_id)

    return {
        "scenario": scenario_name,
        "controller": type(controller).__name__,
        "seed": seed,
        "n_signals": n_signals,
        "total_exited": float(metrics["total_exited"]),
        "total_queue": round(total_queue, 2),
        "wall_time": round(wall, 3),
    }


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------
def load_checkpoint() -> list[dict]:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return []


def save_checkpoint(results: list[dict]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(results, f, indent=2)


def is_completed(results: list[dict], scenario: str, ctrl: str, seed: int) -> bool:
    return any(
        r["scenario"] == scenario and r["controller"] == ctrl and r["seed"] == seed
        for r in results
    )


# ---------------------------------------------------------------------------
# Ranking analysis
# ---------------------------------------------------------------------------
def compute_rankings(results: list[dict]) -> dict:
    """Compute controller rankings per city + Kendall's tau across cities."""
    from scipy.stats import kendalltau

    # Group by city: {city: {controller: [throughputs across seeds]}}
    city_ctrl = {}
    for r in results:
        if "error" in r:
            continue
        city_ctrl.setdefault(r["scenario"], {}).setdefault(
            r["controller"], []).append(r["total_exited"])

    # Rank controllers per city by mean throughput (higher = better)
    city_rankings = {}
    for city, ctrls in city_ctrl.items():
        means = {c: np.mean(vals) for c, vals in ctrls.items()}
        sorted_ctrls = sorted(means.keys(), key=lambda c: -means[c])
        city_rankings[city] = {c: rank + 1 for rank, c in enumerate(sorted_ctrls)}

    # Pairwise Kendall's tau between all city pairs
    cities = sorted(city_rankings.keys())
    shared_ctrls = sorted(
        set.intersection(*[set(city_rankings[c].keys()) for c in cities])
    )
    if len(shared_ctrls) < 3 or len(cities) < 2:
        return {"city_rankings": city_rankings, "pairwise_tau": [], "mean_tau": None}

    taus = []
    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            r1 = [city_rankings[cities[i]][c] for c in shared_ctrls]
            r2 = [city_rankings[cities[j]][c] for c in shared_ctrls]
            tau, p = kendalltau(r1, r2)
            taus.append({
                "city_a": cities[i],
                "city_b": cities[j],
                "kendall_tau": round(tau, 3),
                "p_value": round(p, 4),
            })

    mean_tau = float(np.mean([t["kendall_tau"] for t in taus]))

    return {
        "city_rankings": city_rankings,
        "shared_controllers": shared_ctrls,
        "pairwise_tau": taus,
        "mean_tau": round(mean_tau, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    results = load_checkpoint() if args.resume else []
    if results:
        print(f"Resuming: {len(results)} runs already completed")

    total_runs = len(CITIES) * len(CONTROLLERS) * len(SEEDS)
    done = len(results)

    print("=" * 70)
    print("OSM City Scenario Evaluation")
    print(f"Cities: {len(CITIES)}, Controllers: {len(CONTROLLERS)}, "
          f"Seeds: {len(SEEDS)}")
    print(f"Total runs: {total_runs}, Already done: {done}")
    print("=" * 70)

    for city in CITIES:
        print(f"\n{'#' * 60}")
        print(f"  CITY: {city}")
        print(f"{'#' * 60}")

        for ctrl_name, ctrl_factory in CONTROLLERS:
            for seed in SEEDS:
                if is_completed(results, city, ctrl_name.replace(
                    "FixedTime", "FixedTimeController").replace(
                    "Webster", "WebsterController").replace(
                    "MaxPressure", "MaxPressureController").replace(
                    "SOTL", "SOTLController").replace(
                    "GreenWave", "GreenWaveController"), seed):
                    done += 0  # already counted
                    continue
                # Also check short name
                ctrl_cls_name = ctrl_factory().__class__.__name__
                if is_completed(results, city, ctrl_cls_name, seed):
                    continue

                print(f"  {ctrl_name} seed={seed}...", end="", flush=True)
                try:
                    controller = ctrl_factory()
                    r = run_single(city, controller, seed)
                    results.append(r)
                    save_checkpoint(results)
                    done += 1
                    print(f"  exited={r['total_exited']:.0f}, "
                          f"queue={r['total_queue']:.1f}, "
                          f"wall={r['wall_time']:.1f}s "
                          f"[{done}/{total_runs}]")
                except Exception as e:
                    print(f"  FAILED: {e}")
                    results.append({
                        "scenario": city,
                        "controller": ctrl_cls_name,
                        "seed": seed,
                        "error": str(e),
                    })
                    save_checkpoint(results)
                    done += 1

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    valid = [r for r in results if "error" not in r]

    print(f"\n\n{'=' * 80}")
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'City':<25} {'Controller':<22} {'Throughput':>12} {'Queue':>10}")
    print("-" * 72)

    for city in CITIES:
        city_results = [r for r in valid if r["scenario"] == city]
        # Mean across seeds
        ctrl_stats = {}
        for r in city_results:
            ctrl_stats.setdefault(r["controller"], []).append(
                (r["total_exited"], r["total_queue"]))
        for ctrl, vals in sorted(ctrl_stats.items()):
            throughputs = [v[0] for v in vals]
            queues = [v[1] for v in vals]
            print(f"{city:<25} {ctrl:<22} "
                  f"{np.mean(throughputs):>8.0f}+/-{np.std(throughputs):>4.0f} "
                  f"{np.mean(queues):>8.1f}")

    # Rankings
    print(f"\n{'=' * 80}")
    print("RANKING ANALYSIS")
    print("=" * 80)
    ranking_info = compute_rankings(valid)

    for city, ranks in ranking_info["city_rankings"].items():
        print(f"\n  [{city}]")
        for ctrl, rank in sorted(ranks.items(), key=lambda x: x[1]):
            print(f"    {rank}. {ctrl}")

    if ranking_info["pairwise_tau"]:
        print(f"\n  Pairwise Kendall's tau:")
        for t in ranking_info["pairwise_tau"]:
            city_a = t["city_a"].replace("osm-", "").replace("-v0", "")
            city_b = t["city_b"].replace("osm-", "").replace("-v0", "")
            print(f"    {city_a:<15} vs {city_b:<15} tau={t['kendall_tau']:.3f} "
                  f"(p={t['p_value']:.4f})")
        print(f"\n  Mean Kendall's tau: {ranking_info['mean_tau']:.3f}")

    # Save final results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "osm_city_evaluation",
        "cities": CITIES,
        "controllers": [c[0] for c in CONTROLLERS],
        "seeds": SEEDS,
        "episode_steps": EPISODE_STEPS,
        "results": valid,
        "rankings": ranking_info,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
