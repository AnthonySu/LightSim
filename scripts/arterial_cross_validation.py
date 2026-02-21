"""Arterial Cross-Validation: LightSim vs SUMO on 5-intersection arterial corridor.

Shows controller ranking preservation on arterial networks, where
GreenWave coordination is expected to excel.

Usage::
    python scripts/arterial_cross_validation.py
    python scripts/arterial_cross_validation.py --lightsim-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lightsim.benchmarks.cross_validation import (
    CrossValResult,
    _get_time_in_phase,
    _is_green_phase,
    _phase_entry,
    _run_lightsim,
    _traci_maxpressure_step,
    _traci_sotl_step,
)
from lightsim.benchmarks.sumo_comparison import _find_sumo_binary, _write_arterial
from lightsim.core.signal import (
    FixedTimeController,
    GreenWaveController,
    LostTimeAwareMaxPressureController,
    MaxPressureController,
    SOTLController,
    WebsterController,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCENARIO = "arterial-5-v0"
SIM_SECONDS = 3600
SEED = 42

RESULTS_DIR = Path("results")
RESULTS_FILE = RESULTS_DIR / "arterial_cross_validation.json"

# SUMO_HOME setup (same pattern as rl_cross_validation.py)
def _set_sumo_home():
    if os.environ.get("SUMO_HOME"):
        return
    try:
        import importlib.util
        spec = importlib.util.find_spec("sumolib")
        if spec and spec.origin:
            site_packages = Path(spec.origin).parent.parent
            sumo_home = site_packages / "sumo"
            if sumo_home.exists():
                os.environ["SUMO_HOME"] = str(sumo_home)
    except Exception:
        pass

_set_sumo_home()


# ---------------------------------------------------------------------------
# LightSim controllers
# ---------------------------------------------------------------------------
LIGHTSIM_CONTROLLERS = [
    ("FixedTimeController", FixedTimeController()),
    ("WebsterController", WebsterController()),
    ("MaxPressureController", MaxPressureController(min_green=15.0)),
    ("SOTLController", SOTLController()),
    ("GreenWaveController", GreenWaveController()),
    ("LT-Aware-MP", LostTimeAwareMaxPressureController(min_green=5.0)),
]

# SUMO controllers
SUMO_CONTROLLERS = [
    "FixedTimeController",
    "ActuatedController",
    "MaxPressureController",
    "SOTLController",
]


# ---------------------------------------------------------------------------
# SUMO arterial runner
# ---------------------------------------------------------------------------
def _run_sumo_arterial(controller_name: str,
                       sim_seconds: int = SIM_SECONDS) -> CrossValResult:
    """Run SUMO on arterial-5 with a specified controller."""
    sumo_bin = _find_sumo_binary()

    traci_controller = controller_name in ("MaxPressureController", "SOTLController")
    sumo_tls_type = controller_name
    if traci_controller:
        sumo_tls_type = "FixedTimeController"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        cfg = _write_arterial(tmppath, n_intersections=5,
                              sim_seconds=sim_seconds, step_length=1.0)

        # If actuated, rebuild with actuated TLS
        if controller_name == "ActuatedController":
            # Re-generate with actuated TLS type
            _rebuild_arterial_actuated(tmppath, sumo_bin, sim_seconds)
            cfg = tmppath / "sim.sumocfg"

        try:
            import traci
        except ImportError:
            raise ImportError("traci required: pip install traci")

        t0 = time.perf_counter()
        _phase_entry.clear()
        traci.start([sumo_bin, "-c", str(cfg), "--no-warnings", "true"])

        total_arrived = 0
        for step in range(sim_seconds):
            if traci_controller and step > 0:
                if controller_name == "MaxPressureController":
                    _traci_maxpressure_step(traci, min_green=15.0)
                elif controller_name == "SOTLController":
                    _traci_sotl_step(traci, min_green=10.0, max_green=60.0)

            traci.simulationStep()
            total_arrived += traci.simulation.getArrivedNumber()

        # Collect metrics
        total_exited = total_arrived
        avg_delay = 0.0
        try:
            vehicle_ids = traci.vehicle.getIDList()
            if vehicle_ids:
                delays = [traci.vehicle.getWaitingTime(vid)
                          for vid in vehicle_ids[:200]]
                avg_delay = float(np.mean(delays)) if delays else 0.0
        except Exception:
            pass

        total_queue = 0.0
        try:
            for vid in traci.vehicle.getIDList():
                if traci.vehicle.getSpeed(vid) < 0.1:
                    total_queue += 1
        except Exception:
            pass

        traci.close()
        wall = time.perf_counter() - t0

    return CrossValResult(
        simulator="SUMO",
        scenario=SCENARIO,
        controller=controller_name,
        total_exited=float(total_exited),
        avg_delay=avg_delay,
        total_queue=total_queue,
        wall_time=wall,
    )


def _rebuild_arterial_actuated(tmpdir: Path, sumo_bin: str, sim_seconds: int):
    """Rebuild arterial network with actuated TLS for SUMO actuated controller."""
    import subprocess
    import textwrap

    nod_xml = tmpdir / "nodes.nod.xml"
    edg_xml = tmpdir / "edges.edg.xml"
    net_xml = tmpdir / "net.net.xml"
    cfg_xml = tmpdir / "sim.sumocfg"

    n_intersections = 5
    spacing = 400

    # Rewrite nodes with actuated TLS
    nodes = ['<nodes>']
    nodes.append(f'    <node id="W" x="0" y="0" type="priority"/>')
    for i in range(1, n_intersections + 1):
        nodes.append(f'    <node id="I{i}" x="{i * spacing}" y="0" '
                     f'type="traffic_light" tlType="actuated"/>')
        nodes.append(f'    <node id="N{i}" x="{i * spacing}" y="200" type="priority"/>')
        nodes.append(f'    <node id="S{i}" x="{i * spacing}" y="-200" type="priority"/>')
    nodes.append(f'    <node id="E" x="{(n_intersections + 1) * spacing}" y="0" '
                 f'type="priority"/>')
    nodes.append('</nodes>')
    nod_xml.write_text("\n".join(nodes))

    # netconvert
    netconvert = str(Path(sumo_bin).parent / (
        "netconvert.exe" if sys.platform == "win32" else "netconvert"
    ))
    subprocess.run(
        [netconvert, "-n", str(nod_xml), "-e", str(edg_xml),
         "-o", str(net_xml), "--no-warnings", "true"],
        check=True, capture_output=True,
    )

    # Rewrite config
    rou_xml = tmpdir / "routes.rou.xml"
    cfg_xml.write_text(textwrap.dedent(f"""\
        <configuration>
            <input>
                <net-file value="{net_xml.name}"/>
                <route-files value="{rou_xml.name}"/>
            </input>
            <time>
                <begin value="0"/>
                <end value="{sim_seconds}"/>
                <step-length value="1.0"/>
            </time>
            <processing>
                <no-step-log value="true"/>
            </processing>
            <report>
                <no-warnings value="true"/>
                <verbose value="false"/>
            </report>
        </configuration>
    """))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lightsim-only", action="store_true")
    parser.add_argument("--sumo-only", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("Arterial Cross-Validation: LightSim vs SUMO (arterial-5-v0)")
    print("=" * 70)

    results = []

    # --- LightSim ---
    if not args.sumo_only:
        print(f"\n  [LightSim]")
        for ctrl_name, ctrl_obj in LIGHTSIM_CONTROLLERS:
            print(f"    {ctrl_name}...", end="", flush=True)
            try:
                r = _run_lightsim(SCENARIO, ctrl_obj, episode_steps=SIM_SECONDS)
                results.append(r)
                print(f"  exited={r.total_exited:.0f}, delay={r.avg_delay:.2f}, "
                      f"queue={r.total_queue:.1f}")
            except Exception as e:
                print(f"  FAILED: {e}")

    # --- SUMO ---
    if not args.lightsim_only:
        try:
            sumo_bin = _find_sumo_binary()
            print(f"\n  SUMO binary: {sumo_bin}")
        except FileNotFoundError as e:
            print(f"\n  SUMO not found: {e}")
            print("  Skipping SUMO runs.")
            sumo_bin = None

        if sumo_bin:
            print(f"\n  [SUMO]")
            for ctrl_name in SUMO_CONTROLLERS:
                print(f"    {ctrl_name}...", end="", flush=True)
                try:
                    r = _run_sumo_arterial(ctrl_name)
                    results.append(r)
                    print(f"  exited={r.total_exited:.0f}, delay={r.avg_delay:.2f}, "
                          f"queue={r.total_queue:.1f}")
                except Exception as e:
                    print(f"  FAILED: {e}")

    # --- Summary ---
    print(f"\n\n{'=' * 90}")
    print("ARTERIAL CROSS-VALIDATION SUMMARY")
    print("=" * 90)
    header = (f"{'Simulator':<12} {'Controller':<25} "
              f"{'Exited':>8} {'Delay':>8} {'Queue':>8}")
    print(header)
    print("-" * 65)
    for r in sorted(results, key=lambda x: (x.simulator, -x.total_exited)):
        print(f"{r.simulator:<12} {r.controller:<25} "
              f"{r.total_exited:>8.0f} {r.avg_delay:>8.2f} {r.total_queue:>8.1f}")

    # Ranking comparison
    print(f"\n{'=' * 70}")
    print("RANKING BY THROUGHPUT")
    print("=" * 70)
    for sim in ["LightSim", "SUMO"]:
        sim_r = sorted([r for r in results if r.simulator == sim],
                       key=lambda r: -r.total_exited)
        if not sim_r:
            continue
        print(f"\n  [{sim}]")
        for i, r in enumerate(sim_r):
            print(f"    {i+1}. {r.controller:<25} exited={r.total_exited:>7.0f}  "
                  f"queue={r.total_queue:>7.1f}")

    # Kendall's tau on shared controllers
    from scipy.stats import kendalltau
    ls_r = {r.controller: r.total_exited for r in results if r.simulator == "LightSim"}
    su_r = {r.controller: r.total_exited for r in results if r.simulator == "SUMO"}
    shared = sorted(set(ls_r.keys()) & set(su_r.keys()))
    if len(shared) >= 3:
        ls_sorted = sorted(shared, key=lambda c: -ls_r[c])
        su_sorted = sorted(shared, key=lambda c: -su_r[c])
        ls_ranks = [ls_sorted.index(c) + 1 for c in shared]
        su_ranks = [su_sorted.index(c) + 1 for c in shared]
        tau, p = kendalltau(ls_ranks, su_ranks)
        print(f"\n  Shared controllers: {shared}")
        print(f"  Kendall's tau = {tau:.3f} (p = {p:.4f})")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "arterial_cross_validation",
        "scenario": SCENARIO,
        "sim_seconds": SIM_SECONDS,
        "results": [asdict(r) for r in results],
    }
    if len(shared) >= 3:
        output["ranking_correlation"] = {
            "shared_controllers": shared,
            "kendall_tau": round(tau, 3),
            "p_value": round(p, 4),
        }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
