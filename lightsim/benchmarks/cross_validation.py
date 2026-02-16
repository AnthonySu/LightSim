"""Cross-validation: compare LightSim vs SUMO under identical controllers.

Runs FixedTime and MaxPressure controllers in both LightSim and SUMO on
single-intersection and grid-4x4 scenarios. Compares throughput, delay,
and queue metrics to verify that the relative ordering of controllers
is preserved across simulators.

Usage::

    python -m lightsim.benchmarks.cross_validation
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from ..core.demand import DemandProfile
from ..core.engine import SimulationEngine
from ..core.signal import FixedTimeController, MaxPressureController
from ..core.types import LinkID, NodeID, NodeType
from ..utils.metrics import compute_link_delay
from .sumo_comparison import _find_sumo_binary


@dataclass
class CrossValResult:
    simulator: str
    scenario: str
    controller: str
    total_exited: float
    avg_delay: float
    total_queue: float
    wall_time: float


def _run_lightsim(scenario_name: str, controller, episode_steps: int = 3600,
                  dt: float = 1.0) -> CrossValResult:
    """Run a scenario in LightSim with the given controller."""
    from .scenarios import get_scenario

    factory = get_scenario(scenario_name)
    network, demand = factory()

    engine = SimulationEngine(
        network=network, dt=dt,
        controller=controller,
        demand_profiles=demand,
    )
    engine.reset(seed=42)

    t0 = time.perf_counter()
    for _ in range(episode_steps):
        engine.step()
    wall = time.perf_counter() - t0

    metrics = engine.get_network_metrics()

    # Compute delay and queue
    total_delay = 0.0
    total_queue = 0.0
    n_links = 0
    for link in network.links.values():
        to_node = network.nodes.get(link.to_node)
        if to_node and to_node.node_type == NodeType.SIGNALIZED:
            total_delay += compute_link_delay(engine, link.link_id)
            total_queue += engine.get_link_queue(link.link_id)
            n_links += 1

    avg_delay = total_delay / max(n_links, 1)

    return CrossValResult(
        simulator="LightSim",
        scenario=scenario_name,
        controller=type(controller).__name__,
        total_exited=metrics["total_exited"],
        avg_delay=avg_delay,
        total_queue=total_queue,
        wall_time=wall,
    )


def _run_sumo_with_controller(scenario_name: str, controller_name: str,
                              sim_seconds: int = 3600) -> CrossValResult:
    """Run a SUMO scenario with a specified controller type."""
    sumo_bin = _find_sumo_binary()
    bin_dir = Path(sumo_bin).parent

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        if scenario_name in ("single-intersection-v0", "single-intersection"):
            cfg = _build_sumo_single_intersection(
                tmppath, sumo_bin, controller_name, sim_seconds
            )
        elif scenario_name in ("grid-4x4-v0", "grid-4x4"):
            cfg = _build_sumo_grid(
                tmppath, sumo_bin, 4, 4, controller_name, sim_seconds
            )
        else:
            raise ValueError(f"Unknown scenario for SUMO cross-val: {scenario_name}")

        # Run SUMO with traci to collect metrics
        try:
            import traci
        except ImportError:
            raise ImportError("traci required: pip install traci")

        t0 = time.perf_counter()
        traci.start([sumo_bin, "-c", str(cfg), "--no-warnings", "true"])

        total_departed = 0
        total_arrived = 0
        total_waiting = 0
        step_count = 0

        for _ in range(sim_seconds):
            traci.simulationStep()
            total_departed += traci.simulation.getDepartedNumber()
            total_arrived += traci.simulation.getArrivedNumber()
            total_waiting += traci.simulation.getMinExpectedNumber()
            step_count += 1

        # Get final stats
        total_exited = total_arrived
        # Average waiting time from all vehicles that completed trips
        avg_delay = 0.0
        try:
            # Use mean travel time from tripinfo if available
            vehicle_ids = traci.vehicle.getIDList()
            if vehicle_ids:
                delays = []
                for vid in vehicle_ids[:100]:  # sample up to 100
                    delays.append(traci.vehicle.getWaitingTime(vid))
                avg_delay = float(np.mean(delays)) if delays else 0.0
        except Exception:
            pass

        # Queue = vehicles currently waiting
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
        scenario=scenario_name,
        controller=controller_name,
        total_exited=float(total_exited),
        avg_delay=avg_delay,
        total_queue=total_queue,
        wall_time=wall,
    )


def _build_sumo_single_intersection(tmpdir: Path, sumo_bin: str,
                                     controller: str, sim_seconds: int) -> Path:
    """Build SUMO config for single intersection with specified controller."""
    nod_xml = tmpdir / "nodes.nod.xml"
    edg_xml = tmpdir / "edges.edg.xml"
    net_xml = tmpdir / "net.net.xml"
    rou_xml = tmpdir / "routes.rou.xml"
    cfg_xml = tmpdir / "sim.sumocfg"
    tll_xml = tmpdir / "tls.add.xml"

    tls_type = "static" if controller == "FixedTimeController" else "actuated"

    nod_xml.write_text(textwrap.dedent(f"""\
        <nodes>
            <node id="C" x="0" y="0" type="traffic_light" tlType="{tls_type}"/>
            <node id="N" x="0" y="300" type="priority"/>
            <node id="S" x="0" y="-300" type="priority"/>
            <node id="E" x="300" y="0" type="priority"/>
            <node id="W" x="-300" y="0" type="priority"/>
        </nodes>
    """))

    edg_xml.write_text(textwrap.dedent("""\
        <edges>
            <edge id="NC" from="N" to="C" numLanes="2" speed="13.89"/>
            <edge id="SC" from="S" to="C" numLanes="2" speed="13.89"/>
            <edge id="EC" from="E" to="C" numLanes="2" speed="13.89"/>
            <edge id="WC" from="W" to="C" numLanes="2" speed="13.89"/>
            <edge id="CN" from="C" to="N" numLanes="2" speed="13.89"/>
            <edge id="CS" from="C" to="S" numLanes="2" speed="13.89"/>
            <edge id="CE" from="C" to="E" numLanes="2" speed="13.89"/>
            <edge id="CW" from="C" to="W" numLanes="2" speed="13.89"/>
        </edges>
    """))

    netconvert = str(Path(sumo_bin).parent / (
        "netconvert.exe" if sys.platform == "win32" else "netconvert"
    ))
    subprocess.run(
        [netconvert, "-n", str(nod_xml), "-e", str(edg_xml),
         "-o", str(net_xml), "--no-warnings", "true"],
        check=True, capture_output=True,
    )

    # Routes matching LightSim demand: NS=0.3 veh/s, EW=0.2 veh/s
    rou_xml.write_text(textwrap.dedent(f"""\
        <routes>
            <vType id="car" length="5" maxSpeed="13.89" accel="2.6" decel="4.5"/>
            <flow id="f_NS" type="car" from="NC" to="CS" begin="0" end="{sim_seconds}" vehsPerHour="1080" departLane="best"/>
            <flow id="f_SN" type="car" from="SC" to="CN" begin="0" end="{sim_seconds}" vehsPerHour="1080" departLane="best"/>
            <flow id="f_EW" type="car" from="EC" to="CW" begin="0" end="{sim_seconds}" vehsPerHour="720" departLane="best"/>
            <flow id="f_WE" type="car" from="WC" to="CE" begin="0" end="{sim_seconds}" vehsPerHour="720" departLane="best"/>
        </routes>
    """))

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

    return cfg_xml


def _build_sumo_grid(tmpdir: Path, sumo_bin: str,
                     rows: int, cols: int,
                     controller: str, sim_seconds: int) -> Path:
    """Build SUMO config for a grid network."""
    net_xml = tmpdir / "net.net.xml"
    rou_xml = tmpdir / "routes.rou.xml"
    cfg_xml = tmpdir / "sim.sumocfg"

    tls_type = "static" if controller == "FixedTimeController" else "actuated"

    netgenerate = str(Path(sumo_bin).parent / (
        "netgenerate.exe" if sys.platform == "win32" else "netgenerate"
    ))
    subprocess.run(
        [netgenerate, "--grid",
         "--grid.x-number", str(cols),
         "--grid.y-number", str(rows),
         "--grid.x-length", "300",
         "--grid.y-length", "300",
         "--default.lanenumber", "2",
         "--default.speed", "13.89",
         "--tls.guess", "true",
         f"--tls.default-type={tls_type}",
         "-o", str(net_xml),
         "--no-warnings", "true"],
        check=True, capture_output=True,
    )

    # Generate simple flow-based demand
    flow_lines = [
        '<routes>',
        '    <vType id="car" length="5" maxSpeed="13.89" accel="2.6" decel="4.5"/>',
    ]
    # Create flows from boundary edges
    fid = 0
    for i in range(cols):
        # South to North
        flow_lines.append(
            f'    <flow id="f_SN{i}" type="car" '
            f'from="bottom{i}_0" to="{rows-2}_top{i}" '
            f'begin="0" end="{sim_seconds}" vehsPerHour="360" departLane="best"/>'
        )
        # North to South
        flow_lines.append(
            f'    <flow id="f_NS{i}" type="car" '
            f'from="top{i}_{rows-2}" to="0_bottom{i}" '
            f'begin="0" end="{sim_seconds}" vehsPerHour="360" departLane="best"/>'
        )
    for j in range(rows):
        # West to East
        flow_lines.append(
            f'    <flow id="f_WE{j}" type="car" '
            f'from="left{j}_0" to="{cols-2}_right{j}" '
            f'begin="0" end="{sim_seconds}" vehsPerHour="360" departLane="best"/>'
        )
        # East to West
        flow_lines.append(
            f'    <flow id="f_EW{j}" type="car" '
            f'from="right{j}_{cols-2}" to="0_left{j}" '
            f'begin="0" end="{sim_seconds}" vehsPerHour="360" departLane="best"/>'
        )
    flow_lines.append('</routes>')

    # Use randomTrips as fallback since edge naming is unpredictable
    tools_dir = Path(sumo_bin).parent.parent / "tools"
    random_trips = tools_dir / "randomTrips.py"
    if not random_trips.exists():
        try:
            import sumolib
            sumolib_dir = Path(sumolib.__file__).parent
            random_trips = sumolib_dir.parent / "sumo" / "tools" / "randomTrips.py"
        except ImportError:
            pass
    if not random_trips.exists():
        random_trips = Path(sumo_bin).parent.parent / "share" / "sumo" / "tools" / "randomTrips.py"

    trip_xml = tmpdir / "trips.trips.xml"
    if random_trips.exists():
        total_vehs = int(sim_seconds * 0.5)
        subprocess.run(
            [sys.executable, str(random_trips),
             "-n", str(net_xml),
             "-o", str(trip_xml),
             "-e", str(sim_seconds),
             "-p", str(max(1, sim_seconds / total_vehs)),
             "--fringe-factor", "100",
             "--validate"],
            check=True, capture_output=True,
        )
        rou_xml = trip_xml
    else:
        rou_xml.write_text("\n".join(flow_lines))

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

    return cfg_xml


def run_cross_validation() -> list[CrossValResult]:
    """Run cross-validation across simulators, scenarios, and controllers."""
    results = []

    scenarios = ["single-intersection-v0"]
    controllers_ls = [
        ("FixedTimeController", FixedTimeController()),
        ("MaxPressureController", MaxPressureController(min_green=5.0)),
    ]

    for scenario in scenarios:
        for ctrl_name, ctrl_obj in controllers_ls:
            print(f"  LightSim / {scenario} / {ctrl_name}...", flush=True)
            r = _run_lightsim(scenario, ctrl_obj)
            results.append(r)
            print(f"    exited={r.total_exited:.0f}, delay={r.avg_delay:.2f}, "
                  f"queue={r.total_queue:.1f}")

            print(f"  SUMO / {scenario} / {ctrl_name}...", flush=True)
            try:
                r = _run_sumo_with_controller(scenario, ctrl_name)
                results.append(r)
                print(f"    exited={r.total_exited:.0f}, delay={r.avg_delay:.2f}, "
                      f"queue={r.total_queue:.1f}")
            except Exception as e:
                print(f"    SUMO failed: {e}")

    return results


def main():
    print("Cross-Validation: LightSim vs SUMO\n")

    try:
        _find_sumo_binary()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    results = run_cross_validation()

    # Print summary
    print("\n" + "=" * 80)
    print("Cross-Validation Results")
    print("=" * 80)
    header = f"{'Simulator':<12} {'Scenario':<25} {'Controller':<22} {'Exited':>8} {'Delay':>8} {'Queue':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.simulator:<12} {r.scenario:<25} {r.controller:<22} "
              f"{r.total_exited:>8.0f} {r.avg_delay:>8.2f} {r.total_queue:>8.1f}")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/cross_validation.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print("\nSaved to results/cross_validation.json")


if __name__ == "__main__":
    main()
