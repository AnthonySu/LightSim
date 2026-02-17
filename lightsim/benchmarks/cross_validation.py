"""Cross-validation: compare LightSim vs SUMO under identical controllers.

Runs multiple controllers in both LightSim and SUMO on single-intersection
and grid-4x4 scenarios. Compares throughput, delay, and queue metrics to
verify that the relative ordering of controllers is preserved across
simulators.

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
from xml.etree import ElementTree

import numpy as np

from ..core.demand import DemandProfile
from ..core.engine import SimulationEngine
from ..core.signal import (
    FixedTimeController,
    LostTimeAwareMaxPressureController,
    MaxPressureController,
    SOTLController,
    WebsterController,
)
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


def _is_green_phase(state: str) -> bool:
    """Return True if phase state contains at least one 'G' or 'g'."""
    return any(c in ('G', 'g') for c in state)


# Track when each TLS entered its current phase: {tls_id: (phase_idx, start_time)}
_phase_entry: dict[str, tuple[int, float]] = {}


def _get_time_in_phase(traci_mod, tls_id: str) -> float:
    """Return seconds spent in the current phase, tracked across setPhaseDuration calls."""
    current = traci_mod.trafficlight.getPhase(tls_id)
    sim_time = traci_mod.simulation.getTime()
    prev = _phase_entry.get(tls_id)
    if prev is None or prev[0] != current:
        _phase_entry[tls_id] = (current, sim_time)
        return 0.0
    return sim_time - prev[1]


def _traci_maxpressure_step(traci_mod, min_green: float = 15.0):
    """One-step MaxPressure controller via TraCI.

    Reads queue counts per incoming lane, computes pressure for each green
    phase, and holds/ends the current green to favour the highest-pressure
    direction.  Yellow/all-red phases are never interrupted.
    """
    for tls_id in traci_mod.trafficlight.getIDList():
        logic = traci_mod.trafficlight.getAllProgramLogics(tls_id)[0]
        n_phases = len(logic.phases)
        current = traci_mod.trafficlight.getPhase(tls_id)
        current_state = logic.phases[current].state

        time_in_phase = _get_time_in_phase(traci_mod, tls_id)

        # Only act during green phases — let yellow/all-red run naturally
        if not _is_green_phase(current_state):
            continue

        # Identify green phases and compute pressure for each
        controlled_links = traci_mod.trafficlight.getControlledLinks(tls_id)
        green_phases = []
        phase_pressure = {}
        for pi in range(n_phases):
            state = logic.phases[pi].state
            if not _is_green_phase(state):
                continue
            green_phases.append(pi)
            pressure = 0.0
            for li, link_info in enumerate(controlled_links):
                if li < len(state) and state[li] in ('G', 'g'):
                    for in_lane, out_lane, _ in link_info:
                        try:
                            in_halt = traci_mod.lane.getLastStepHaltingNumber(
                                in_lane)
                            out_halt = traci_mod.lane.getLastStepHaltingNumber(
                                out_lane)
                            pressure += max(0, in_halt - out_halt)
                        except Exception:
                            pass
            phase_pressure[pi] = pressure

        if not green_phases:
            continue

        best_phase = max(green_phases, key=lambda p: phase_pressure[p])

        if time_in_phase < min_green:
            # Haven't served min_green yet — hold
            traci_mod.trafficlight.setPhaseDuration(
                tls_id, min_green - time_in_phase)
        elif best_phase != current:
            # A different phase has higher pressure — end current green
            traci_mod.trafficlight.setPhaseDuration(tls_id, 0)
        else:
            # Current phase has highest pressure — extend by 5s
            traci_mod.trafficlight.setPhaseDuration(tls_id, 5.0)


def _traci_sotl_step(traci_mod, mu: float = 5.0,
                     min_green: float = 10.0, max_green: float = 60.0):
    """One-step SOTL controller via TraCI.

    Extends green while vehicles are approaching the stop bar.
    Switches when no approaching vehicles detected or max_green reached.
    Yellow/all-red phases are never interrupted.
    """
    for tls_id in traci_mod.trafficlight.getIDList():
        logic = traci_mod.trafficlight.getAllProgramLogics(tls_id)[0]
        current = traci_mod.trafficlight.getPhase(tls_id)
        current_state = logic.phases[current].state

        time_in_phase = _get_time_in_phase(traci_mod, tls_id)

        # Only act during green phases
        if not _is_green_phase(current_state):
            continue

        controlled_links = traci_mod.trafficlight.getControlledLinks(tls_id)

        # Count approaching vehicles on current green lanes
        approaching = 0
        for li, link_info in enumerate(controlled_links):
            if li < len(current_state) and current_state[li] in ('G', 'g'):
                for in_lane, _, _ in link_info:
                    try:
                        approaching += traci_mod.lane.getLastStepVehicleNumber(
                            in_lane)
                    except Exception:
                        pass

        # Decision: end current green or extend it
        if time_in_phase < min_green:
            # Haven't served min_green — hold
            traci_mod.trafficlight.setPhaseDuration(
                tls_id, min_green - time_in_phase)
        elif time_in_phase >= max_green:
            # Exceeded max_green — force switch
            traci_mod.trafficlight.setPhaseDuration(tls_id, 0)
        elif approaching <= mu:
            # No vehicles approaching — switch
            traci_mod.trafficlight.setPhaseDuration(tls_id, 0)
        else:
            # Vehicles still approaching — extend green by 5s
            traci_mod.trafficlight.setPhaseDuration(tls_id, 5.0)


def _run_sumo_with_controller(scenario_name: str, controller_name: str,
                              sim_seconds: int = 3600) -> CrossValResult:
    """Run a SUMO scenario with a specified controller type."""
    sumo_bin = _find_sumo_binary()

    # TraCI-based controllers use static TLS as base to avoid conflicts
    traci_controller = controller_name in (
        "MaxPressureController", "SOTLController",
    )
    sumo_tls_type = controller_name
    if traci_controller:
        sumo_tls_type = "FixedTimeController"  # Static base — TraCI manages switching

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        if scenario_name in ("single-intersection-v0", "single-intersection"):
            cfg = _build_sumo_single_intersection(
                tmppath, sumo_bin, sumo_tls_type, sim_seconds
            )
        elif scenario_name in ("grid-4x4-v0", "grid-4x4"):
            cfg = _build_sumo_grid(
                tmppath, sumo_bin, 4, 4, sumo_tls_type, sim_seconds
            )
        else:
            raise ValueError(f"Unknown scenario for SUMO cross-val: {scenario_name}")

        try:
            import traci
        except ImportError:
            raise ImportError("traci required: pip install traci")

        t0 = time.perf_counter()
        _phase_entry.clear()  # Reset phase timers for new simulation
        traci.start([sumo_bin, "-c", str(cfg), "--no-warnings", "true"])

        total_arrived = 0

        for step in range(sim_seconds):
            # Apply TraCI-based controller before stepping
            if traci_controller and step > 0:
                if controller_name == "MaxPressureController":
                    _traci_maxpressure_step(traci, min_green=15.0)
                elif controller_name == "SOTLController":
                    _traci_sotl_step(traci, min_green=10.0, max_green=60.0)

            traci.simulationStep()
            total_arrived += traci.simulation.getArrivedNumber()

        # Collect final metrics
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
                     controller: str, sim_seconds: int,
                     demand_rate: float = 0.15) -> Path:
    """Build SUMO config for a grid network matching LightSim's structure.

    Creates a (rows+2)x(cols+2) grid with boundary priority nodes and
    interior signalized nodes, matching LightSim's grid_4x4 scenario exactly.
    Demand is injected from each boundary node at ``demand_rate`` veh/s.
    """
    nod_xml = tmpdir / "nodes.nod.xml"
    edg_xml = tmpdir / "edges.edg.xml"
    net_xml = tmpdir / "net.net.xml"
    rou_xml = tmpdir / "routes.rou.xml"
    cfg_xml = tmpdir / "sim.sumocfg"

    tls_type = "static" if controller == "FixedTimeController" else "actuated"

    total_rows = rows + 2
    total_cols = cols + 2
    spacing = 300  # meters

    def nid(r: int, c: int) -> str:
        return f"n{r}_{c}"

    # --- Nodes ---
    node_lines = ['<nodes>']
    for r in range(total_rows):
        for c in range(total_cols):
            is_corner = (r in (0, total_rows - 1)) and (c in (0, total_cols - 1))
            if is_corner:
                continue
            is_boundary = r == 0 or r == total_rows - 1 or c == 0 or c == total_cols - 1
            # Check if boundary node connects to interior
            if is_boundary:
                has_interior = False
                if r == 0 and 1 <= c <= cols:
                    has_interior = True
                elif r == total_rows - 1 and 1 <= c <= cols:
                    has_interior = True
                elif c == 0 and 1 <= r <= rows:
                    has_interior = True
                elif c == total_cols - 1 and 1 <= r <= rows:
                    has_interior = True
                if not has_interior:
                    continue
            ntype = "priority" if is_boundary else "traffic_light"
            tl_attr = f' tlType="{tls_type}"' if ntype == "traffic_light" else ""
            x = c * spacing
            y = r * spacing
            node_lines.append(
                f'    <node id="{nid(r, c)}" x="{x}" y="{y}" '
                f'type="{ntype}"{tl_attr}/>'
            )
    node_lines.append('</nodes>')
    nod_xml.write_text("\n".join(node_lines))

    # --- Edges ---
    edge_lines = ['<edges>']
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Collect valid node IDs
    valid_nodes = set()
    tree = ElementTree.fromstring("\n".join(node_lines))
    for n in tree.findall("node"):
        valid_nodes.add(n.get("id"))

    for r in range(total_rows):
        for c in range(total_cols):
            fid = nid(r, c)
            if fid not in valid_nodes:
                continue
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                tid = nid(nr, nc)
                if tid in valid_nodes:
                    eid = f"{fid}{tid}"
                    edge_lines.append(
                        f'    <edge id="{eid}" from="{fid}" to="{tid}" '
                        f'numLanes="2" speed="13.89"/>'
                    )
    edge_lines.append('</edges>')
    edg_xml.write_text("\n".join(edge_lines))

    # --- Build net ---
    netconvert = str(Path(sumo_bin).parent / (
        "netconvert.exe" if sys.platform == "win32" else "netconvert"
    ))
    subprocess.run(
        [netconvert, "-n", str(nod_xml), "-e", str(edg_xml),
         "-o", str(net_xml), "--no-warnings", "true"],
        check=True, capture_output=True,
    )

    # --- Routes: inject demand from each boundary→interior edge ---
    # LightSim's grid has 2 origin→signalized links per boundary node
    # (a duplicate in the network builder), each at demand_rate.
    # Match effective input: 2 × demand_rate per boundary access point.
    veh_per_hour = int(2 * demand_rate * 3600)
    route_lines = [
        '<routes>',
        '    <vType id="car" length="5" maxSpeed="13.89" accel="2.6" decel="4.5"/>',
    ]
    fid = 0
    for r in range(total_rows):
        for c in range(total_cols):
            is_boundary = r == 0 or r == total_rows - 1 or c == 0 or c == total_cols - 1
            if not is_boundary:
                continue
            src = nid(r, c)
            if src not in valid_nodes:
                continue
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                # Only create flows from boundary → interior (signalized) nodes
                is_interior = 1 <= nr <= rows and 1 <= nc <= cols
                if not is_interior:
                    continue
                dst = nid(nr, nc)
                if dst not in valid_nodes:
                    continue
                from_edge = f"{src}{dst}"
                # Find opposite boundary node for destination
                # Travel in the same direction until hitting boundary
                opp_r, opp_c = nr, nc
                while 1 <= opp_r <= rows and 1 <= opp_c <= cols:
                    opp_r += dr
                    opp_c += dc
                # Back up one step to get last interior node
                last_r, last_c = opp_r - dr, opp_c - dc
                opp_nid = nid(opp_r, opp_c)
                last_nid = nid(last_r, last_c)
                if opp_nid in valid_nodes and last_nid in valid_nodes:
                    to_edge = f"{last_nid}{opp_nid}"
                    route_lines.append(
                        f'    <flow id="f{fid}" type="car" '
                        f'from="{from_edge}" to="{to_edge}" '
                        f'begin="0" end="{sim_seconds}" '
                        f'vehsPerHour="{veh_per_hour}" departLane="best"/>'
                    )
                    fid += 1
    route_lines.append('</routes>')
    rou_xml.write_text("\n".join(route_lines))

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

    scenarios = ["single-intersection-v0", "grid-4x4-v0"]

    # LightSim controllers
    controllers_ls = [
        ("FixedTimeController", FixedTimeController()),
        ("MaxPressureController", MaxPressureController(min_green=15.0)),
        ("SOTLController", SOTLController()),
        ("WebsterController", WebsterController()),
        ("LT-Aware-MP", LostTimeAwareMaxPressureController(min_green=5.0)),
    ]

    # SUMO controllers (name must match what _run_sumo_with_controller expects)
    sumo_controllers = [
        "FixedTimeController",      # SUMO static TLS
        "ActuatedController",       # SUMO built-in actuated
        "MaxPressureController",    # TraCI MaxPressure
        "SOTLController",           # TraCI SOTL
    ]

    for scenario in scenarios:
        print(f"\n{'#' * 60}")
        print(f"  SCENARIO: {scenario}")
        print(f"{'#' * 60}")

        # LightSim
        print(f"\n  [LightSim]")
        for ctrl_name, ctrl_obj in controllers_ls:
            print(f"    {ctrl_name}...", end="", flush=True)
            r = _run_lightsim(scenario, ctrl_obj)
            results.append(r)
            print(f"  exited={r.total_exited:.0f}, delay={r.avg_delay:.2f}, "
                  f"queue={r.total_queue:.1f}")

        # SUMO
        print(f"\n  [SUMO]")
        for ctrl_name in sumo_controllers:
            print(f"    {ctrl_name}...", end="", flush=True)
            try:
                r = _run_sumo_with_controller(scenario, ctrl_name)
                results.append(r)
                print(f"  exited={r.total_exited:.0f}, delay={r.avg_delay:.2f}, "
                      f"queue={r.total_queue:.1f}")
            except Exception as e:
                print(f"  FAILED: {e}")

    return results


def main():
    print("=" * 80)
    print("Cross-Validation: LightSim vs SUMO")
    print("Scenarios: single-intersection, grid-4x4")
    print("=" * 80)

    try:
        _find_sumo_binary()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    results = run_cross_validation()

    # Print summary table
    print(f"\n\n{'=' * 90}")
    print("SUMMARY")
    print("=" * 90)
    header = (f"{'Scenario':<25} {'Simulator':<12} {'Controller':<22} "
              f"{'Exited':>8} {'Delay':>8} {'Queue':>8}")
    print(header)
    print("-" * 90)
    for r in sorted(results, key=lambda x: (x.scenario, x.simulator, x.controller)):
        print(f"{r.scenario:<25} {r.simulator:<12} {r.controller:<22} "
              f"{r.total_exited:>8.0f} {r.avg_delay:>8.2f} {r.total_queue:>8.1f}")

    # Ranking comparison
    print(f"\n{'=' * 90}")
    print("RANKING COMPARISON (sorted by throughput)")
    print("=" * 90)
    for scenario in ["single-intersection-v0", "grid-4x4-v0"]:
        scen_results = [r for r in results if r.scenario == scenario]
        if not scen_results:
            continue
        print(f"\n  --- {scenario} ---")
        for sim in ["LightSim", "SUMO"]:
            sim_r = sorted([r for r in scen_results if r.simulator == sim],
                           key=lambda r: -r.total_exited)
            if not sim_r:
                continue
            print(f"\n  [{sim}]")
            for i, r in enumerate(sim_r):
                marker = " <-- best" if i == 0 else ""
                print(f"    {i+1}. {r.controller:<22} "
                      f"exited={r.total_exited:>8.0f}  "
                      f"delay={r.avg_delay:>7.2f}  "
                      f"queue={r.total_queue:>7.1f}{marker}")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/cross_validation.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nSaved to results/cross_validation.json")


if __name__ == "__main__":
    main()
