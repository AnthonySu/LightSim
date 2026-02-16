#!/usr/bin/env python
"""Fix SUMO grid-4x4 demand mismatch in cross-validation.

The _build_sumo_grid function in cross_validation.py uses hardcoded edge names
(bottom{i}_0, top{i}_{rows-2}, etc.) that do NOT match SUMO netgenerate
edge naming convention (which uses LetterNumber pairs like A0A1, B1C1).

When the flow definitions reference non-existent edges, SUMO falls back to
randomTrips.py which generates only ~0.5 veh/s (~1,725 vehicles in 3600s),
whereas LightSim processes ~18,500 vehicles from a demand of ~9.3 veh/s.

This script:
  1. Generates a SUMO 4x4 grid network using netgenerate
  2. Parses the net.xml to discover ACTUAL edge names
  3. Identifies boundary entry/exit edges programmatically
  4. Creates flow definitions matching LightSim ~0.15 veh/s per boundary link
  5. Runs SUMO with FixedTime (static) and MaxPressure (actuated) controllers
  6. Reports throughput, delay, queue metrics
  7. Saves results to results/sumo_grid_aligned.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path


def find_sumo_binary() -> str:
    """Find the sumo executable."""
    import shutil
    s = shutil.which("sumo")
    if s:
        return s
    try:
        import importlib.util
        spec = importlib.util.find_spec("sumolib")
        if spec and spec.origin:
            site_packages = Path(spec.origin).parent.parent
            for name in ("sumo.exe", "sumo"):
                candidate = site_packages / "sumo" / "bin" / name
                if candidate.exists():
                    return str(candidate)
    except Exception:
        pass
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        for name in ("sumo.exe", "sumo"):
            p = Path(sumo_home) / "bin" / name
            if p.exists():
                return str(p)
    raise FileNotFoundError("Could not find SUMO binary.")


def find_netgenerate() -> str:
    """Find the netgenerate binary."""
    sumo_bin = find_sumo_binary()
    bin_dir = Path(sumo_bin).parent
    suffix = ".exe" if sys.platform == "win32" else ""
    netgen = bin_dir / f"netgenerate{suffix}"
    if netgen.exists():
        return str(netgen)
    raise FileNotFoundError(f"netgenerate not found in {bin_dir}")


@dataclass
class GridTopology:
    """Parsed topology of a SUMO-generated grid network."""
    all_edges: dict
    node_types: dict
    outgoing: dict
    incoming: dict
    boundary_nodes: set
    corner_nodes: set
    entry_edges: list
    exit_edges: list
    tls_ids: list


def parse_grid_network(net_xml_path: str) -> GridTopology:
    """Parse a SUMO net.xml and extract the full grid topology.

    KEY FIX: inspect actual edge names rather than guessing with
    hardcoded patterns like bottom{i}_0, top{i}_{rows-2}.
    """
    tree = ET.parse(net_xml_path)
    root = tree.getroot()

    all_edges = {}
    node_types = {}
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    nodes = set()

    for edge_elem in root.findall("edge"):
        eid = edge_elem.get("id")
        if eid.startswith(":"):
            continue
        f = edge_elem.get("from")
        t = edge_elem.get("to")
        all_edges[eid] = (f, t)
        outgoing[f].append(eid)
        incoming[t].append(eid)
        nodes.add(f)
        nodes.add(t)

    for junc in root.findall("junction"):
        jid = junc.get("id")
        if jid.startswith(":"):
            continue
        node_types[jid] = junc.get("type", "")

    tls_ids = [tl.get("id") for tl in root.findall("tlLogic")]

    corner_nodes = set()
    boundary_nodes = set()
    for n in nodes:
        deg = len(outgoing[n]) + len(incoming[n])
        if deg <= 4:
            corner_nodes.add(n)
            boundary_nodes.add(n)
        elif deg <= 6:
            boundary_nodes.add(n)

    entry_edges = []
    exit_edges = []
    for eid, (f, t) in all_edges.items():
        f_deg = len(outgoing[f]) + len(incoming[f])
        t_deg = len(outgoing[t]) + len(incoming[t])
        if f in boundary_nodes and t_deg > f_deg:
            entry_edges.append(eid)
        elif t in boundary_nodes and f_deg > t_deg:
            exit_edges.append(eid)

    for eid, (f, t) in all_edges.items():
        if eid in entry_edges or eid in exit_edges:
            continue
        if f in corner_nodes and t in boundary_nodes and t not in corner_nodes:
            entry_edges.append(eid)
        elif t in corner_nodes and f in boundary_nodes and f not in corner_nodes:
            exit_edges.append(eid)

    entry_edges.sort()
    exit_edges.sort()

    return GridTopology(
        all_edges=all_edges, node_types=node_types,
        outgoing=dict(outgoing), incoming=dict(incoming),
        boundary_nodes=boundary_nodes, corner_nodes=corner_nodes,
        entry_edges=entry_edges, exit_edges=exit_edges, tls_ids=tls_ids,
    )


def classify_edge_side(topo, eid):
    """Classify an edge by direction."""
    ef, et = topo.all_edges[eid]
    el, en = ef[0], int(ef[1:])
    tl, tn = et[0], int(et[1:])
    if tl > el: return "left_to_right"
    if tl < el: return "right_to_left"
    if tn > en: return "bottom_to_top"
    if tn < en: return "top_to_bottom"
    return "unknown"


OPPOSITE = {
    "left_to_right": "right_to_left",
    "right_to_left": "left_to_right",
    "bottom_to_top": "top_to_bottom",
    "top_to_bottom": "bottom_to_top",
}

PERP = {
    "left_to_right": ["top_to_bottom", "bottom_to_top"],
    "right_to_left": ["top_to_bottom", "bottom_to_top"],
    "bottom_to_top": ["left_to_right", "right_to_left"],
    "top_to_bottom": ["left_to_right", "right_to_left"],
}


def _xml(tag, **attrs):
    """Build a self-closing XML element."""
    parts = [f"<{tag}"]
    for k, v in attrs.items():
        parts.append(f' {k}="{v}"')
    parts.append("/>")
    return "".join(parts)


def build_flow_routes(topo, net_xml_path, sim_seconds, demand_rate):
    """Build SUMO flows using CORRECT edge names discovered from net.xml."""
    import sumolib
    net = sumolib.net.readNet(net_xml_path)
    vph = int(demand_rate * 3600)

    lines = [
        "<routes>",
        "    " + _xml("vType", id="car", length="5",
                       maxSpeed="13.89", accel="2.6", decel="4.5"),
    ]

    exit_by_side = defaultdict(list)
    for eid in topo.exit_edges:
        exit_by_side[classify_edge_side(topo, eid)].append(eid)

    flow_id = 0
    created = 0
    failed = 0
    used = set()

    def try_add(entry, exit_eid):
        nonlocal flow_id, created
        key = (entry, exit_eid)
        if key in used:
            return False
        try:
            path = net.getShortestPath(net.getEdge(entry), net.getEdge(exit_eid))
            if path and path[0]:
                edges = " ".join(e.getID() for e in path[0])
                rn = f"route_{flow_id}"
                lines.append("    " + _xml("route", id=rn, edges=edges))
                lines.append("    " + _xml("flow", id=f"f_{flow_id}", type="car",
                    route=rn, begin="0", end=str(sim_seconds),
                    vehsPerHour=str(vph), departLane="best"))
                used.add(key)
                created += 1
                flow_id += 1
                return True
        except Exception:
            pass
        return False

    # Phase 1: through-flows
    for entry in topo.entry_edges:
        side = classify_edge_side(topo, entry)
        cands = exit_by_side.get(OPPOSITE.get(side, ""), [])
        if not cands:
            cands = [e for e in topo.exit_edges if e != entry]
        ok = False
        for ex in cands:
            if try_add(entry, ex):
                ok = True
                break
        if not ok:
            failed += 1

    # Phase 2: perpendicular cross-flows
    for entry in topo.entry_edges:
        side = classify_edge_side(topo, entry)
        for ps in PERP.get(side, []):
            for ex in exit_by_side.get(ps, [])[:2]:
                try_add(entry, ex)

    lines.append("</routes>")

    tot_vph = created * vph
    tot_vps = tot_vph / 3600
    print(f"  Flows created: {created} (failed: {failed})")
    print(f"  Total demand: {tot_vph} veh/hr = {tot_vps:.1f} veh/s")
    print(f"  Expected vehicles in {sim_seconds}s: ~{int(tot_vps * sim_seconds)}")
    print(f"  LightSim comparison: 62 origins x 0.15 = 9.3 veh/s = 33,480 veh/hr")
    return "\n".join(lines)


@dataclass
class SimResult:
    controller: str
    tls_type: str
    total_departed: int
    total_arrived: int
    avg_delay: float
    max_queue: float
    avg_queue: float
    total_queue_sum: float
    wall_time: float
    sim_seconds: int


def run_sumo_simulation(sumo_bin, cfg_path, sim_seconds, ctrl_name, tls_type):
    """Run SUMO via traci and collect metrics."""
    import traci

    label = f"sim_{ctrl_name}_{int(time.time() * 1000)}"
    t0 = time.perf_counter()
    traci.start([sumo_bin, "-c", cfg_path, "--no-warnings", "true"], label=label)

    total_dep = 0
    total_arr = 0
    queues = []
    conn = traci.getConnection(label)

    for step in range(sim_seconds):
        conn.simulationStep()
        total_dep += conn.simulation.getDepartedNumber()
        total_arr += conn.simulation.getArrivedNumber()
        qc = 0
        try:
            for vid in conn.vehicle.getIDList():
                if conn.vehicle.getSpeed(vid) < 0.1:
                    qc += 1
        except Exception:
            pass
        queues.append(qc)
        if (step + 1) % 600 == 0:
            print(f"    Step {step+1}/{sim_seconds}: dep={total_dep}, arr={total_arr}, q={qc}")

    delays = []
    try:
        for vid in conn.vehicle.getIDList():
            delays.append(conn.vehicle.getWaitingTime(vid))
    except Exception:
        pass

    conn.close()
    wall = time.perf_counter() - t0
    return SimResult(
        controller=ctrl_name, tls_type=tls_type,
        total_departed=total_dep, total_arrived=total_arr,
        avg_delay=float(sum(delays)/len(delays)) if delays else 0.0,
        max_queue=float(max(queues)) if queues else 0.0,
        avg_queue=float(sum(queues)/len(queues)) if queues else 0.0,
        total_queue_sum=float(sum(queues)),
        wall_time=wall, sim_seconds=sim_seconds,
    )


def make_sumocfg(net_name, rou_name, sim_seconds):
    """Generate SUMO config XML."""
    p = []
    p.append("<configuration>")
    p.append("    <input>")
    p.append(f'        <net-file value="{net_name}"/>')
    p.append(f'        <route-files value="{rou_name}"/>')
    p.append("    </input>")
    p.append("    <time>")
    p.append('        <begin value="0"/>')
    p.append(f'        <end value="{sim_seconds}"/>')
    p.append('        <step-length value="1.0"/>')
    p.append("    </time>")
    p.append("    <processing>")
    p.append('        <no-step-log value="true"/>')
    p.append("    </processing>")
    p.append("    <report>")
    p.append('        <no-warnings value="true"/>')
    p.append('        <verbose value="false"/>')
    p.append("    </report>")
    p.append("</configuration>")
    return "\n".join(p)


def main():
    rows, cols = 4, 4
    sim_seconds = 3600
    demand_rate = 0.15

    print("=" * 72)
    print("SUMO Grid-4x4 Demand Alignment Fix")
    print("=" * 72)
    print()

    sumo_bin = find_sumo_binary()
    netgen_bin = find_netgenerate()
    print(f"SUMO binary:        {sumo_bin}")
    print(f"netgenerate binary: {netgen_bin}")
    print()

    results_all = {}

    for ctrl_name, tls_type in [("FixedTime", "static"), ("MaxPressure", "actuated")]:
        print("-" * 72)
        print(f"Controller: {ctrl_name}  (SUMO tls type: {tls_type})")
        print("-" * 72)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Step 1: Generate network
            print()
            print("[1] Generating 4x4 grid network with netgenerate...")
            net_xml = tmppath / "net.net.xml"
            cmd = [
                find_netgenerate(), "--grid",
                "--grid.x-number", str(cols), "--grid.y-number", str(rows),
                "--grid.x-length", "300", "--grid.y-length", "300",
                "--default.lanenumber", "2", "--default.speed", "13.89",
                "--tls.guess", "true", f"--tls.default-type={tls_type}",
                "-o", str(net_xml), "--no-warnings", "true",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"ERROR: netgenerate failed: {res.stderr}")
                sys.exit(1)
            print(f"    Network saved to: {net_xml}")

            # Step 2: Inspect actual edge names
            print()
            print("[2] Inspecting actual edge names in generated network...")
            topo = parse_grid_network(str(net_xml))

            print(f"    Total edges: {len(topo.all_edges)}")
            print(f"    Boundary nodes ({len(topo.boundary_nodes)}): {sorted(topo.boundary_nodes)}")
            print(f"    Corner nodes ({len(topo.corner_nodes)}): {sorted(topo.corner_nodes)}")
            print(f"    TLS IDs ({len(topo.tls_ids)}): {topo.tls_ids}")

            print()
            print(f"    Entry edges ({len(topo.entry_edges)}):")
            for eid in topo.entry_edges:
                f, t = topo.all_edges[eid]
                print(f"      {eid:15s}  {f} -> {t}")

            print()
            print(f"    Exit edges ({len(topo.exit_edges)}):")
            for eid in topo.exit_edges:
                f, t = topo.all_edges[eid]
                print(f"      {eid:15s}  {f} -> {t}")

            print()
            print("    ALL edge names:")
            for eid in sorted(topo.all_edges.keys()):
                f, t = topo.all_edges[eid]
                print(f"      {eid:15s}  {f} -> {t}")

            # Show what the buggy code expected
            print()
            print("    [BUG DIAGNOSIS] cross_validation.py _build_sumo_grid used these")
            print("    WRONG hardcoded edge names:")
            for i in range(cols):
                print(f'      SN: from="bottom{i}_0"  to="{rows-2}_top{i}"')
            for i in range(cols):
                print(f'      NS: from="top{i}_{rows-2}"  to="0_bottom{i}"')
            for j in range(rows):
                print(f'      WE: from="left{j}_0"  to="{cols-2}_right{j}"')
            for j in range(rows):
                print(f'      EW: from="right{j}_{cols-2}"  to="0_left{j}"')
            print("    NONE of these exist in the netgenerate output!")
            print("    Actual naming: nodes=LetterNumber (A0,B1,C2,D3)")
            print("    Edges=FromTo concatenation (A0A1, B1C1, D2D3)")

            # Step 3: Create flows
            print()
            print(f"[3] Creating flow definitions with rate={demand_rate} veh/s per link...")
            route_xml = build_flow_routes(topo, str(net_xml), sim_seconds, demand_rate)
            rou_xml = tmppath / "routes.rou.xml"
            rou_xml.write_text(route_xml)

            # Step 4: Config
            cfg_xml = tmppath / "sim.sumocfg"
            cfg_xml.write_text(make_sumocfg(net_xml.name, rou_xml.name, sim_seconds))

            # Step 5: Run SUMO
            print()
            print(f"[4] Running SUMO with {ctrl_name} ({sim_seconds}s)...")
            sim_result = run_sumo_simulation(
                sumo_bin, str(cfg_xml), sim_seconds, ctrl_name, tls_type)

            print()
            print(f"    Results for {ctrl_name}:")
            print(f"      Total departed:  {sim_result.total_departed}")
            print(f"      Total arrived:   {sim_result.total_arrived}")
            print(f"      Avg delay:       {sim_result.avg_delay:.2f} s")
            print(f"      Max queue:       {sim_result.max_queue:.0f} vehicles")
            print(f"      Avg queue:       {sim_result.avg_queue:.1f} vehicles")
            print(f"      Wall time:       {sim_result.wall_time:.2f} s")
            results_all[ctrl_name] = asdict(sim_result)

    # Summary
    print()
    print("=" * 72)
    print("SUMMARY: SUMO Grid-4x4 with Aligned Demand")
    print("=" * 72)
    hdr = "{:<15} {:>10} {:>10} {:>10} {:>8} {:>8} {:>8}".format(
        "Controller", "Departed", "Arrived", "Delay", "MaxQ", "AvgQ", "Wall(s)")
    print(hdr)
    print("-" * len(hdr))
    for cn, r in results_all.items():
        print("{:<15} {:>10} {:>10} {:>10.2f} {:>8.0f} {:>8.1f} {:>8.2f}".format(
            cn, r["total_departed"], r["total_arrived"],
            r["avg_delay"], r["max_queue"], r["avg_queue"], r["wall_time"]))

    print()
    print(f"  LightSim reference: ~18,500 vehicles processed in {sim_seconds}s")
    print("  Old SUMO (randomTrips fallback): ~1,725 vehicles")
    for cn, r in results_all.items():
        arr = r["total_arrived"]
        pct = arr / 18500 * 100 if arr > 0 else 0
        print(f"  Fixed SUMO ({cn}): {arr} vehicles ({pct:.0f}% of LightSim)")

    # Save
    script_dir = Path(__file__).resolve().parent.parent
    results_dir = script_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "sumo_grid_aligned.json"

    out_data = {
        "description": "SUMO grid-4x4 cross-validation with corrected demand alignment",
        "fix_applied": (
            "Replaced hardcoded edge names (bottom{i}_0, top{i}_{rows-2}, etc.) "
            "with actual netgenerate edge names discovered by parsing net.xml. "
            "Edge naming: netgenerate uses LetterNumber pairs (A0, B1, C2, D3) "
            "for nodes and concatenation for edges (A0A1, B1C1)."
        ),
        "parameters": {
            "rows": rows, "cols": cols, "sim_seconds": sim_seconds,
            "demand_rate_per_entry_link": demand_rate,
            "lightsim_reference_throughput": 18500,
            "old_sumo_throughput_randomtrips": 1725,
        },
        "results": results_all,
    }
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print()
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
