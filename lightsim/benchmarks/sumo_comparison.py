"""Speed benchmark: LightSim vs SUMO on equivalent scenarios.

Generates SUMO network/route/config files that match each LightSim benchmark
scenario, runs both simulators, and compares wall-clock throughput.

Usage::

    python -m lightsim.benchmarks.sumo_comparison
    python -m lightsim.benchmarks.sumo_comparison --steps 5000

Requires: eclipse-sumo, sumolib, traci (pip install eclipse-sumo sumolib traci)
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Locate SUMO binary
# ---------------------------------------------------------------------------

def _find_sumo_binary() -> str:
    """Find the sumo executable."""
    import shutil

    # 1. On PATH
    s = shutil.which("sumo")
    if s:
        return s

    # 2. pip-installed eclipse-sumo
    try:
        import importlib.util
        spec = importlib.util.find_spec("sumolib")
        if spec and spec.origin:
            site_packages = Path(spec.origin).parent.parent
            candidate = site_packages / "sumo" / "bin" / "sumo.exe"
            if candidate.exists():
                return str(candidate)
            candidate = site_packages / "sumo" / "bin" / "sumo"
            if candidate.exists():
                return str(candidate)
    except Exception:
        pass

    # 3. SUMO_HOME
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        for name in ("sumo.exe", "sumo"):
            p = Path(sumo_home) / "bin" / name
            if p.exists():
                return str(p)

    raise FileNotFoundError(
        "Could not find SUMO binary. Install with: pip install eclipse-sumo"
    )


# ---------------------------------------------------------------------------
# SUMO network generators — write .net.xml, .rou.xml, .sumocfg
# ---------------------------------------------------------------------------

def _write_single_intersection(tmpdir: Path, sim_seconds: int, step_length: float) -> Path:
    """Generate SUMO files for a single 4-leg intersection."""
    # Use netgenerate-style inline XML
    net_xml = tmpdir / "net.net.xml"
    rou_xml = tmpdir / "routes.rou.xml"
    cfg_xml = tmpdir / "sim.sumocfg"

    # Build net via node/edge XML + netconvert
    nod_xml = tmpdir / "nodes.nod.xml"
    edg_xml = tmpdir / "edges.edg.xml"
    con_xml = tmpdir / "connections.con.xml"
    tll_xml = tmpdir / "tls.add.xml"

    nod_xml.write_text(textwrap.dedent("""\
        <nodes>
            <node id="C" x="0" y="0" type="traffic_light"/>
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

    # Run netconvert
    sumo_bin = _find_sumo_binary()
    netconvert = str(Path(sumo_bin).parent / ("netconvert.exe" if sys.platform == "win32" else "netconvert"))
    subprocess.run(
        [netconvert, "-n", str(nod_xml), "-e", str(edg_xml),
         "-o", str(net_xml), "--no-warnings", "true"],
        check=True, capture_output=True,
    )

    # Routes: uniform demand from each approach
    # 0.3 veh/s from N and S, 0.2 from E and W
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
                <step-length value="{step_length}"/>
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


def _write_grid(tmpdir: Path, rows: int, cols: int,
                sim_seconds: int, step_length: float) -> Path:
    """Generate SUMO files for an NxM grid via netgenerate + randomTrips."""
    net_xml = tmpdir / "net.net.xml"
    trip_xml = tmpdir / "trips.trips.xml"
    rou_xml = tmpdir / "routes.rou.xml"
    cfg_xml = tmpdir / "sim.sumocfg"

    sumo_bin = _find_sumo_binary()
    bin_dir = Path(sumo_bin).parent
    netgenerate = str(bin_dir / ("netgenerate.exe" if sys.platform == "win32" else "netgenerate"))

    subprocess.run(
        [netgenerate, "--grid",
         "--grid.x-number", str(cols),
         "--grid.y-number", str(rows),
         "--grid.x-length", "300",
         "--grid.y-length", "300",
         "--default.lanenumber", "2",
         "--default.speed", "13.89",
         "--tls.guess", "true",
         "-o", str(net_xml),
         "--no-warnings", "true"],
        check=True, capture_output=True,
    )

    # Use SUMO's randomTrips.py to generate valid routes
    # Find randomTrips.py in the SUMO tools directory
    tools_dir = bin_dir.parent / "tools"
    random_trips = tools_dir / "randomTrips.py"
    if not random_trips.exists():
        # pip-installed sumo may have tools elsewhere
        import sumolib
        sumolib_dir = Path(sumolib.__file__).parent
        random_trips = sumolib_dir.parent / "sumo" / "tools" / "randomTrips.py"
    if not random_trips.exists():
        # Fallback: find via share/sumo/tools
        random_trips = bin_dir.parent / "share" / "sumo" / "tools" / "randomTrips.py"

    if random_trips.exists():
        # Generate trips and routes via randomTrips + duarouter
        total_vehs = int(sim_seconds * 1.0)  # ~1 veh/s total
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
        # route file was created as trips.trips.xml with validated routes
        rou_xml = trip_xml
    else:
        # Fallback: manual route generation using sumolib
        import sumolib
        net = sumolib.net.readNet(str(net_xml))
        # Get all edges, find fringe (boundary) edges
        all_edges = net.getEdges()
        # Boundary edges: those where the fromNode or toNode is on the fringe
        fringe_from = []
        fringe_to = []
        for e in all_edges:
            fn = e.getFromNode()
            tn = e.getToNode()
            # Fringe = node with only incoming or only outgoing
            if len(fn.getIncoming()) == 0:
                fringe_from.append(e)
            if len(tn.getOutgoing()) == 0:
                fringe_to.append(e)

        if not fringe_from:
            fringe_from = all_edges[:4]
        if not fringe_to:
            fringe_to = all_edges[-4:]

        flow_lines = ['<routes>',
                      '    <vType id="car" length="5" maxSpeed="13.89" accel="2.6" decel="4.5"/>']
        fid = 0
        rate = max(60, int(3600 / max(len(fringe_from), 1)))
        for e_from in fringe_from[:16]:
            # Pick a destination fringe edge that's different
            e_to = fringe_to[fid % len(fringe_to)]
            if e_from.getID() == e_to.getID():
                e_to = fringe_to[(fid + 1) % len(fringe_to)]
            # Build an explicit route via shortest path
            route_edges = net.getShortestPath(e_from, e_to)
            if route_edges and route_edges[0]:
                edge_ids = " ".join(e.getID() for e in route_edges[0])
                flow_lines.append(
                    f'    <vehicle id="v{fid}" type="car" depart="{fid}" departLane="best">'
                )
                flow_lines.append(f'        <route edges="{edge_ids}"/>')
                flow_lines.append('    </vehicle>')
                # Also add periodic copies
                for t in range(fid + len(fringe_from), sim_seconds, max(len(fringe_from) * 2, 1)):
                    flow_lines.append(
                        f'    <vehicle id="v{fid}_{t}" type="car" depart="{t}" departLane="best">'
                    )
                    flow_lines.append(f'        <route edges="{edge_ids}"/>')
                    flow_lines.append('    </vehicle>')
            fid += 1
        flow_lines.append('</routes>')
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
                <step-length value="{step_length}"/>
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


def _write_arterial(tmpdir: Path, n_intersections: int,
                    sim_seconds: int, step_length: float) -> Path:
    """Generate SUMO files for a linear arterial corridor."""
    net_xml = tmpdir / "net.net.xml"
    rou_xml = tmpdir / "routes.rou.xml"
    cfg_xml = tmpdir / "sim.sumocfg"
    nod_xml = tmpdir / "nodes.nod.xml"
    edg_xml = tmpdir / "edges.edg.xml"

    spacing = 400

    # Nodes: one at each intersection + two endpoints
    nodes = ['<nodes>']
    nodes.append(f'    <node id="W" x="0" y="0" type="priority"/>')
    for i in range(1, n_intersections + 1):
        nodes.append(f'    <node id="I{i}" x="{i * spacing}" y="0" type="traffic_light"/>')
        # Side street endpoints
        nodes.append(f'    <node id="N{i}" x="{i * spacing}" y="200" type="priority"/>')
        nodes.append(f'    <node id="S{i}" x="{i * spacing}" y="-200" type="priority"/>')
    nodes.append(f'    <node id="E" x="{(n_intersections + 1) * spacing}" y="0" type="priority"/>')
    nodes.append('</nodes>')
    nod_xml.write_text("\n".join(nodes))

    # Edges
    edges = ['<edges>']
    # Main arterial
    prev = "W"
    for i in range(1, n_intersections + 1):
        cur = f"I{i}"
        edges.append(f'    <edge id="{prev}_{cur}" from="{prev}" to="{cur}" numLanes="2" speed="13.89"/>')
        edges.append(f'    <edge id="{cur}_{prev}" from="{cur}" to="{prev}" numLanes="2" speed="13.89"/>')
        prev = cur
    edges.append(f'    <edge id="{prev}_E" from="{prev}" to="E" numLanes="2" speed="13.89"/>')
    edges.append(f'    <edge id="E_{prev}" from="E" to="{prev}" numLanes="2" speed="13.89"/>')
    # Side streets
    for i in range(1, n_intersections + 1):
        edges.append(f'    <edge id="N{i}_I{i}" from="N{i}" to="I{i}" numLanes="1" speed="13.89"/>')
        edges.append(f'    <edge id="I{i}_N{i}" from="I{i}" to="N{i}" numLanes="1" speed="13.89"/>')
        edges.append(f'    <edge id="S{i}_I{i}" from="S{i}" to="I{i}" numLanes="1" speed="13.89"/>')
        edges.append(f'    <edge id="I{i}_S{i}" from="I{i}" to="S{i}" numLanes="1" speed="13.89"/>')
    edges.append('</edges>')
    edg_xml.write_text("\n".join(edges))

    # Run netconvert
    sumo_bin = _find_sumo_binary()
    netconvert = str(Path(sumo_bin).parent / ("netconvert.exe" if sys.platform == "win32" else "netconvert"))
    subprocess.run(
        [netconvert, "-n", str(nod_xml), "-e", str(edg_xml),
         "-o", str(net_xml), "--no-warnings", "true"],
        check=True, capture_output=True,
    )

    # Routes
    last_int = f"I{n_intersections}"
    flow_lines = [
        '<routes>',
        '    <vType id="car" length="5" maxSpeed="13.89" accel="2.6" decel="4.5"/>',
        f'    <flow id="f_EB" type="car" from="W_I1" to="{last_int}_E" begin="0" end="{sim_seconds}" vehsPerHour="1440" departLane="best"/>',
        f'    <flow id="f_WB" type="car" from="E_{last_int}" to="I1_W" begin="0" end="{sim_seconds}" vehsPerHour="1440" departLane="best"/>',
    ]
    for i in range(1, n_intersections + 1):
        flow_lines.append(
            f'    <flow id="f_NB{i}" type="car" from="N{i}_I{i}" to="I{i}_S{i}" '
            f'begin="0" end="{sim_seconds}" vehsPerHour="360" departLane="best"/>'
        )
    flow_lines.append('</routes>')
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
                <step-length value="{step_length}"/>
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


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    scenario: str
    n_intersections: int
    sim_seconds: float
    lightsim_wall: float
    lightsim_steps_per_sec: float
    sumo_wall: float
    sumo_steps_per_sec: float
    speedup: float


def _run_sumo(cfg_path: Path) -> float:
    """Run SUMO and return wall-clock time."""
    sumo_bin = _find_sumo_binary()
    t0 = time.perf_counter()
    result = subprocess.run(
        [sumo_bin, "-c", str(cfg_path)],
        capture_output=True, text=True,
    )
    t1 = time.perf_counter()
    if result.returncode != 0:
        raise RuntimeError(f"SUMO failed:\n{result.stderr[:1000]}")
    return t1 - t0


def _run_lightsim_scenario(name: str, n_steps: int) -> float:
    """Run a LightSim scenario and return wall-clock time."""
    from .speed_benchmark import benchmark_scenario
    from ..core.demand import DemandProfile
    from ..core.types import NodeType

    if name == "single-intersection":
        from ..benchmarks.single_intersection import create_single_intersection
        net, demand = create_single_intersection()
    elif name.startswith("grid-"):
        size = int(name.split("-")[1].split("x")[0])
        from ..networks.grid import create_grid_network
        net = create_grid_network(rows=size, cols=size, n_cells_per_link=3)
        demand = []
        for link in net.links.values():
            fn = net.nodes.get(link.from_node)
            if fn and fn.node_type == NodeType.ORIGIN:
                demand.append(DemandProfile(link.link_id, [0.0], [0.1]))
    elif name.startswith("arterial-"):
        n = int(name.split("-")[1])
        from ..networks.arterial import create_arterial_network
        net = create_arterial_network(n_intersections=n, n_cells_per_link=4)
        demand = []
        for link in net.links.values():
            fn = net.nodes.get(link.from_node)
            if fn and fn.node_type == NodeType.ORIGIN:
                demand.append(DemandProfile(link.link_id, [0.0], [0.1]))
    else:
        raise ValueError(f"Unknown scenario: {name}")

    r = benchmark_scenario(name, net, demand, n_steps=n_steps)
    return r.wall_time


def run_comparison(n_steps: int = 3600) -> list[ComparisonResult]:
    """Run LightSim vs SUMO on matched scenarios."""
    scenarios = [
        ("single-intersection", 1, lambda d, s, sl: _write_single_intersection(d, s, sl)),
        ("grid-2x2",            4, lambda d, s, sl: _write_grid(d, 2, 2, s, sl)),
        ("grid-4x4",           16, lambda d, s, sl: _write_grid(d, 4, 4, s, sl)),
        ("grid-6x6",           36, lambda d, s, sl: _write_grid(d, 6, 6, s, sl)),
        ("grid-8x8",           64, lambda d, s, sl: _write_grid(d, 8, 8, s, sl)),
        ("arterial-3",          3, lambda d, s, sl: _write_arterial(d, 3, s, sl)),
        ("arterial-5",          5, lambda d, s, sl: _write_arterial(d, 5, s, sl)),
        ("arterial-10",        10, lambda d, s, sl: _write_arterial(d, 10, s, sl)),
        ("arterial-20",        20, lambda d, s, sl: _write_arterial(d, 20, s, sl)),
    ]

    dt = 1.0  # both simulators use 1s step
    results = []

    for name, n_int, gen_fn in scenarios:
        print(f"  {name}...", end=" ", flush=True)

        # --- LightSim ---
        ls_wall = _run_lightsim_scenario(name, n_steps)
        ls_sps = n_steps / ls_wall

        # --- SUMO ---
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = gen_fn(Path(tmpdir), n_steps, dt)
            sumo_wall = _run_sumo(cfg)
        sumo_sps = n_steps / sumo_wall

        speedup = sumo_wall / ls_wall if ls_wall > 0 else float("inf")

        results.append(ComparisonResult(
            scenario=name,
            n_intersections=n_int,
            sim_seconds=n_steps * dt,
            lightsim_wall=ls_wall,
            lightsim_steps_per_sec=ls_sps,
            sumo_wall=sumo_wall,
            sumo_steps_per_sec=sumo_sps,
            speedup=speedup,
        ))
        print(f"LightSim {ls_wall:.3f}s  SUMO {sumo_wall:.3f}s  {speedup:.0f}x faster")

    return results


def print_comparison(results: list[ComparisonResult]) -> None:
    """Print formatted comparison table."""
    print()
    header = (
        f"{'Scenario':<22} {'Intx':>5} {'SimTime':>8} "
        f"{'LS Wall':>9} {'LS stp/s':>10} "
        f"{'SUMO Wall':>10} {'SUMO stp/s':>11} "
        f"{'Speedup':>8}"
    )
    print("=" * len(header))
    print("LightSim vs SUMO Speed Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.scenario:<22} {r.n_intersections:>5} {r.sim_seconds:>7.0f}s "
            f"{r.lightsim_wall:>9.3f} {r.lightsim_steps_per_sec:>10.0f} "
            f"{r.sumo_wall:>10.3f} {r.sumo_steps_per_sec:>11.0f} "
            f"{r.speedup:>7.0f}x"
        )
    print("-" * len(header))

    avg_speedup = np.mean([r.speedup for r in results])
    min_speedup = min(r.speedup for r in results)
    max_speedup = max(r.speedup for r in results)
    print(f"\nSpeedup over SUMO:  avg={avg_speedup:.0f}x  "
          f"min={min_speedup:.0f}x  max={max_speedup:.0f}x")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LightSim vs SUMO benchmark")
    parser.add_argument("--steps", type=int, default=3600,
                        help="Simulation steps (= seconds at dt=1)")
    args = parser.parse_args()

    print(f"Running LightSim vs SUMO comparison ({args.steps} steps each)...\n")

    # Verify SUMO
    try:
        sumo = _find_sumo_binary()
        print(f"SUMO binary: {sumo}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    results = run_comparison(n_steps=args.steps)
    print_comparison(results)


if __name__ == "__main__":
    main()
