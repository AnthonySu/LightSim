"""OpenStreetMap network importer via osmnx.

Converts an OSM road network into a LightSim Network with signalised
intersections, through-movements, and two-phase signal plans.

Requires: ``pip install lightsim[osm]`` (installs osmnx).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core.network import Network
from ..core.types import LinkID, NodeID, NodeType, TurnType

try:
    import osmnx as ox
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False


def _classify_node(
    G,
    node: int,
    boundary_nodes: set[int],
    signal_tags: set[str] | None = None,
) -> NodeType:
    """Determine node type from OSM tags and graph topology."""
    if node in boundary_nodes:
        return NodeType.ORIGIN  # boundary nodes are origins/destinations

    data = G.nodes[node]
    highway = data.get("highway", "")

    # Check for traffic signal tag
    if signal_tags is None:
        signal_tags = {"traffic_signals", "traffic_signals;crossing"}

    if highway in signal_tags:
        return NodeType.SIGNALIZED

    # Nodes with degree >= 4 (intersection) default to signalized
    degree = G.degree(node)
    if degree >= 4:
        return NodeType.SIGNALIZED

    return NodeType.UNSIGNALIZED


def from_osm(
    query: str | tuple[float, float, float, float] | None = None,
    *,
    point: tuple[float, float] | None = None,
    dist: float = 1000.0,
    network_type: str = "drive",
    free_flow_speed: float = 13.89,
    wave_speed: float = 5.56,
    jam_density: float = 0.15,
    capacity: float = 0.5,
    min_cell_length: float = 15.0,
    signal_degree_threshold: int = 4,
    simplify: bool = True,
) -> Network:
    """Import a road network from OpenStreetMap.

    Parameters
    ----------
    query : str or (north, south, east, west), optional
        Place name (geocoded) or bounding box.  Exactly one of ``query``
        or ``point`` must be given.
    point : (lat, lon), optional
        Centre point; used with ``dist`` to define a bounding circle.
    dist : float
        Radius in metres (used with ``point``).
    network_type : str
        OSM network type passed to osmnx (``"drive"``, ``"walk"``, etc.).
    free_flow_speed : float
        Default free-flow speed in m/s for links without ``maxspeed``.
    wave_speed : float
        Backward wave speed (m/s).
    jam_density : float
        Jam density (veh/m/lane).
    capacity : float
        Capacity per lane (veh/s).
    min_cell_length : float
        Minimum cell length in metres.  Short links are collapsed to 1 cell.
    signal_degree_threshold : int
        Nodes with degree >= this are treated as signalised if they lack
        an explicit traffic_signals tag.
    simplify : bool
        Whether to simplify the graph (merge degree-2 nodes).

    Returns
    -------
    Network
        A LightSim Network with signalised intersections, movements, and
        two-phase signal plans.
    """
    if not HAS_OSMNX:
        raise ImportError(
            "osmnx is required for OSM import. "
            "Install with: pip install lightsim[osm]"
        )

    # --- 1. Download graph ---
    if point is not None:
        G = ox.graph_from_point(point, dist=dist, network_type=network_type,
                                simplify=simplify)
    elif isinstance(query, tuple) and len(query) == 4:
        north, south, east, west = query
        G = ox.graph_from_bbox(north, south, east, west,
                               network_type=network_type, simplify=simplify)
    elif isinstance(query, str):
        G = ox.graph_from_place(query, network_type=network_type,
                                simplify=simplify)
    else:
        raise ValueError("Provide either `query` (str or bbox) or `point`.")

    # Project to UTM for metre coordinates
    G = ox.project_graph(G)

    # --- 2. Identify boundary nodes (dead-ends + spatial periphery) ---
    boundary_nodes = set()
    for node in G.nodes():
        # Nodes with only in-edges or only out-edges are boundary
        in_deg = G.in_degree(node) if G.is_directed() else G.degree(node)
        out_deg = G.out_degree(node) if G.is_directed() else G.degree(node)
        if in_deg == 0 or out_deg == 0:
            boundary_nodes.add(node)
        # Also treat degree-1 nodes as boundary
        total = in_deg + out_deg if G.is_directed() else G.degree(node)
        if total <= 1:
            boundary_nodes.add(node)

    # Spatial boundary detection: in dense grids the degree-based check
    # misses nodes at the edge of the clipped area.  Pick the closest
    # nodes to each bbox edge so every side has entry/exit points.
    if len(boundary_nodes) < 4:
        nodes_xy = [(n, G.nodes[n].get("x", 0.0), G.nodes[n].get("y", 0.0))
                     for n in G.nodes()]
        if nodes_xy:
            xs = [x for _, x, _ in nodes_xy]
            ys = [y for _, _, y in nodes_xy]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            per_side = 1
            # For each bbox edge, pick the closest `per_side` nodes
            edges = [
                (lambda _, y: y - y_min),          # south
                (lambda _, y: y_max - y),           # north
                (lambda x, _: x - x_min),           # west
                (lambda x, _: x_max - x),           # east
            ]
            for dist_fn in edges:
                ranked = sorted(nodes_xy, key=lambda t: dist_fn(t[1], t[2]))
                for n, _, _ in ranked[:per_side]:
                    boundary_nodes.add(n)

    # --- 3. Build LightSim Network ---
    net = Network()

    # Map OSM node IDs → sequential NodeIDs
    osm_to_nid: dict[int, NodeID] = {}
    # For boundary nodes, we need both origin and destination
    boundary_dest_offset = len(G.nodes()) + 1000000

    for i, osm_node in enumerate(G.nodes()):
        nid = NodeID(i)
        osm_to_nid[osm_node] = nid

        data = G.nodes[osm_node]
        x = data.get("x", 0.0)
        y = data.get("y", 0.0)

        ntype = _classify_node(G, osm_node, boundary_nodes)
        if ntype == NodeType.ORIGIN:
            # Add both origin and destination at boundary
            net.add_node(nid, NodeType.ORIGIN, x=x, y=y)
            dest_nid = NodeID(i + boundary_dest_offset)
            net.add_node(dest_nid, NodeType.DESTINATION, x=x, y=y)
        else:
            net.add_node(nid, ntype, x=x, y=y)

    # --- 4. Create links ---
    link_counter = 0
    # Track inbound/outbound links per signalised node
    node_inbound: dict[NodeID, list[LinkID]] = {}
    node_outbound: dict[NodeID, list[LinkID]] = {}

    for u, v, edge_data in G.edges(data=True):
        from_nid = osm_to_nid[u]
        to_nid = osm_to_nid[v]

        # If destination is a boundary node, point to its destination node
        actual_to = to_nid
        to_node = net.nodes[to_nid]
        if to_node.node_type == NodeType.ORIGIN:
            # Route to the destination version of this boundary node
            dest_nid = NodeID(to_nid + boundary_dest_offset)
            if dest_nid in net.nodes:
                actual_to = dest_nid

        length = edge_data.get("length", 100.0)
        lanes = edge_data.get("lanes", 1)
        if isinstance(lanes, list):
            lanes = int(lanes[0])
        elif isinstance(lanes, str):
            try:
                lanes = int(lanes)
            except ValueError:
                lanes = 1
        lanes = max(1, int(lanes))

        # Parse maxspeed
        maxspeed = edge_data.get("maxspeed", None)
        vf = free_flow_speed
        if maxspeed is not None:
            if isinstance(maxspeed, list):
                maxspeed = maxspeed[0]
            if isinstance(maxspeed, str):
                try:
                    # km/h → m/s
                    vf = float(maxspeed.split()[0]) / 3.6
                except (ValueError, IndexError):
                    pass
            elif isinstance(maxspeed, (int, float)):
                vf = float(maxspeed) / 3.6

        # Number of cells: ensure cell_length >= vf * dt (CFL condition)
        # For very short links, use 1 cell and cap vf to satisfy CFL
        n_cells = max(1, int(length / max(min_cell_length, vf * 1.0)))
        cell_len = length / n_cells
        if cell_len < vf * 1.0:
            n_cells = 1
            cell_len = length
        # If link is shorter than vf, reduce vf to satisfy CFL
        if length < vf * 1.0:
            vf = length * 0.95  # 5% margin

        lid = LinkID(link_counter)
        link_counter += 1

        net.add_link(
            lid,
            from_node=from_nid,
            to_node=actual_to,
            length=length,
            lanes=lanes,
            n_cells=n_cells,
            free_flow_speed=vf,
            wave_speed=wave_speed,
            jam_density=jam_density,
            capacity=capacity,
        )

        # Track connectivity for movement/phase generation
        if actual_to in net.nodes and net.nodes[actual_to].node_type == NodeType.SIGNALIZED:
            node_inbound.setdefault(actual_to, []).append(lid)
        if from_nid in net.nodes and net.nodes[from_nid].node_type == NodeType.SIGNALIZED:
            node_outbound.setdefault(from_nid, []).append(lid)

    # --- 5. Generate movements and phases at signalised intersections ---
    for node_id, node in net.nodes.items():
        if node.node_type != NodeType.SIGNALIZED:
            continue

        inbound = node_inbound.get(node_id, [])
        outbound = node_outbound.get(node_id, [])

        if not inbound or not outbound:
            continue

        # Create through movements for each inbound → outbound pair
        # Classify by angle into two groups for two-phase signal
        group_a_movs = []
        group_b_movs = []

        for in_lid in inbound:
            in_link = net.links[in_lid]
            from_node = net.nodes[in_link.from_node]
            # Inbound direction vector
            in_dx = node.x - from_node.x
            in_dy = node.y - from_node.y
            in_angle = math.atan2(in_dy, in_dx)

            for out_lid in outbound:
                out_link = net.links[out_lid]
                to_node = net.nodes[out_link.to_node]
                # Don't create U-turn on same edge
                if out_link.to_node == in_link.from_node:
                    continue

                turn_ratio = 1.0 / max(len(outbound), 1)
                mov = net.add_movement(
                    from_link=in_lid,
                    to_link=out_lid,
                    node_id=node_id,
                    turn_type=TurnType.THROUGH,
                    turn_ratio=turn_ratio,
                )

                # Split into two groups by inbound angle:
                # roughly NS vs EW (0/pi vs pi/2/-pi/2)
                angle_deg = math.degrees(in_angle) % 360
                if (315 <= angle_deg or angle_deg < 45) or (135 <= angle_deg < 225):
                    group_a_movs.append(mov.movement_id)
                else:
                    group_b_movs.append(mov.movement_id)

        # Create two-phase signal plan
        if group_a_movs and group_b_movs:
            net.add_phase(node_id, group_a_movs)
            net.add_phase(node_id, group_b_movs)
        elif group_a_movs:
            net.add_phase(node_id, group_a_movs)
        elif group_b_movs:
            net.add_phase(node_id, group_b_movs)

    return net


def from_osm_bbox(
    north: float,
    south: float,
    east: float,
    west: float,
    **kwargs,
) -> Network:
    """Convenience wrapper: import from a lat/lon bounding box."""
    return from_osm(query=(north, south, east, west), **kwargs)


def from_osm_point(
    lat: float,
    lon: float,
    dist: float = 1000.0,
    **kwargs,
) -> Network:
    """Convenience wrapper: import from a centre point + radius."""
    return from_osm(point=(lat, lon), dist=dist, **kwargs)


def generate_demand(
    net: Network,
    rate: float = 0.3,
) -> list["DemandProfile"]:
    """Auto-generate constant demand for all origin links in the network.

    Parameters
    ----------
    net : Network
        A LightSim network (typically from OSM import).
    rate : float
        Constant flow rate in veh/s for each origin link.

    Returns
    -------
    list[DemandProfile]
        One DemandProfile per origin link.
    """
    from ..core.demand import DemandProfile

    profiles = []
    for link_id, link in net.links.items():
        from_node = net.nodes.get(link.from_node)
        if from_node is not None and from_node.node_type == NodeType.ORIGIN:
            profiles.append(
                DemandProfile(link_id, [0.0], [rate])
            )
    return profiles
