"""Grid network generator."""

from __future__ import annotations

from ..core.network import Network
from ..core.types import LinkID, MovementID, NodeID, NodeType, TurnType


def create_grid_network(
    rows: int = 4,
    cols: int = 4,
    link_length: float = 300.0,
    lanes: int = 2,
    n_cells_per_link: int = 3,
    free_flow_speed: float = 13.89,
    wave_speed: float = 5.56,
    jam_density: float = 0.15,
    capacity: float = 0.5,
    green_time: float = 30.0,
) -> Network:
    """Create an NxM grid network with signalised intersections.

    Boundary nodes are origins/destinations. Interior nodes are signalised.
    Links run in both directions (NSEW) between adjacent nodes.

    Node numbering: row * (cols + 2) + col, with boundary padding.
    """
    net = Network()

    total_rows = rows + 2  # +2 for boundary
    total_cols = cols + 2

    def nid(r: int, c: int) -> NodeID:
        return NodeID(r * total_cols + c)

    # Create nodes
    for r in range(total_rows):
        for c in range(total_cols):
            is_boundary = r == 0 or r == total_rows - 1 or c == 0 or c == total_cols - 1
            is_corner = (r in (0, total_rows - 1)) and (c in (0, total_cols - 1))
            if is_corner:
                continue  # skip corners

            # Boundary row/col nodes are origins on the perimeter
            if is_boundary:
                # Check if this boundary node connects to an interior node
                has_interior = False
                if r == 0 and 1 <= c <= cols:
                    has_interior = True
                elif r == total_rows - 1 and 1 <= c <= cols:
                    has_interior = True
                elif c == 0 and 1 <= r <= rows:
                    has_interior = True
                elif c == total_cols - 1 and 1 <= r <= rows:
                    has_interior = True

                if has_interior:
                    net.add_node(nid(r, c), NodeType.ORIGIN,
                                x=c * link_length, y=r * link_length)
                    # Also add a destination node (separate ID)
                    dest_id = NodeID(nid(r, c) + 10000)
                    net.add_node(dest_id, NodeType.DESTINATION,
                                x=c * link_length, y=r * link_length)
            else:
                net.add_node(nid(r, c), NodeType.SIGNALIZED,
                             x=c * link_length, y=r * link_length)

    link_counter = 0

    def next_link_id() -> LinkID:
        nonlocal link_counter
        lid = LinkID(link_counter)
        link_counter += 1
        return lid

    # Create links between adjacent nodes
    # Also track inbound/outbound links per intersection for movements
    node_inbound: dict[NodeID, list[tuple[LinkID, str]]] = {}  # direction = N/S/E/W
    node_outbound: dict[NodeID, list[tuple[LinkID, str]]] = {}

    directions = [(0, 1, "E", "W"), (0, -1, "W", "E"),
                  (1, 0, "S", "N"), (-1, 0, "N", "S")]

    for r in range(total_rows):
        for c in range(total_cols):
            from_nid = nid(r, c)
            if from_nid not in net.nodes:
                continue

            for dr, dc, fwd_dir, rev_dir in directions:
                nr, nc = r + dr, c + dc
                to_nid = nid(nr, nc)

                # Check if target is a boundary destination
                is_to_boundary = (nr == 0 or nr == total_rows - 1 or
                                  nc == 0 or nc == total_cols - 1)

                if is_to_boundary:
                    dest_nid = NodeID(to_nid + 10000)
                    if dest_nid in net.nodes:
                        lid = next_link_id()
                        net.add_link(
                            lid, from_nid, dest_nid,
                            length=link_length, lanes=lanes,
                            n_cells=n_cells_per_link,
                            free_flow_speed=free_flow_speed,
                            wave_speed=wave_speed,
                            jam_density=jam_density,
                            capacity=capacity,
                        )
                        node_outbound.setdefault(from_nid, []).append((lid, fwd_dir))
                elif to_nid in net.nodes:
                    lid = next_link_id()
                    net.add_link(
                        lid, from_nid, to_nid,
                        length=link_length, lanes=lanes,
                        n_cells=n_cells_per_link,
                        free_flow_speed=free_flow_speed,
                        wave_speed=wave_speed,
                        jam_density=jam_density,
                        capacity=capacity,
                    )
                    node_outbound.setdefault(from_nid, []).append((lid, fwd_dir))
                    node_inbound.setdefault(to_nid, []).append((lid, fwd_dir))

    # Add links from boundary origins to interior nodes
    for r in range(total_rows):
        for c in range(total_cols):
            is_boundary = r == 0 or r == total_rows - 1 or c == 0 or c == total_cols - 1
            if not is_boundary:
                continue
            origin_nid = nid(r, c)
            if origin_nid not in net.nodes:
                continue

            for dr, dc, fwd_dir, rev_dir in directions:
                nr, nc = r + dr, c + dc
                to_nid = nid(nr, nc)
                if to_nid in net.nodes and net.nodes[to_nid].node_type == NodeType.SIGNALIZED:
                    lid = next_link_id()
                    net.add_link(
                        lid, origin_nid, to_nid,
                        length=link_length, lanes=lanes,
                        n_cells=n_cells_per_link,
                        free_flow_speed=free_flow_speed,
                        wave_speed=wave_speed,
                        jam_density=jam_density,
                        capacity=capacity,
                    )
                    node_inbound.setdefault(to_nid, []).append((lid, fwd_dir))

    # Create movements and phases at signalised intersections
    for node_id, node in net.nodes.items():
        if node.node_type != NodeType.SIGNALIZED:
            continue

        inbound = node_inbound.get(node_id, [])
        outbound = node_outbound.get(node_id, [])

        if not inbound or not outbound:
            continue

        # Create through movements: each inbound → matching outbound direction
        ns_movements = []
        ew_movements = []

        for in_lid, in_dir in inbound:
            for out_lid, out_dir in outbound:
                # Through: N→N (continuing south), S→S (continuing north), etc.
                if in_dir == out_dir:
                    # Same direction = through movement
                    n_out_same_dir = sum(1 for _, d in outbound if d == out_dir)
                    turn_ratio = 1.0 / max(len(outbound), 1)
                    mov = net.add_movement(
                        from_link=in_lid,
                        to_link=out_lid,
                        node_id=node_id,
                        turn_type=TurnType.THROUGH,
                        turn_ratio=turn_ratio,
                    )
                    if in_dir in ("N", "S"):
                        ns_movements.append(mov.movement_id)
                    else:
                        ew_movements.append(mov.movement_id)

        # If no through movements found, assign all movements to phase 1
        if not ns_movements and not ew_movements:
            all_movs = []
            for in_lid, _ in inbound:
                for out_lid, _ in outbound:
                    mov = net.add_movement(
                        from_link=in_lid,
                        to_link=out_lid,
                        node_id=node_id,
                        turn_type=TurnType.THROUGH,
                        turn_ratio=1.0 / max(len(outbound), 1),
                    )
                    all_movs.append(mov.movement_id)
            if all_movs:
                net.add_phase(node_id, all_movs, min_green=5.0, max_green=60.0)
        else:
            # Two phases: NS and EW
            if ns_movements:
                net.add_phase(node_id, ns_movements, min_green=5.0, max_green=60.0)
            if ew_movements:
                net.add_phase(node_id, ew_movements, min_green=5.0, max_green=60.0)

    return net
