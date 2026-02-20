"""Arterial corridor network generator."""

from __future__ import annotations

from ..core.network import Network
from ..core.types import LinkID, NodeID, NodeType, TurnType


def create_arterial_network(
    n_intersections: int = 5,
    link_length: float = 400.0,
    lanes: int = 2,
    n_cells_per_link: int = 4,
    free_flow_speed: float = 13.89,
    wave_speed: float = 5.56,
    jam_density: float = 0.15,
    capacity: float = 0.5,
    side_street_lanes: int = 1,
    side_street_length: float = 200.0,
) -> Network:
    """Create a linear arterial corridor with side streets.

    Layout (n_intersections=3):
        O--[link]--S1--[link]--S2--[link]--S3--[link]--D
                   |           |           |
                   O           O           O     (side streets, origin/dest pairs)
                   |           |           |
                   D           D           D

    Numbering:
        Main arterial nodes: 0 (origin), 1..n (signalised), n+1 (destination)
        Side street origins: 100+i (north), 200+i (south)
        Side street dests:   300+i (north), 400+i (south)
    """
    net = Network()
    link_counter = 0

    def next_lid() -> LinkID:
        nonlocal link_counter
        lid = LinkID(link_counter)
        link_counter += 1
        return lid

    # Main arterial nodes
    origin_id = NodeID(0)
    dest_id = NodeID(n_intersections + 1)
    net.add_node(origin_id, NodeType.ORIGIN, x=0, y=0)
    net.add_node(dest_id, NodeType.DESTINATION,
                 x=(n_intersections + 1) * link_length, y=0)

    for i in range(1, n_intersections + 1):
        net.add_node(NodeID(i), NodeType.SIGNALIZED,
                     x=i * link_length, y=0)

    # Main arterial links (eastbound)
    # Origin → first intersection
    eb_links = []
    lid = next_lid()
    net.add_link(lid, origin_id, NodeID(1), length=link_length, lanes=lanes,
                 n_cells=n_cells_per_link, free_flow_speed=free_flow_speed,
                 wave_speed=wave_speed, jam_density=jam_density, capacity=capacity)
    eb_links.append(lid)

    for i in range(1, n_intersections):
        lid = next_lid()
        net.add_link(lid, NodeID(i), NodeID(i + 1), length=link_length, lanes=lanes,
                     n_cells=n_cells_per_link, free_flow_speed=free_flow_speed,
                     wave_speed=wave_speed, jam_density=jam_density, capacity=capacity)
        eb_links.append(lid)

    # Last intersection → destination
    lid = next_lid()
    net.add_link(lid, NodeID(n_intersections), dest_id, length=link_length, lanes=lanes,
                 n_cells=n_cells_per_link, free_flow_speed=free_flow_speed,
                 wave_speed=wave_speed, jam_density=jam_density, capacity=capacity)

    # Westbound (reverse) — add origin/dest at the other end
    wb_origin = NodeID(n_intersections + 2)
    wb_dest = NodeID(n_intersections + 3)
    net.add_node(wb_origin, NodeType.ORIGIN,
                 x=(n_intersections + 1) * link_length, y=0)
    net.add_node(wb_dest, NodeType.DESTINATION, x=0, y=0)

    wb_links = []
    lid = next_lid()
    net.add_link(lid, wb_origin, NodeID(n_intersections), length=link_length,
                 lanes=lanes, n_cells=n_cells_per_link,
                 free_flow_speed=free_flow_speed, wave_speed=wave_speed,
                 jam_density=jam_density, capacity=capacity)
    wb_links.append(lid)

    for i in range(n_intersections, 1, -1):
        lid = next_lid()
        net.add_link(lid, NodeID(i), NodeID(i - 1), length=link_length, lanes=lanes,
                     n_cells=n_cells_per_link, free_flow_speed=free_flow_speed,
                     wave_speed=wave_speed, jam_density=jam_density, capacity=capacity)
        wb_links.append(lid)

    lid = next_lid()
    net.add_link(lid, NodeID(1), wb_dest, length=link_length, lanes=lanes,
                 n_cells=n_cells_per_link, free_flow_speed=free_flow_speed,
                 wave_speed=wave_speed, jam_density=jam_density, capacity=capacity)

    # Side streets at each intersection
    for i in range(1, n_intersections + 1):
        int_node = NodeID(i)
        x = i * link_length

        # North side street: origin → intersection, intersection → dest
        n_origin = NodeID(100 + i)
        n_dest = NodeID(300 + i)
        net.add_node(n_origin, NodeType.ORIGIN, x=x, y=side_street_length)
        net.add_node(n_dest, NodeType.DESTINATION, x=x, y=side_street_length)

        nb_in = next_lid()
        net.add_link(nb_in, n_origin, int_node, length=side_street_length,
                     lanes=side_street_lanes, n_cells=max(1, n_cells_per_link // 2),
                     free_flow_speed=free_flow_speed, wave_speed=wave_speed,
                     jam_density=jam_density, capacity=capacity)
        nb_out = next_lid()
        net.add_link(nb_out, int_node, n_dest, length=side_street_length,
                     lanes=side_street_lanes, n_cells=max(1, n_cells_per_link // 2),
                     free_flow_speed=free_flow_speed, wave_speed=wave_speed,
                     jam_density=jam_density, capacity=capacity)

        # South side street
        s_origin = NodeID(200 + i)
        s_dest = NodeID(400 + i)
        net.add_node(s_origin, NodeType.ORIGIN, x=x, y=-side_street_length)
        net.add_node(s_dest, NodeType.DESTINATION, x=x, y=-side_street_length)

        sb_in = next_lid()
        net.add_link(sb_in, s_origin, int_node, length=side_street_length,
                     lanes=side_street_lanes, n_cells=max(1, n_cells_per_link // 2),
                     free_flow_speed=free_flow_speed, wave_speed=wave_speed,
                     jam_density=jam_density, capacity=capacity)
        sb_out = next_lid()
        net.add_link(sb_out, int_node, s_dest, length=side_street_length,
                     lanes=side_street_lanes, n_cells=max(1, n_cells_per_link // 2),
                     free_flow_speed=free_flow_speed, wave_speed=wave_speed,
                     jam_density=jam_density, capacity=capacity)

        # Movements at intersection i
        # Find inbound/outbound links
        inbound_eb = []  # eastbound into this intersection
        inbound_wb = []  # westbound into this intersection
        for link in net.links.values():
            if link.to_node == int_node:
                if link.from_node == n_origin:
                    pass  # side street north
                elif link.from_node == s_origin:
                    pass  # side street south
                # We'll handle all movements below

        # Gather all inbound and outbound
        in_links = [l for l in net.links.values() if l.to_node == int_node]
        out_links = [l for l in net.links.values() if l.from_node == int_node]

        # Through movements for main arterial (EB and WB)
        ew_movs = []
        ns_movs = []

        for il in in_links:
            # Classify by the incoming link's own direction
            il_from = net.nodes[il.from_node]
            il_to = net.nodes[il.to_node]  # the intersection node
            il_dx = il_to.x - il_from.x
            il_dy = il_to.y - il_from.y
            is_incoming_ew = abs(il_dx) >= abs(il_dy) if (abs(il_dx) + abs(il_dy)) > 0 else True

            for ol in out_links:
                if il.link_id == ol.link_id:
                    continue
                turn_ratio = 1.0 / max(len(out_links), 1)
                mov = net.add_movement(
                    from_link=il.link_id,
                    to_link=ol.link_id,
                    node_id=int_node,
                    turn_type=TurnType.THROUGH,
                    turn_ratio=turn_ratio,
                )
                if is_incoming_ew:
                    ew_movs.append(mov.movement_id)
                else:
                    ns_movs.append(mov.movement_id)

        # Two phases: EW (main arterial) and NS (side streets)
        if ew_movs:
            net.add_phase(int_node, ew_movs, min_green=5.0, max_green=60.0)
        if ns_movs:
            net.add_phase(int_node, ns_movs, min_green=5.0, max_green=60.0)

    return net
