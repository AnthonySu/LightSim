"""JSON export/import for Network and DemandProfile objects.

Enables reproducible scenario sharing without code dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..core.demand import DemandProfile
from ..core.network import Network
from ..core.types import LinkID, MovementID, NodeID, NodeType, TurnType


def network_to_dict(
    network: Network,
    demand: list[DemandProfile] | None = None,
) -> dict:
    """Serialize a Network (and optional demand) to a plain dict."""
    nodes = []
    for nid, node in network.nodes.items():
        nodes.append({
            "id": int(nid),
            "type": node.node_type.name,
            "x": node.x,
            "y": node.y,
        })

    links = []
    for lid, link in network.links.items():
        cell = link.cells[0] if link.cells else None
        links.append({
            "id": int(lid),
            "from_node": int(link.from_node),
            "to_node": int(link.to_node),
            "length": cell.length * len(link.cells) if cell else 0,
            "lanes": cell.lanes if cell else 1,
            "n_cells": len(link.cells),
            "free_flow_speed": cell.free_flow_speed if cell else 13.89,
            "wave_speed": cell.wave_speed if cell else 5.56,
            "jam_density": cell.jam_density if cell else 0.15,
            "capacity": cell.capacity if cell else 0.5,
        })

    movements = []
    for mid, mov in network.movements.items():
        movements.append({
            "id": int(mid),
            "from_link": int(mov.from_link),
            "to_link": int(mov.to_link),
            "node_id": int(mov.node_id),
            "turn_type": mov.turn_type.name,
            "turn_ratio": mov.turn_ratio,
            "saturation_rate": mov.saturation_rate,
        })

    phases = []
    for nid, node in network.nodes.items():
        for phase in node.phases:
            phases.append({
                "id": int(phase.phase_id),
                "node_id": int(nid),
                "movements": [int(m) for m in phase.movements],
                "min_green": phase.min_green,
                "max_green": phase.max_green,
                "yellow": phase.yellow,
                "all_red": phase.all_red,
                "lost_time": phase.lost_time,
            })

    result = {
        "version": 1,
        "nodes": nodes,
        "links": links,
        "movements": movements,
        "phases": phases,
    }

    if demand:
        result["demand"] = [
            {
                "link_id": int(d.link_id),
                "time_points": d.time_points,
                "flow_rates": d.flow_rates,
            }
            for d in demand
        ]

    return result


def dict_to_network(data: dict) -> tuple[Network, list[DemandProfile]]:
    """Deserialize a Network and DemandProfiles from a dict."""
    net = Network()

    for nd in data["nodes"]:
        net.add_node(
            NodeID(nd["id"]),
            NodeType[nd["type"]],
            x=nd.get("x", 0.0),
            y=nd.get("y", 0.0),
        )

    for ld in data["links"]:
        net.add_link(
            LinkID(ld["id"]),
            NodeID(ld["from_node"]),
            NodeID(ld["to_node"]),
            length=ld["length"],
            lanes=ld.get("lanes", 1),
            n_cells=ld.get("n_cells", 1),
            free_flow_speed=ld.get("free_flow_speed", 13.89),
            wave_speed=ld.get("wave_speed", 5.56),
            jam_density=ld.get("jam_density", 0.15),
            capacity=ld.get("capacity", 0.5),
        )

    # Movements need to be added in order to match IDs
    mov_data = sorted(data.get("movements", []), key=lambda m: m["id"])
    for md in mov_data:
        net.add_movement(
            LinkID(md["from_link"]),
            LinkID(md["to_link"]),
            NodeID(md["node_id"]),
            turn_type=TurnType[md.get("turn_type", "THROUGH")],
            turn_ratio=md.get("turn_ratio", 1.0),
            saturation_rate=md.get("saturation_rate"),
        )

    # Phases need to be added in order
    phase_data = sorted(data.get("phases", []), key=lambda p: p["id"])
    for pd in phase_data:
        net.add_phase(
            NodeID(pd["node_id"]),
            [MovementID(m) for m in pd["movements"]],
            min_green=pd.get("min_green", 5.0),
            max_green=pd.get("max_green", 60.0),
            yellow=pd.get("yellow", 3.0),
            all_red=pd.get("all_red", 2.0),
            lost_time=pd.get("lost_time", 0.0),
        )

    demand = []
    for dd in data.get("demand", []):
        demand.append(DemandProfile(
            LinkID(dd["link_id"]),
            dd["time_points"],
            dd["flow_rates"],
        ))

    return net, demand


def save_scenario(
    path: str | Path,
    network: Network,
    demand: list[DemandProfile] | None = None,
) -> None:
    """Save a scenario to a JSON file."""
    data = network_to_dict(network, demand)
    Path(path).write_text(json.dumps(data, indent=2))


def load_scenario(path: str | Path) -> tuple[Network, list[DemandProfile]]:
    """Load a scenario from a JSON file."""
    data = json.loads(Path(path).read_text())
    return dict_to_network(data)
