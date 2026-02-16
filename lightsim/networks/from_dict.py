"""Build a Network from a JSON/YAML dictionary specification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.network import Network
from ..core.types import LinkID, NodeID, NodeType, TurnType


def _parse_node_type(s: str) -> NodeType:
    return NodeType[s.upper()]


def _parse_turn_type(s: str) -> TurnType:
    return TurnType[s.upper()]


def from_dict(spec: dict[str, Any]) -> Network:
    """Build a Network from a dictionary specification.

    Expected format::

        {
            "nodes": [
                {"id": 0, "type": "origin", "x": 0, "y": 0},
                {"id": 1, "type": "signalized", "x": 300, "y": 0},
                ...
            ],
            "links": [
                {
                    "id": 0, "from": 0, "to": 1,
                    "length": 300, "lanes": 2, "n_cells": 3,
                    "free_flow_speed": 13.89, "wave_speed": 5.56,
                    "jam_density": 0.15, "capacity": 0.5
                },
                ...
            ],
            "movements": [
                {
                    "from_link": 0, "to_link": 1, "node": 1,
                    "turn_type": "through", "turn_ratio": 0.5
                },
                ...
            ],
            "phases": [
                {
                    "node": 1, "movements": [0, 1],
                    "min_green": 5, "max_green": 60, "yellow": 3, "all_red": 2
                },
                ...
            ]
        }
    """
    net = Network()

    # Nodes
    for n in spec.get("nodes", []):
        net.add_node(
            NodeID(n["id"]),
            _parse_node_type(n["type"]),
            x=n.get("x", 0.0),
            y=n.get("y", 0.0),
        )

    # Links
    for l in spec.get("links", []):
        net.add_link(
            LinkID(l["id"]),
            from_node=NodeID(l["from"]),
            to_node=NodeID(l["to"]),
            length=l["length"],
            lanes=l.get("lanes", 1),
            n_cells=l.get("n_cells"),
            free_flow_speed=l.get("free_flow_speed", 13.89),
            wave_speed=l.get("wave_speed", 5.56),
            jam_density=l.get("jam_density", 0.15),
            capacity=l.get("capacity", 0.5),
        )

    # Movements
    mov_id_map: dict[int, int] = {}  # user-specified index → internal MovementID
    for i, m in enumerate(spec.get("movements", [])):
        mov = net.add_movement(
            from_link=LinkID(m["from_link"]),
            to_link=LinkID(m["to_link"]),
            node_id=NodeID(m["node"]),
            turn_type=_parse_turn_type(m.get("turn_type", "through")),
            turn_ratio=m.get("turn_ratio", 1.0),
            saturation_rate=m.get("saturation_rate"),
        )
        mov_id_map[i] = mov.movement_id

    # Phases
    for p in spec.get("phases", []):
        mov_ids = [mov_id_map[i] for i in p["movements"]]
        net.add_phase(
            NodeID(p["node"]),
            movements=mov_ids,
            min_green=p.get("min_green", 5.0),
            max_green=p.get("max_green", 60.0),
            yellow=p.get("yellow", 3.0),
            all_red=p.get("all_red", 2.0),
        )

    return net


def from_json(path: str | Path) -> Network:
    """Load a network from a JSON file."""
    with open(path) as f:
        spec = json.load(f)
    return from_dict(spec)


def from_yaml(path: str | Path) -> Network:
    """Load a network from a YAML file (requires PyYAML)."""
    import yaml
    with open(path) as f:
        spec = yaml.safe_load(f)
    return from_dict(spec)
