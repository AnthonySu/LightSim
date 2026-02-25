"""Tests for network generators: grid, arterial, and from_dict."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from lightsim.core.types import NodeType
from lightsim.networks.grid import create_grid_network
from lightsim.networks.arterial import create_arterial_network
from lightsim.networks.from_dict import from_dict, from_json


# ---------------------------------------------------------------------------
# Grid network tests
# ---------------------------------------------------------------------------


class TestGridNetwork:
    def test_default_grid(self):
        net = create_grid_network()
        # 4x4 interior signalized nodes
        signal_nodes = [n for n in net.nodes.values() if n.node_type == NodeType.SIGNALIZED]
        assert len(signal_nodes) == 16

    def test_custom_grid_size(self):
        net = create_grid_network(rows=2, cols=3)
        signal_nodes = [n for n in net.nodes.values() if n.node_type == NodeType.SIGNALIZED]
        assert len(signal_nodes) == 6

    def test_grid_has_origins_and_destinations(self):
        net = create_grid_network(rows=2, cols=2)
        origins = [n for n in net.nodes.values() if n.node_type == NodeType.ORIGIN]
        dests = [n for n in net.nodes.values() if n.node_type == NodeType.DESTINATION]
        assert len(origins) > 0
        assert len(dests) > 0

    def test_grid_has_links(self):
        net = create_grid_network(rows=2, cols=2)
        assert len(net.links) > 0

    def test_grid_has_movements_and_phases(self):
        net = create_grid_network(rows=2, cols=2)
        assert len(net.movements) > 0
        signal_nodes = [n for n in net.nodes.values() if n.node_type == NodeType.SIGNALIZED]
        for node in signal_nodes:
            assert len(node.phases) >= 1, f"Node {node.node_id} has no phases"

    def test_grid_compiles(self):
        net = create_grid_network(rows=2, cols=2)
        compiled = net.compile(dt=1.0)
        assert compiled.n_cells > 0
        assert compiled.n_movements > 0

    def test_grid_1x1(self):
        """Smallest possible grid: single intersection."""
        net = create_grid_network(rows=1, cols=1)
        signal_nodes = [n for n in net.nodes.values() if n.node_type == NodeType.SIGNALIZED]
        assert len(signal_nodes) == 1

    def test_grid_custom_link_params(self):
        net = create_grid_network(
            rows=1, cols=1,
            link_length=500.0,
            lanes=3,
            n_cells_per_link=5,
            free_flow_speed=20.0,
        )
        # Each link should have the specified number of cells
        for link in net.links.values():
            assert link.num_cells == 5


# ---------------------------------------------------------------------------
# Arterial network tests
# ---------------------------------------------------------------------------


class TestArterialNetwork:
    def test_default_arterial(self):
        net = create_arterial_network()
        signal_nodes = [n for n in net.nodes.values() if n.node_type == NodeType.SIGNALIZED]
        assert len(signal_nodes) == 5

    def test_custom_arterial_size(self):
        net = create_arterial_network(n_intersections=3)
        signal_nodes = [n for n in net.nodes.values() if n.node_type == NodeType.SIGNALIZED]
        assert len(signal_nodes) == 3

    def test_arterial_has_side_streets(self):
        """Each intersection has N and S side streets (2 origins + 2 dests)."""
        net = create_arterial_network(n_intersections=2)
        origins = [n for n in net.nodes.values() if n.node_type == NodeType.ORIGIN]
        # 1 main EB origin + 1 WB origin + 2 N side + 2 S side = 6
        assert len(origins) >= 6

    def test_arterial_has_movements(self):
        net = create_arterial_network(n_intersections=2)
        assert len(net.movements) > 0

    def test_arterial_has_two_phases_per_intersection(self):
        net = create_arterial_network(n_intersections=3)
        for node in net.nodes.values():
            if node.node_type == NodeType.SIGNALIZED:
                assert len(node.phases) == 2, (
                    f"Node {node.node_id} has {len(node.phases)} phases, expected 2"
                )

    def test_arterial_compiles(self):
        net = create_arterial_network(n_intersections=2)
        compiled = net.compile(dt=1.0)
        assert compiled.n_cells > 0

    def test_arterial_single_intersection(self):
        net = create_arterial_network(n_intersections=1)
        signal_nodes = [n for n in net.nodes.values() if n.node_type == NodeType.SIGNALIZED]
        assert len(signal_nodes) == 1


# ---------------------------------------------------------------------------
# from_dict / from_json tests
# ---------------------------------------------------------------------------


class TestFromDict:
    @pytest.fixture
    def simple_spec(self) -> dict:
        """Minimal single-intersection network spec."""
        return {
            "nodes": [
                {"id": 0, "type": "origin", "x": 0, "y": 0},
                {"id": 1, "type": "signalized", "x": 300, "y": 0},
                {"id": 2, "type": "destination", "x": 600, "y": 0},
                {"id": 3, "type": "origin", "x": 300, "y": 300},
                {"id": 4, "type": "destination", "x": 300, "y": -300},
            ],
            "links": [
                {"id": 0, "from": 0, "to": 1, "length": 300, "lanes": 2, "n_cells": 3},
                {"id": 1, "from": 1, "to": 2, "length": 300, "lanes": 2, "n_cells": 3},
                {"id": 2, "from": 3, "to": 1, "length": 300, "lanes": 1, "n_cells": 3},
                {"id": 3, "from": 1, "to": 4, "length": 300, "lanes": 1, "n_cells": 3},
            ],
            "movements": [
                {"from_link": 0, "to_link": 1, "node": 1, "turn_type": "through", "turn_ratio": 0.7},
                {"from_link": 0, "to_link": 3, "node": 1, "turn_type": "right", "turn_ratio": 0.3},
                {"from_link": 2, "to_link": 1, "node": 1, "turn_type": "left", "turn_ratio": 0.3},
                {"from_link": 2, "to_link": 3, "node": 1, "turn_type": "through", "turn_ratio": 0.7},
            ],
            "phases": [
                {"node": 1, "movements": [0, 1], "min_green": 5, "max_green": 60},
                {"node": 1, "movements": [2, 3], "min_green": 5, "max_green": 60},
            ],
        }

    def test_from_dict_creates_network(self, simple_spec):
        net = from_dict(simple_spec)
        assert len(net.nodes) == 5
        assert len(net.links) == 4
        assert len(net.movements) == 4

    def test_from_dict_node_types(self, simple_spec):
        net = from_dict(simple_spec)
        assert net.nodes[0].node_type == NodeType.ORIGIN
        assert net.nodes[1].node_type == NodeType.SIGNALIZED
        assert net.nodes[2].node_type == NodeType.DESTINATION

    def test_from_dict_phases(self, simple_spec):
        net = from_dict(simple_spec)
        node = net.nodes[1]
        assert len(node.phases) == 2

    def test_from_dict_compiles(self, simple_spec):
        net = from_dict(simple_spec)
        compiled = net.compile(dt=1.0)
        assert compiled.n_cells > 0

    def test_from_dict_empty_spec(self):
        """Empty spec should produce empty network."""
        net = from_dict({})
        assert len(net.nodes) == 0
        assert len(net.links) == 0

    def test_from_dict_nodes_only(self):
        spec = {
            "nodes": [
                {"id": 0, "type": "origin"},
                {"id": 1, "type": "destination"},
            ]
        }
        net = from_dict(spec)
        assert len(net.nodes) == 2
        assert len(net.links) == 0

    def test_from_dict_custom_link_params(self):
        spec = {
            "nodes": [
                {"id": 0, "type": "origin"},
                {"id": 1, "type": "destination"},
            ],
            "links": [
                {
                    "id": 0, "from": 0, "to": 1,
                    "length": 500, "lanes": 3, "n_cells": 5,
                    "free_flow_speed": 20.0, "wave_speed": 8.0,
                    "jam_density": 0.2, "capacity": 0.8,
                },
            ],
        }
        net = from_dict(spec)
        link = net.links[0]
        assert link.num_cells == 5

    def test_from_json_file(self, simple_spec, tmp_path):
        json_path = tmp_path / "test_network.json"
        json_path.write_text(json.dumps(simple_spec))
        net = from_json(json_path)
        assert len(net.nodes) == 5
        assert len(net.links) == 4

    def test_from_dict_coordinates(self, simple_spec):
        net = from_dict(simple_spec)
        assert net.nodes[0].x == 0.0
        assert net.nodes[1].x == 300.0

    def test_from_dict_default_coordinates(self):
        """Nodes without x/y should default to (0, 0)."""
        spec = {"nodes": [{"id": 0, "type": "origin"}]}
        net = from_dict(spec)
        assert net.nodes[0].x == 0.0
        assert net.nodes[0].y == 0.0
