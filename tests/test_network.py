"""Tests for network validation and summary methods."""

import pytest

from lightsim.core.network import Network
from lightsim.core.types import LinkID, MovementID, NodeID, NodeType, TurnType


class TestNetworkValidation:
    """Test Network.validate() error detection."""

    def test_valid_network_no_errors(self):
        """A properly constructed network should have no errors."""
        net = Network()
        net.add_node(NodeID(0), NodeType.SIGNALIZED)
        net.add_node(NodeID(1), NodeType.ORIGIN)
        net.add_node(NodeID(2), NodeType.DESTINATION)

        vf = 13.89
        net.add_link(LinkID(0), NodeID(1), NodeID(0),
                     length=vf * 3, lanes=1, n_cells=3,
                     free_flow_speed=vf, wave_speed=5.56,
                     jam_density=0.15, capacity=0.5)
        net.add_link(LinkID(1), NodeID(0), NodeID(2),
                     length=vf * 3, lanes=1, n_cells=3,
                     free_flow_speed=vf, wave_speed=5.56,
                     jam_density=0.15, capacity=0.5)

        m = net.add_movement(LinkID(0), LinkID(1), NodeID(0),
                             TurnType.THROUGH, 1.0)
        net.add_phase(NodeID(0), [m.movement_id])

        errors = net.validate()
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_signalized_no_phases(self):
        """Signalized node without phases should be flagged."""
        net = Network()
        net.add_node(NodeID(0), NodeType.SIGNALIZED)
        net.add_node(NodeID(1), NodeType.ORIGIN)
        net.add_link(LinkID(0), NodeID(1), NodeID(0),
                     length=100.0, lanes=1)
        errors = net.validate()
        assert any("no phases" in e for e in errors)

    def test_disconnected_node(self):
        """Node with no links should be flagged."""
        net = Network()
        net.add_node(NodeID(0), NodeType.SIGNALIZED)
        net.add_node(NodeID(1), NodeType.ORIGIN)
        # NodeID(0) has no links
        errors = net.validate()
        assert any("disconnected" in e for e in errors)

    def test_invalid_turn_ratio(self):
        """Turn ratio outside (0,1] should be flagged."""
        net = Network()
        net.add_node(NodeID(0), NodeType.SIGNALIZED)
        net.add_node(NodeID(1), NodeType.ORIGIN)
        net.add_node(NodeID(2), NodeType.DESTINATION)
        net.add_link(LinkID(0), NodeID(1), NodeID(0), length=100.0, lanes=1)
        net.add_link(LinkID(1), NodeID(0), NodeID(2), length=100.0, lanes=1)
        net.add_movement(LinkID(0), LinkID(1), NodeID(0),
                         TurnType.THROUGH, turn_ratio=0.0)
        errors = net.validate()
        assert any("turn_ratio" in e for e in errors)


class TestCompiledNetworkSummary:
    """Test CompiledNetwork.summary()."""

    def test_summary_keys(self):
        """Summary should contain expected keys."""
        net = Network()
        net.add_node(NodeID(0), NodeType.ORIGIN)
        net.add_node(NodeID(1), NodeType.DESTINATION)
        net.add_link(LinkID(0), NodeID(0), NodeID(1),
                     length=100.0, lanes=2, n_cells=5,
                     free_flow_speed=13.89)
        compiled = net.compile(dt=1.0)
        s = compiled.summary()
        assert s["n_cells"] == 5
        assert s["n_links"] == 1
        assert s["n_movements"] == 0
        assert s["n_signalized_nodes"] == 0
        assert s["total_lane_metres"] == pytest.approx(200.0, rel=0.01)

    def test_summary_with_intersection(self):
        """Summary for a network with an intersection."""
        net = Network()
        vf = 13.89
        net.add_node(NodeID(0), NodeType.SIGNALIZED)
        net.add_node(NodeID(1), NodeType.ORIGIN)
        net.add_node(NodeID(2), NodeType.DESTINATION)
        net.add_link(LinkID(0), NodeID(1), NodeID(0),
                     length=vf * 3, lanes=1, n_cells=3,
                     free_flow_speed=vf, wave_speed=5.56,
                     jam_density=0.15, capacity=0.5)
        net.add_link(LinkID(1), NodeID(0), NodeID(2),
                     length=vf * 3, lanes=1, n_cells=3,
                     free_flow_speed=vf, wave_speed=5.56,
                     jam_density=0.15, capacity=0.5)
        m = net.add_movement(LinkID(0), LinkID(1), NodeID(0),
                             TurnType.THROUGH, 1.0)
        net.add_phase(NodeID(0), [m.movement_id])
        compiled = net.compile(dt=1.0)
        s = compiled.summary()
        assert s["n_cells"] == 6
        assert s["n_links"] == 2
        assert s["n_movements"] == 1
        assert s["n_signalized_nodes"] == 1
        assert s["n_phases"] == 1
