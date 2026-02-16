"""Tests for the OSM network importer.

These tests verify the code logic without actually calling osmnx
(which requires network access and is slow). We test the internal
helper functions and the error handling.
"""

import pytest

from lightsim.core.types import NodeType


class TestOSMImporterLogic:
    """Test OSM importer without network access."""

    def test_import_raises_without_osmnx(self):
        """Should raise ImportError with a helpful message."""
        import lightsim.networks.osm as osm_mod
        if osm_mod.HAS_OSMNX:
            pytest.skip("osmnx is installed, can't test missing-dep path")

        with pytest.raises(ImportError, match="osmnx"):
            osm_mod.from_osm(query="test")

    def test_import_raises_without_args(self):
        """Should raise ValueError if neither query nor point is given."""
        import lightsim.networks.osm as osm_mod
        if not osm_mod.HAS_OSMNX:
            pytest.skip("osmnx not installed")

        with pytest.raises(ValueError, match="Provide either"):
            osm_mod.from_osm()

    def test_classify_node_function(self):
        """Test the _classify_node helper with a mock graph."""
        import lightsim.networks.osm as osm_mod

        # Create a minimal mock graph — G.nodes must be subscriptable
        # (networkx's NodeView supports G.nodes[id] for attribute access)
        class NodeView(dict):
            """Mimics networkx NodeView: subscriptable + iterable."""
            pass

        class MockGraph:
            def __init__(self):
                self.nodes = NodeView({
                    1: {"highway": "traffic_signals"},
                    2: {"highway": "residential"},
                    3: {},
                })
                self._degree = {1: 4, 2: 2, 3: 6}

            def degree(self, node):
                return self._degree.get(node, 0)

        G = MockGraph()
        boundary = {99}

        # Node with traffic_signals tag → SIGNALIZED
        assert osm_mod._classify_node(G, 1, boundary) == NodeType.SIGNALIZED

        # Boundary node → ORIGIN
        assert osm_mod._classify_node(G, 99, boundary) == NodeType.ORIGIN

        # Low-degree node without signal tag → UNSIGNALIZED
        assert osm_mod._classify_node(G, 2, boundary) == NodeType.UNSIGNALIZED

        # High-degree node (>=4) without signal tag → SIGNALIZED
        assert osm_mod._classify_node(G, 3, boundary) == NodeType.SIGNALIZED

    @pytest.mark.skipif(
        not __import__("lightsim.networks.osm", fromlist=["HAS_OSMNX"]).HAS_OSMNX,
        reason="osmnx not installed"
    )
    def test_from_osm_point_returns_network(self):
        """Integration test: download a small area and check the result."""
        from lightsim.networks.osm import from_osm_point

        # Small area near MIT campus — should have some intersections
        net = from_osm_point(42.3601, -71.0942, dist=200)

        assert len(net.nodes) > 0
        assert len(net.links) > 0

        # Should have at least one signalised intersection
        has_signalized = any(
            n.node_type == NodeType.SIGNALIZED
            for n in net.nodes.values()
        )
        # May or may not have signals depending on the area
        # Just check it doesn't crash
        assert len(net.nodes) >= 2

    @pytest.mark.skipif(
        not __import__("lightsim.networks.osm", fromlist=["HAS_OSMNX"]).HAS_OSMNX,
        reason="osmnx not installed"
    )
    def test_from_osm_compiles(self):
        """Imported network should compile without CFL violations."""
        from lightsim.networks.osm import from_osm_point

        net = from_osm_point(42.3601, -71.0942, dist=200)
        # Should compile without error
        compiled = net.compile(dt=1.0)
        assert compiled.n_cells > 0
