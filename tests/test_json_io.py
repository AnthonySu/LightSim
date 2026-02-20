"""Tests for JSON scenario export/import round-trip."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from lightsim.benchmarks.single_intersection import create_single_intersection
from lightsim.core.engine import SimulationEngine
from lightsim.io.json_io import (
    dict_to_network,
    load_scenario,
    network_to_dict,
    save_scenario,
)


class TestJsonRoundTrip:
    """Test that export → import produces identical simulation results."""

    def test_single_intersection_round_trip(self):
        """Single intersection scenario survives JSON round-trip."""
        net, demand = create_single_intersection()

        # Export
        data = network_to_dict(net, demand)
        json_str = json.dumps(data)
        data_back = json.loads(json_str)

        # Import
        net2, demand2 = dict_to_network(data_back)

        # Both should produce identical simulation results
        engine1 = SimulationEngine(network=net, dt=1.0, demand_profiles=demand)
        engine1.reset(seed=42)
        for _ in range(200):
            engine1.step()

        engine2 = SimulationEngine(network=net2, dt=1.0, demand_profiles=demand2)
        engine2.reset(seed=42)
        for _ in range(200):
            engine2.step()

        np.testing.assert_allclose(
            engine1.state.density, engine2.state.density, rtol=1e-10,
            err_msg="Round-trip density mismatch"
        )
        assert abs(engine1.state.total_exited - engine2.state.total_exited) < 1e-6

    def test_save_load_file(self):
        """save_scenario → load_scenario via file."""
        net, demand = create_single_intersection()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_scenario(path, net, demand)

            net2, demand2 = load_scenario(path)

            # Quick check: same structure
            assert len(net2.nodes) == len(net.nodes)
            assert len(net2.links) == len(net.links)
            assert len(net2.movements) == len(net.movements)
            assert len(demand2) == len(demand)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_export_structure(self):
        """Exported JSON should have expected structure."""
        net, demand = create_single_intersection()
        data = network_to_dict(net, demand)

        assert data["version"] == 1
        assert len(data["nodes"]) == len(net.nodes)
        assert len(data["links"]) == len(net.links)
        assert len(data["movements"]) == len(net.movements)
        assert "demand" in data
        assert len(data["demand"]) == len(demand)

    def test_no_demand(self):
        """Export without demand should work."""
        net, _ = create_single_intersection()
        data = network_to_dict(net)
        assert "demand" not in data

        net2, demand2 = dict_to_network(data)
        assert len(demand2) == 0
