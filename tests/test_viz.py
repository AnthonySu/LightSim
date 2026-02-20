"""Tests for visualization: recorder and server."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import lightsim
from lightsim.core.demand import DemandProfile
from lightsim.core.engine import SimulationEngine
from lightsim.core.network import Network
from lightsim.core.signal import FixedTimeController
from lightsim.core.types import LinkID, NodeID, NodeType, TurnType
from lightsim.viz.recorder import Recorder


def _make_engine():
    """Create a simple engine for testing."""
    from lightsim.benchmarks.single_intersection import create_single_intersection
    net, demand = create_single_intersection()
    engine = SimulationEngine(
        network=net, dt=1.0, demand_profiles=demand,
        controller=FixedTimeController(),
    )
    engine.reset(seed=42)
    return engine


class TestRecorder:
    def test_capture_frame(self):
        engine = _make_engine()
        recorder = Recorder(engine)
        engine.step()
        frame = recorder.capture()

        assert frame.time == 1.0
        assert frame.step == 1
        assert len(frame.cell_density) == engine.net.n_cells
        assert isinstance(frame.signal_states, dict)
        assert isinstance(frame.metrics, dict)

    def test_topology(self):
        engine = _make_engine()
        recorder = Recorder(engine)
        topo = recorder.get_topology()

        assert "nodes" in topo
        assert "links" in topo
        assert "cells" in topo
        assert topo["n_cells"] == engine.net.n_cells
        assert len(topo["nodes"]) > 0
        assert len(topo["links"]) > 0

        # Check node structure
        node = topo["nodes"][0]
        assert "id" in node
        assert "type" in node
        assert "x" in node
        assert "y" in node

        # Check link structure
        link = topo["links"][0]
        assert "id" in link
        assert "from_node" in link
        assert "to_node" in link
        assert "cells" in link
        assert len(link["cells"]) > 0

    def test_multiple_captures(self):
        engine = _make_engine()
        recorder = Recorder(engine)
        for _ in range(10):
            engine.step()
            recorder.capture()

        assert len(recorder.frames) == 10
        assert recorder.frames[0].time < recorder.frames[-1].time

    def test_to_dict(self):
        engine = _make_engine()
        recorder = Recorder(engine)
        for _ in range(5):
            engine.step()
            recorder.capture()

        data = recorder.to_dict()
        assert "topology" in data
        assert "frames" in data
        assert len(data["frames"]) == 5

    def test_save_and_load(self):
        engine = _make_engine()
        recorder = Recorder(engine)
        for _ in range(5):
            engine.step()
            recorder.capture()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        recorder.save(path)
        loaded = Recorder.load(path)

        assert "topology" in loaded
        assert len(loaded["frames"]) == 5
        assert loaded["frames"][0]["time"] == recorder.frames[0].time
        Path(path).unlink()

    def test_get_frame_dict(self):
        engine = _make_engine()
        recorder = Recorder(engine)
        engine.step()
        frame = recorder.capture()
        d = recorder.get_frame_dict(frame)

        assert d["type"] == "frame"
        assert "time" in d
        assert "step" in d
        assert "density" in d
        assert "signals" in d
        assert "metrics" in d

    def test_frame_density_values_reasonable(self):
        engine = _make_engine()
        recorder = Recorder(engine)
        for _ in range(50):
            engine.step()
        frame = recorder.capture()

        # All densities should be non-negative
        for d in frame.cell_density:
            assert d >= 0.0

    def test_signal_states_present(self):
        engine = _make_engine()
        recorder = Recorder(engine)
        for _ in range(10):
            engine.step()
        frame = recorder.capture()

        # Should have signal state for the signalised node (node 0)
        assert "0" in frame.signal_states
        sig = frame.signal_states["0"]
        assert "phase_idx" in sig
        assert "in_yellow" in sig
        assert "in_all_red" in sig


class TestServerImport:
    """Test that the server module can be imported and app created."""

    def test_create_app(self):
        try:
            from lightsim.viz.server import create_app
            app = create_app("single-intersection-v0")
            assert app is not None
        except ImportError:
            pytest.skip("fastapi/uvicorn not installed")

    def test_create_app_replay(self):
        try:
            from lightsim.viz.server import create_app
        except ImportError:
            pytest.skip("fastapi/uvicorn not installed")

        # Create a recording first
        engine = _make_engine()
        recorder = Recorder(engine)
        for _ in range(5):
            engine.step()
            recorder.capture()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        recorder.save(path)

        try:
            app = create_app(replay_path=path)
        except ImportError:
            pytest.skip("fastapi/uvicorn not installed")
        assert app.state.replay_data is not None
        assert len(app.state.replay_data["frames"]) == 5
        Path(path).unlink()

    def test_static_index_exists(self):
        from lightsim.viz.server import STATIC_DIR
        index = STATIC_DIR / "index.html"
        assert index.exists()
        content = index.read_text()
        assert "LightSim" in content
