"""Visualization resilience tests: recorder frame cap, server error handling."""

import pytest

from lightsim.benchmarks.single_intersection import create_single_intersection
from lightsim.core.demand import DemandProfile
from lightsim.core.engine import SimulationEngine
from lightsim.core.signal import FixedTimeController
from lightsim.core.types import LinkID
from lightsim.viz.recorder import Recorder


class TestRecorderMaxFramesCap:
    """Recorder should track the correct number of captured frames."""

    def test_capture_count_matches_steps(self):
        """Number of captured frames should equal the number of capture() calls."""
        network, demand = create_single_intersection()
        engine = SimulationEngine(
            network=network, dt=1.0,
            controller=FixedTimeController(),
            demand_profiles=demand,
        )
        engine.reset(seed=42)
        recorder = Recorder(engine)

        n_steps = 50
        for _ in range(n_steps):
            engine.step()
            recorder.capture()

        assert len(recorder.frames) == n_steps

    def test_manual_frame_cap(self):
        """User can enforce a max_frames cap by slicing or limiting captures."""
        network, demand = create_single_intersection()
        engine = SimulationEngine(
            network=network, dt=1.0,
            controller=FixedTimeController(),
            demand_profiles=demand,
        )
        engine.reset(seed=42)
        recorder = Recorder(engine)

        max_frames = 20
        for i in range(100):
            engine.step()
            if len(recorder.frames) < max_frames:
                recorder.capture()

        assert len(recorder.frames) == max_frames

    def test_frame_data_integrity(self):
        """Each captured frame should have valid time, step, and density data."""
        network, demand = create_single_intersection()
        engine = SimulationEngine(
            network=network, dt=1.0,
            controller=FixedTimeController(),
            demand_profiles=demand,
        )
        engine.reset(seed=42)
        recorder = Recorder(engine)

        for _ in range(10):
            engine.step()
            frame = recorder.capture()
            assert frame.time > 0
            assert frame.step > 0
            assert len(frame.cell_density) == engine.net.n_cells
            assert all(isinstance(d, float) for d in frame.cell_density)

    def test_to_dict_structure(self):
        """Recorder.to_dict() should produce valid topology + frames structure."""
        network, demand = create_single_intersection()
        engine = SimulationEngine(
            network=network, dt=1.0,
            controller=FixedTimeController(),
            demand_profiles=demand,
        )
        engine.reset(seed=42)
        recorder = Recorder(engine)

        for _ in range(5):
            engine.step()
            recorder.capture()

        data = recorder.to_dict()
        assert "topology" in data
        assert "frames" in data
        assert len(data["frames"]) == 5
        assert "nodes" in data["topology"]
        assert "links" in data["topology"]
        assert "cells" in data["topology"]


class TestServerInvalidScenario:
    """Server should handle invalid scenario names gracefully."""

    def test_get_scenario_unknown_raises_keyerror(self):
        """get_scenario with an invalid name should raise KeyError."""
        from lightsim.benchmarks.scenarios import get_scenario

        with pytest.raises(KeyError, match="Unknown scenario"):
            get_scenario("nonexistent-scenario-xyz")

    def test_make_controller_unknown_raises_valueerror(self):
        """_make_controller with an invalid name should raise ValueError."""
        from lightsim.viz.server import _make_controller

        with pytest.raises(ValueError, match="Unknown controller"):
            _make_controller("NonexistentController")

    def test_valid_scenarios_listed(self):
        """list_scenarios should return a non-empty list of known scenario names."""
        from lightsim.benchmarks.scenarios import list_scenarios

        scenarios = list_scenarios()
        assert isinstance(scenarios, list)
        assert len(scenarios) > 0
        assert "single-intersection-v0" in scenarios

    def test_valid_controllers_listed(self):
        """The server's controller registry should contain known controllers."""
        from lightsim.viz.server import _CONTROLLERS

        assert "FixedTime" in _CONTROLLERS
        assert "MaxPressure" in _CONTROLLERS
        assert "Webster" in _CONTROLLERS
        assert len(_CONTROLLERS) >= 5

