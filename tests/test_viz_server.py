"""Integration tests for the visualization server HTTP routes and WebSocket."""

import json
import tempfile
from pathlib import Path

import pytest

from lightsim.core.engine import SimulationEngine
from lightsim.core.signal import FixedTimeController
from lightsim.viz.recorder import Recorder

try:
    from httpx import AsyncClient, ASGITransport
    from lightsim.viz.server import create_app
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="fastapi/httpx not installed")


def _make_recording_path():
    from lightsim.benchmarks.single_intersection import create_single_intersection
    net, demand = create_single_intersection()
    engine = SimulationEngine(
        network=net, dt=1.0, demand_profiles=demand,
        controller=FixedTimeController(),
    )
    engine.reset(seed=42)
    rec = Recorder(engine)
    for _ in range(10):
        engine.step()
        rec.capture()
    path = tempfile.mktemp(suffix=".json")
    rec.save(path)
    return path


@pytest.fixture
def live_app():
    return create_app("single-intersection-v0", speed=100)


@pytest.fixture
def replay_app():
    path = _make_recording_path()
    app = create_app(replay_path=path)
    yield app
    Path(path).unlink(missing_ok=True)


@pytest.mark.anyio
async def test_index_returns_html(live_app):
    transport = ASGITransport(app=live_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "LightSim" in resp.text


@pytest.mark.anyio
async def test_api_scenarios(live_app):
    transport = ASGITransport(app=live_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/scenarios")
        assert resp.status_code == 200
        data = resp.json()
        assert "single-intersection-v0" in data["scenarios"]


@pytest.mark.anyio
async def test_api_topology_live(live_app):
    transport = ASGITransport(app=live_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/topology")
        assert resp.status_code == 200
        topo = resp.json()
        assert "nodes" in topo
        assert "links" in topo
        assert len(topo["nodes"]) > 0


@pytest.mark.anyio
async def test_api_topology_replay(replay_app):
    transport = ASGITransport(app=replay_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/topology")
        assert resp.status_code == 200
        topo = resp.json()
        assert "nodes" in topo


@pytest.mark.anyio
async def test_api_info_live(live_app):
    transport = ASGITransport(app=live_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/info")
        assert resp.status_code == 200
        info = resp.json()
        assert info["mode"] == "live"
        assert info["scenario"] == "single-intersection-v0"


@pytest.mark.anyio
async def test_api_info_replay(replay_app):
    transport = ASGITransport(app=replay_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/info")
        assert resp.status_code == 200
        info = resp.json()
        assert info["mode"] == "replay"
        assert info["total_frames"] == 10
