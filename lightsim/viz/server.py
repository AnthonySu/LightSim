"""FastAPI + WebSocket visualization server.

Supports two modes:
  - **Live mode**: runs a simulation in real-time, streaming frames over WS.
  - **Replay mode**: loads a recorded JSON file and plays it back over WS.

Start with::

    python -m lightsim.viz.server                           # live, single-intersection
    python -m lightsim.viz.server --scenario grid-4x4-v0    # live, grid
    python -m lightsim.viz.server --replay recording.json   # replay
"""

from __future__ import annotations

import asyncio
import json
import argparse
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    HAS_SERVER_DEPS = True
except ImportError:
    HAS_SERVER_DEPS = False

from ..benchmarks.scenarios import get_scenario, list_scenarios
from ..core.demand import DemandProfile
from ..core.engine import SimulationEngine
from ..core.network import Network
from ..core.signal import FixedTimeController
from .recorder import Recorder

STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    scenario: str = "single-intersection-v0",
    dt: float = 1.0,
    speed: float = 10.0,
    replay_path: str | None = None,
    scenario_kwargs: dict[str, Any] | None = None,
) -> "FastAPI":
    """Create the FastAPI application.

    Parameters
    ----------
    scenario : str
        Scenario name for live mode.
    dt : float
        Simulation time step.
    speed : float
        Simulation steps per second for live streaming.
    replay_path : str, optional
        Path to a recorded JSON file for replay mode.
    scenario_kwargs : dict, optional
        Extra kwargs for the scenario factory.
    """
    if not HAS_SERVER_DEPS:
        raise ImportError(
            "FastAPI and uvicorn are required for visualization. "
            "Install with: pip install lightsim[viz]"
        )

    app = FastAPI(title="LightSim Visualization")

    # State shared across endpoints
    app.state.replay_data = None
    app.state.engine = None
    app.state.recorder = None
    app.state.speed = speed
    app.state.scenario_name = scenario

    if replay_path:
        app.state.replay_data = Recorder.load(replay_path)
    else:
        factory = get_scenario(scenario)
        network, demand = factory(**(scenario_kwargs or {}))
        engine = SimulationEngine(
            network=network,
            dt=dt,
            controller=FixedTimeController(),
            demand_profiles=demand,
        )
        engine.reset(seed=42)
        recorder = Recorder(engine)
        app.state.engine = engine
        app.state.recorder = recorder

    # --- HTTP routes ---

    @app.get("/", response_class=HTMLResponse)
    async def index():
        index_path = STATIC_DIR / "index.html"
        return index_path.read_text(encoding="utf-8")

    @app.get("/api/scenarios")
    async def api_scenarios():
        return {"scenarios": list_scenarios()}

    @app.get("/api/topology")
    async def api_topology():
        if app.state.replay_data:
            return app.state.replay_data["topology"]
        return app.state.recorder.get_topology()

    @app.get("/api/info")
    async def api_info():
        mode = "replay" if app.state.replay_data else "live"
        info = {"mode": mode, "scenario": app.state.scenario_name}
        if app.state.replay_data:
            info["total_frames"] = len(app.state.replay_data["frames"])
        return info

    # --- WebSocket ---

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        try:
            # Send topology first
            if app.state.replay_data:
                topo = app.state.replay_data["topology"]
            else:
                topo = app.state.recorder.get_topology()

            await ws.send_json({"type": "topology", "data": topo})

            if app.state.replay_data:
                await _replay_loop(ws, app)
            else:
                await _live_loop(ws, app)

        except WebSocketDisconnect:
            pass

    return app


async def _live_loop(ws: "WebSocket", app: "FastAPI") -> None:
    """Run simulation in real-time, streaming frames to the client."""
    engine = app.state.engine
    recorder = app.state.recorder
    speed = app.state.speed
    interval = 1.0 / speed

    paused = False
    running = True

    # Start a task to listen for client commands
    async def listen():
        nonlocal paused, running, speed, interval
        try:
            while running:
                msg = await ws.receive_text()
                data = json.loads(msg)
                cmd = data.get("cmd")
                if cmd == "pause":
                    paused = True
                elif cmd == "resume":
                    paused = False
                elif cmd == "speed":
                    speed = float(data.get("value", 10.0))
                    interval = 1.0 / max(speed, 0.1)
                elif cmd == "reset":
                    engine.reset(seed=42)
                    recorder.frames.clear()
                    paused = False
                elif cmd == "step":
                    # Single step while paused
                    engine.step()
                    frame = recorder.capture()
                    await ws.send_json(recorder.get_frame_dict(frame))
                elif cmd == "close":
                    running = False
        except WebSocketDisconnect:
            running = False

    listener = asyncio.create_task(listen())

    try:
        while running:
            if not paused:
                engine.step()
                frame = recorder.capture()
                await ws.send_json(recorder.get_frame_dict(frame))
            await asyncio.sleep(interval)
    finally:
        listener.cancel()


async def _replay_loop(ws: "WebSocket", app: "FastAPI") -> None:
    """Play back recorded frames to the client."""
    frames = app.state.replay_data["frames"]
    speed = app.state.speed
    interval = 1.0 / speed
    idx = 0
    paused = False
    running = True

    async def listen():
        nonlocal paused, running, speed, interval, idx
        try:
            while running:
                msg = await ws.receive_text()
                data = json.loads(msg)
                cmd = data.get("cmd")
                if cmd == "pause":
                    paused = True
                elif cmd == "resume":
                    paused = False
                elif cmd == "speed":
                    speed = float(data.get("value", 10.0))
                    interval = 1.0 / max(speed, 0.1)
                elif cmd == "reset":
                    idx = 0
                    paused = False
                elif cmd == "seek":
                    idx = max(0, min(int(data.get("frame", 0)), len(frames) - 1))
                    frame = frames[idx]
                    await ws.send_json({"type": "frame", **frame})
                elif cmd == "step":
                    if idx < len(frames):
                        frame = frames[idx]
                        await ws.send_json({"type": "frame", **frame})
                        idx += 1
                elif cmd == "close":
                    running = False
        except WebSocketDisconnect:
            running = False

    listener = asyncio.create_task(listen())

    try:
        while running and idx < len(frames):
            if not paused:
                frame = frames[idx]
                await ws.send_json({"type": "frame", **frame})
                idx += 1
            await asyncio.sleep(interval)

        # Signal end of replay
        if running:
            await ws.send_json({"type": "end"})
            # Wait for client commands (seek/reset)
            while running:
                await asyncio.sleep(0.1)
    finally:
        listener.cancel()


def main():
    """CLI entry point for the visualization server."""
    if not HAS_SERVER_DEPS:
        print("Error: FastAPI and uvicorn are required.")
        print("Install with: pip install lightsim[viz]")
        return

    parser = argparse.ArgumentParser(description="LightSim Visualization Server")
    parser.add_argument("--scenario", default="single-intersection-v0",
                        help="Scenario name for live mode")
    parser.add_argument("--replay", default=None,
                        help="Path to recorded JSON file for replay mode")
    parser.add_argument("--dt", type=float, default=1.0,
                        help="Simulation time step (seconds)")
    parser.add_argument("--speed", type=float, default=10.0,
                        help="Simulation steps per second")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    app = create_app(
        scenario=args.scenario,
        dt=args.dt,
        speed=args.speed,
        replay_path=args.replay,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
