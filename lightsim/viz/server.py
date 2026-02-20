"""FastAPI + WebSocket visualization server.

Supports three modes:
  - **Live mode**: runs a simulation in real-time, streaming frames over WS.
  - **Replay mode**: loads a recorded JSON file and plays it back over WS.
  - **Checkpoint mode**: loads a trained SB3 model, pre-runs an episode
    recording every engine step, then plays it back as a replay.

Start with::

    python -m lightsim.viz.server                           # live, single-intersection
    python -m lightsim.viz.server --scenario grid-4x4-v0    # live, grid
    python -m lightsim.viz.server --replay recording.json   # replay
    python -m lightsim.viz --controller MaxPressure          # live, MaxPressure
    python -m lightsim.viz --checkpoint model.zip            # RL checkpoint replay
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
from ..core.signal import (
    EfficientMaxPressureController,
    FixedTimeController,
    GreenWaveController,
    LostTimeAwareMaxPressureController,
    MaxPressureController,
    SignalController,
    SOTLController,
    WebsterController,
)
from .recorder import Recorder

STATIC_DIR = Path(__file__).parent / "static"

# Mapping of CLI names → controller classes
_CONTROLLERS: dict[str, type[SignalController]] = {
    "FixedTime": FixedTimeController,
    "Webster": WebsterController,
    "SOTL": SOTLController,
    "MaxPressure": MaxPressureController,
    "LTAwareMP": LostTimeAwareMaxPressureController,
    "EfficientMP": EfficientMaxPressureController,
    "GreenWave": GreenWaveController,
}


def _make_controller(name: str) -> SignalController:
    """Instantiate a controller by CLI name."""
    if name not in _CONTROLLERS:
        valid = ", ".join(_CONTROLLERS)
        raise ValueError(f"Unknown controller '{name}'. Choose from: {valid}")
    return _CONTROLLERS[name]()


def _prerun_checkpoint(
    scenario: str,
    checkpoint_path: str,
    algo: str | None = None,
    scenario_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load an SB3 checkpoint and pre-run a full episode, recording every engine step.

    Returns a replay dict (topology + frames) compatible with Recorder.load().
    """
    try:
        import stable_baselines3 as sb3
    except ImportError:
        raise ImportError(
            "stable_baselines3 is required for checkpoint visualization. "
            "Install with: pip install stable-baselines3"
        )

    from .. import make as lightsim_make

    # Auto-detect algorithm from checkpoint metadata if not specified
    if algo is None:
        import zipfile
        try:
            with zipfile.ZipFile(checkpoint_path, "r") as zf:
                if "data" in zf.namelist():
                    import io, torch
                    data = torch.load(
                        io.BytesIO(zf.read("data")),
                        map_location="cpu",
                        weights_only=False,
                    )
                    # SB3 stores the class name in various ways
                    algo = None
                    # Try common metadata keys
                    for key in ("algo", "algorithm"):
                        if key in data:
                            algo = str(data[key])
                            break
        except Exception:
            pass
        if algo is None:
            algo = "PPO"  # Default fallback

    # Map algo name to SB3 class
    algo_map = {
        "PPO": sb3.PPO,
        "A2C": sb3.A2C,
        "DQN": sb3.DQN,
        "SAC": sb3.SAC,
        "TD3": sb3.TD3,
    }
    algo_upper = algo.upper()
    if algo_upper not in algo_map:
        valid = ", ".join(algo_map)
        raise ValueError(f"Unknown algorithm '{algo}'. Choose from: {valid}")
    algo_cls = algo_map[algo_upper]

    # Create environment and load model
    env = lightsim_make(scenario, **(scenario_kwargs or {}))
    model = algo_cls.load(checkpoint_path, env=env)

    # Pre-run episode, recording every engine step
    obs, info = env.reset(seed=42)
    recorder = Recorder(env.engine)

    # Capture initial state
    recorder.capture()

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)

        # Replicate env.step() but capture every engine step
        env._action_handler.apply(action, env.engine, env.agent_node)
        for _ in range(env.sim_steps_per_action):
            env.engine.step()
            recorder.capture()
        env._step_count += 1
        obs = env._obs_builder.observe(env.engine, env.agent_node)

        done = env._step_count >= env.max_steps

    return recorder.to_dict()


def create_app(
    scenario: str = "single-intersection-v0",
    dt: float = 1.0,
    speed: float = 10.0,
    replay_path: str | None = None,
    controller: str | None = None,
    checkpoint: str | None = None,
    algo: str | None = None,
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
    controller : str, optional
        Controller name for live mode (default: FixedTime).
    checkpoint : str, optional
        Path to an SB3 ``.zip`` model file for checkpoint replay mode.
    algo : str, optional
        SB3 algorithm name (PPO, A2C, DQN, SAC, TD3). Auto-detected if omitted.
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
    app.state.controller_name = controller or "FixedTime"
    app.state.dt = dt

    if replay_path:
        app.state.replay_data = Recorder.load(replay_path)
    elif checkpoint:
        print(f"Pre-running checkpoint: {checkpoint} (algo={algo or 'auto'})")
        app.state.replay_data = _prerun_checkpoint(
            scenario=scenario,
            checkpoint_path=checkpoint,
            algo=algo,
            scenario_kwargs=scenario_kwargs,
        )
        n_frames = len(app.state.replay_data["frames"])
        print(f"Recorded {n_frames} frames. Starting replay server...")
    else:
        ctrl = _make_controller(controller or "FixedTime")
        factory = get_scenario(scenario)
        network, demand = factory(**(scenario_kwargs or {}))
        engine = SimulationEngine(
            network=network,
            dt=dt,
            controller=ctrl,
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

    @app.get("/api/controllers")
    async def api_controllers():
        return {"controllers": list(_CONTROLLERS.keys())}

    @app.get("/api/topology")
    async def api_topology():
        if app.state.replay_data:
            return app.state.replay_data["topology"]
        return app.state.recorder.get_topology()

    @app.get("/api/info")
    async def api_info():
        mode = "replay" if app.state.replay_data else "live"
        info = {
            "mode": mode,
            "scenario": app.state.scenario_name,
            "controller": app.state.controller_name,
        }
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

    def _rebuild(scenario_name: str, controller_name: str) -> None:
        """Rebuild engine with a new scenario and/or controller."""
        nonlocal engine, recorder
        ctrl = _make_controller(controller_name)
        factory = get_scenario(scenario_name)
        network, demand = factory()
        engine = SimulationEngine(
            network=network,
            dt=app.state.dt,
            controller=ctrl,
            demand_profiles=demand,
        )
        engine.reset(seed=42)
        recorder = Recorder(engine)
        app.state.engine = engine
        app.state.recorder = recorder
        app.state.scenario_name = scenario_name
        app.state.controller_name = controller_name

    # Start a task to listen for client commands
    async def listen():
        nonlocal paused, running, speed, interval, engine, recorder
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
                elif cmd == "switch_scenario":
                    scenario_name = data.get("scenario", app.state.scenario_name)
                    controller_name = data.get("controller", app.state.controller_name)
                    try:
                        _rebuild(scenario_name, controller_name)
                        topo = recorder.get_topology()
                        await ws.send_json({"type": "topology", "data": topo})
                    except (KeyError, ValueError) as e:
                        await ws.send_json({"type": "error", "message": str(e)})
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
    parser.add_argument("--controller", default=None,
                        help="Controller for live mode: "
                        + ", ".join(_CONTROLLERS))
    parser.add_argument("--checkpoint", default=None,
                        help="Path to SB3 .zip model for checkpoint replay")
    parser.add_argument("--algo", default=None,
                        help="SB3 algorithm (PPO, A2C, DQN, SAC, TD3)")
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
        controller=args.controller,
        checkpoint=args.checkpoint,
        algo=args.algo,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
