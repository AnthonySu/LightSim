"""State recording for replay visualization.

Records simulation snapshots (density, signal states, metrics) at each step
so they can be replayed in the frontend without re-running the simulation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core.engine import SimulationEngine
from ..core.types import NodeID, NodeType


@dataclass
class Frame:
    """One recorded time step."""
    time: float
    step: int
    cell_density: list[float]
    signal_states: dict[str, dict[str, Any]]
    metrics: dict[str, float]


class Recorder:
    """Records simulation state at each step for later replay.

    Usage::

        recorder = Recorder(engine)
        engine.reset()
        for _ in range(1000):
            engine.step()
            recorder.capture()
        recorder.save("replay.json")
    """

    def __init__(self, engine: SimulationEngine) -> None:
        self.engine = engine
        self.frames: list[Frame] = []
        self._topology: dict[str, Any] | None = None

    def capture(self) -> Frame:
        """Capture the current engine state as a frame."""
        engine = self.engine
        state = engine.state
        net = engine.net

        # Signal states
        sig_states = {}
        for node_id, sig in engine.signal_manager.states.items():
            sig_states[str(int(node_id))] = {
                "phase_idx": sig.current_phase_idx,
                "time_in_phase": round(sig.time_in_phase, 2),
                "in_yellow": sig.in_yellow,
                "in_all_red": sig.in_all_red,
            }

        metrics = engine.get_network_metrics()
        # Round for compactness
        metrics = {k: round(v, 4) for k, v in metrics.items()}

        frame = Frame(
            time=round(state.time, 2),
            step=state.step_count,
            cell_density=[round(float(d), 6) for d in state.density],
            signal_states=sig_states,
            metrics=metrics,
        )
        self.frames.append(frame)
        return frame

    def get_topology(self) -> dict[str, Any]:
        """Extract static network topology for the frontend."""
        if self._topology is not None:
            return self._topology

        engine = self.engine
        network = engine.network
        net = engine.net

        nodes = []
        for node in network.nodes.values():
            n_phases = net.n_phases_per_node.get(node.node_id, 0)
            nodes.append({
                "id": int(node.node_id),
                "type": node.node_type.name.lower(),
                "x": node.x,
                "y": node.y,
                "n_phases": n_phases,
            })

        links = []
        for link in network.links.values():
            from_node = network.nodes[link.from_node]
            to_node = network.nodes[link.to_node]
            cell_ids = [int(c.cell_id) for c in link.cells]
            links.append({
                "id": int(link.link_id),
                "from_node": int(link.from_node),
                "to_node": int(link.to_node),
                "from_x": from_node.x,
                "from_y": from_node.y,
                "to_x": to_node.x,
                "to_y": to_node.y,
                "cells": cell_ids,
                "lanes": int(link.cells[0].lanes) if link.cells else 1,
                "n_cells": len(cell_ids),
            })

        cell_props = []
        for i in range(net.n_cells):
            cell_props.append({
                "length": float(net.length[i]),
                "kj": float(net.kj[i]),
                "Q": float(net.Q[i]),
                "vf": float(net.vf[i]),
            })

        self._topology = {
            "nodes": nodes,
            "links": links,
            "cells": cell_props,
            "n_cells": net.n_cells,
        }
        return self._topology

    def to_dict(self) -> dict[str, Any]:
        """Export topology + all frames as a dict."""
        return {
            "topology": self.get_topology(),
            "frames": [
                {
                    "time": f.time,
                    "step": f.step,
                    "density": f.cell_density,
                    "signals": f.signal_states,
                    "metrics": f.metrics,
                }
                for f in self.frames
            ],
        }

    def save(self, path: str | Path) -> None:
        """Save recording to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        """Load a recording from a JSON file."""
        with open(path) as f:
            return json.load(f)

    def get_frame_dict(self, frame: Frame) -> dict[str, Any]:
        """Convert a single frame to a dict for WebSocket transmission."""
        return {
            "type": "frame",
            "time": frame.time,
            "step": frame.step,
            "density": frame.cell_density,
            "signals": frame.signal_states,
            "metrics": frame.metrics,
        }
