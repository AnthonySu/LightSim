"""Signal controllers for LightSim.

SignalManager holds per-node signal state and delegates decisions to
a SignalController (fixed-time or RL-driven).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .network import CompiledNetwork, Phase
from .types import FLOAT, NodeID, PhaseID


@dataclass
class SignalState:
    """Per-node signal state."""
    current_phase_idx: int = 0    # index into node's phase list
    time_in_phase: float = 0.0    # seconds spent in current phase
    in_yellow: bool = False
    in_all_red: bool = False
    yellow_timer: float = 0.0
    all_red_timer: float = 0.0


class SignalController(ABC):
    """Abstract signal controller."""

    @abstractmethod
    def get_phase_index(
        self,
        node_id: NodeID,
        state: SignalState,
        net: CompiledNetwork,
        density: np.ndarray,
    ) -> int:
        """Return the desired phase index for this node."""


class FixedTimeController(SignalController):
    """Cycle through phases with fixed green times."""

    def __init__(self, green_times: dict[NodeID, list[float]] | None = None):
        self.green_times = green_times or {}

    def get_phase_index(
        self,
        node_id: NodeID,
        state: SignalState,
        net: CompiledNetwork,
        density: np.ndarray,
    ) -> int:
        n_phases = net.n_phases_per_node.get(node_id, 1)
        greens = self.green_times.get(node_id)
        if greens:
            green_time = greens[state.current_phase_idx % len(greens)]
        else:
            green_time = 30.0  # default
        if state.time_in_phase >= green_time:
            return (state.current_phase_idx + 1) % n_phases
        return state.current_phase_idx


class RLController(SignalController):
    """Controller driven by external RL actions."""

    def __init__(self) -> None:
        self._actions: dict[NodeID, int] = {}

    def set_action(self, node_id: NodeID, phase_idx: int) -> None:
        self._actions[node_id] = phase_idx

    def get_phase_index(
        self,
        node_id: NodeID,
        state: SignalState,
        net: CompiledNetwork,
        density: np.ndarray,
    ) -> int:
        return self._actions.get(node_id, state.current_phase_idx)


class SignalManager:
    """Manages signal states for all signalised nodes."""

    def __init__(
        self,
        net: CompiledNetwork,
        controller: SignalController,
    ) -> None:
        self.net = net
        self.controller = controller
        self.states: dict[NodeID, SignalState] = {}
        for node_id in net.node_phases:
            self.states[node_id] = SignalState()

    def get_movement_mask(self) -> np.ndarray:
        """Return boolean mask: True if movement is green."""
        mask = np.ones(self.net.n_movements, dtype=FLOAT)
        for node_id, state in self.states.items():
            phase_ids = self.net.node_phases[node_id]
            if not phase_ids:
                continue
            current_pid = phase_ids[state.current_phase_idx]
            # Zero out all movements at this node that are NOT in the current phase
            node_movs = self.net.node_movements.get(node_id, [])
            for mid in node_movs:
                if not self.net.phase_mov_mask[current_pid, mid]:
                    mask[mid] = 0.0
            # If in yellow or all-red, all movements are red
            if state.in_yellow or state.in_all_red:
                for mid in node_movs:
                    mask[mid] = 0.0
        return mask

    def step(self, dt: float, density: np.ndarray) -> None:
        """Advance signal states by dt seconds."""
        for node_id, state in self.states.items():
            phase_ids = self.net.node_phases[node_id]
            if not phase_ids:
                continue

            if state.in_all_red:
                state.all_red_timer += dt
                current_pid = phase_ids[state.current_phase_idx]
                # Look up all_red duration from the phase that just ended
                # (use previous phase)
                prev_idx = (state.current_phase_idx - 1) % len(phase_ids)
                prev_pid = phase_ids[prev_idx]
                # Find the Phase object — we store all_red on each phase
                all_red_dur = self._get_phase_all_red(prev_pid)
                if state.all_red_timer >= all_red_dur:
                    state.in_all_red = False
                    state.all_red_timer = 0.0
                    state.time_in_phase = 0.0
                continue

            if state.in_yellow:
                state.yellow_timer += dt
                prev_idx = state.current_phase_idx
                prev_pid = phase_ids[prev_idx]
                yellow_dur = self._get_phase_yellow(prev_pid)
                if state.yellow_timer >= yellow_dur:
                    state.in_yellow = False
                    state.yellow_timer = 0.0
                    # Advance to next phase
                    desired = self.controller.get_phase_index(
                        node_id, state, self.net, density
                    )
                    state.current_phase_idx = desired % len(phase_ids)
                    state.in_all_red = True
                    state.all_red_timer = 0.0
                continue

            state.time_in_phase += dt
            desired = self.controller.get_phase_index(
                node_id, state, self.net, density
            )
            if desired != state.current_phase_idx:
                # Trigger yellow
                state.in_yellow = True
                state.yellow_timer = 0.0

    def _get_phase_yellow(self, phase_id: PhaseID) -> float:
        """Look up yellow duration for a phase."""
        for node in self.net.node_phases.values():
            for pid in node:
                if pid == phase_id:
                    # Access from the global_phase_list or fall back
                    return 3.0  # default
        return 3.0

    def _get_phase_all_red(self, phase_id: PhaseID) -> float:
        return 2.0  # default
