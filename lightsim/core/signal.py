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


class MaxPressureController(SignalController):
    """MaxPressure adaptive signal controller.

    At each decision point, selects the phase that maximises the pressure
    (sum of upstream density - downstream density for each movement in
    the phase).  This is a well-known decentralised adaptive controller
    with provable stability guarantees.

    Reference: Varaiya (2013), "Max pressure control of a network of
    signalized intersections", Transportation Research Part C.
    """

    def __init__(self, min_green: float = 5.0) -> None:
        self.min_green = min_green

    def get_phase_index(
        self,
        node_id: NodeID,
        state: SignalState,
        net: CompiledNetwork,
        density: np.ndarray,
    ) -> int:
        # Don't switch if minimum green hasn't elapsed
        if state.time_in_phase < self.min_green:
            return state.current_phase_idx

        phase_ids = net.node_phases.get(node_id, [])
        if not phase_ids:
            return 0

        best_idx = 0
        best_pressure = -np.inf

        for idx, pid in enumerate(phase_ids):
            pressure = 0.0
            for mid in range(net.n_movements):
                if net.phase_mov_mask[pid, mid]:
                    upstream_k = density[net.mov_from_cell[mid]]
                    downstream_k = density[net.mov_to_cell[mid]]
                    pressure += (upstream_k - downstream_k) * net.mov_turn_ratio[mid]
            if pressure > best_pressure:
                best_pressure = pressure
                best_idx = idx

        return best_idx


class LostTimeAwareMaxPressureController(SignalController):
    """MaxPressure that accounts for switching cost from lost time.

    Only switches phase if the pressure gain from switching exceeds the
    throughput cost of the transition (yellow + all_red + lost_time ramp).
    This prevents excessive switching when mesoscopic lost_time is enabled.

    The switching cost is estimated as:
        cost = sat_rate × (yellow + all_red + lost_time / 2)
    where the lost_time/2 accounts for the average capacity reduction during
    the ramp-up period.  A switch is only triggered if:
        pressure(best) - pressure(current) > switch_threshold_factor × cost

    Reference: Varaiya (2013) + switching-cost extension.
    """

    def __init__(
        self,
        min_green: float = 5.0,
        switch_threshold_factor: float = 0.01,
    ) -> None:
        self.min_green = min_green
        self.switch_threshold_factor = switch_threshold_factor

    def get_phase_index(
        self,
        node_id: NodeID,
        state: SignalState,
        net: CompiledNetwork,
        density: np.ndarray,
    ) -> int:
        if state.time_in_phase < self.min_green:
            return state.current_phase_idx

        phase_ids = net.node_phases.get(node_id, [])
        if not phase_ids:
            return 0

        # Compute pressure for each phase
        pressures = []
        for idx, pid in enumerate(phase_ids):
            pressure = 0.0
            for mid in range(net.n_movements):
                if net.phase_mov_mask[pid, mid]:
                    upstream_k = density[net.mov_from_cell[mid]]
                    downstream_k = density[net.mov_to_cell[mid]]
                    pressure += (upstream_k - downstream_k) * net.mov_turn_ratio[mid]
            pressures.append(pressure)

        best_idx = int(np.argmax(pressures))
        current_pressure = pressures[state.current_phase_idx]
        best_pressure = pressures[best_idx]

        if best_idx == state.current_phase_idx:
            return state.current_phase_idx

        # Compute switching cost from lost time
        current_pid = phase_ids[state.current_phase_idx]
        yellow = float(net.phase_yellow[current_pid]) if len(net.phase_yellow) > current_pid else 3.0
        all_red = float(net.phase_all_red[current_pid]) if len(net.phase_all_red) > current_pid else 2.0
        lost = float(net.phase_lost_time[current_pid]) if len(net.phase_lost_time) > current_pid else 0.0
        dead_time = yellow + all_red + lost / 2.0
        threshold = self.switch_threshold_factor * dead_time

        if best_pressure - current_pressure > threshold:
            return best_idx
        return state.current_phase_idx


class EfficientMaxPressureController(SignalController):
    """Efficient MaxPressure (EMP) — adjusts green duration based on pressure.

    Instead of binary switch/no-switch, EMP extends the current phase green
    proportionally to its pressure advantage over the next-best phase.
    This naturally produces longer greens when one direction is dominant
    and shorter greens when pressures are balanced.

    Implements a simplified version of "Efficient MaxPressure" from:
    Wu et al. (2021), "Efficient pressure: improving efficiency for
    signalized intersections".
    """

    def __init__(
        self,
        min_green: float = 5.0,
        max_green: float = 45.0,
        base_green: float = 10.0,
        pressure_scale: float = 100.0,
    ) -> None:
        self.min_green = min_green
        self.max_green = max_green
        self.base_green = base_green
        self.pressure_scale = pressure_scale

    def get_phase_index(
        self,
        node_id: NodeID,
        state: SignalState,
        net: CompiledNetwork,
        density: np.ndarray,
    ) -> int:
        phase_ids = net.node_phases.get(node_id, [])
        if not phase_ids:
            return 0

        # Compute pressure for current phase
        current_pid = phase_ids[state.current_phase_idx]
        current_pressure = 0.0
        for mid in range(net.n_movements):
            if net.phase_mov_mask[current_pid, mid]:
                upstream_k = density[net.mov_from_cell[mid]]
                downstream_k = density[net.mov_to_cell[mid]]
                current_pressure += (upstream_k - downstream_k) * net.mov_turn_ratio[mid]

        # Adaptive green time: higher pressure → longer green
        adaptive_green = self.base_green + current_pressure * self.pressure_scale
        adaptive_green = np.clip(adaptive_green, self.min_green, self.max_green)

        if state.time_in_phase < adaptive_green:
            return state.current_phase_idx

        # Time to switch — pick highest pressure phase
        best_idx = 0
        best_pressure = -np.inf
        for idx, pid in enumerate(phase_ids):
            pressure = 0.0
            for mid in range(net.n_movements):
                if net.phase_mov_mask[pid, mid]:
                    upstream_k = density[net.mov_from_cell[mid]]
                    downstream_k = density[net.mov_to_cell[mid]]
                    pressure += (upstream_k - downstream_k) * net.mov_turn_ratio[mid]
            if pressure > best_pressure:
                best_pressure = pressure
                best_idx = idx

        return best_idx


class SOTLController(SignalController):
    """Self-Organizing Traffic Light (SOTL) controller.

    A simple adaptive controller that extends green while vehicles are
    approaching the stop bar, and switches when the approaching count
    drops below a threshold or max green is reached.

    Reference: Cools et al. (2013), "Self-organizing traffic lights:
    A realistic simulation".
    """

    def __init__(
        self,
        min_green: float = 5.0,
        max_green: float = 50.0,
        theta: float = 0.03,  # density threshold to extend green
    ) -> None:
        self.min_green = min_green
        self.max_green = max_green
        self.theta = theta

    def get_phase_index(
        self,
        node_id: NodeID,
        state: SignalState,
        net: CompiledNetwork,
        density: np.ndarray,
    ) -> int:
        if state.time_in_phase < self.min_green:
            return state.current_phase_idx

        if state.time_in_phase >= self.max_green:
            n_phases = net.n_phases_per_node.get(node_id, 1)
            return (state.current_phase_idx + 1) % n_phases

        phase_ids = net.node_phases.get(node_id, [])
        if not phase_ids:
            return 0

        # Check if current green phase still has demand approaching
        current_pid = phase_ids[state.current_phase_idx]
        approaching_density = 0.0
        for mid in range(net.n_movements):
            if net.phase_mov_mask[current_pid, mid]:
                approaching_density += density[net.mov_from_cell[mid]]

        # Extend green if vehicles are approaching
        if approaching_density > self.theta:
            return state.current_phase_idx

        # Otherwise, switch to next phase
        n_phases = len(phase_ids)
        return (state.current_phase_idx + 1) % n_phases


class WebsterController(SignalController):
    """Webster's optimal fixed-time controller.

    Computes the optimal cycle length and green splits based on
    observed demand ratios using Webster's 1958 formula.  Re-optimizes
    periodically as demand changes.

    Reference: Webster (1958), "Traffic signal settings", Road Research
    Technical Paper No. 39.
    """

    def __init__(
        self,
        cycle_length: float | None = None,
        lost_time_per_phase: float = 5.0,
        reoptimize_interval: float = 300.0,
    ) -> None:
        self.fixed_cycle = cycle_length
        self.lost_time_per_phase = lost_time_per_phase
        self.reoptimize_interval = reoptimize_interval
        self._green_times: dict[NodeID, list[float]] = {}
        self._last_optimize: dict[NodeID, float] = {}

    def _optimize(self, node_id: NodeID, net: CompiledNetwork,
                  density: np.ndarray) -> None:
        """Compute Webster optimal green splits."""
        phase_ids = net.node_phases.get(node_id, [])
        if not phase_ids:
            return

        n_phases = len(phase_ids)
        total_lost = self.lost_time_per_phase * n_phases

        # Compute critical flow ratio per phase
        y_values = []
        for pid in phase_ids:
            max_ratio = 0.0
            for mid in range(net.n_movements):
                if net.phase_mov_mask[pid, mid]:
                    # flow ratio ≈ density × vf / saturation_rate
                    k = density[net.mov_from_cell[mid]]
                    sat = net.mov_sat_rate[mid]
                    vf = net.vf[net.mov_from_cell[mid]]
                    flow = k * vf  # approximate flow
                    ratio = flow / max(sat, 1e-9)
                    max_ratio = max(max_ratio, ratio)
            y_values.append(max(max_ratio, 0.01))

        Y = sum(y_values)
        Y = min(Y, 0.9)  # cap to avoid infinite cycle

        if self.fixed_cycle:
            C = self.fixed_cycle
        else:
            # Webster formula: C_opt = (1.5L + 5) / (1 - Y)
            C = (1.5 * total_lost + 5) / (1 - Y)
            C = np.clip(C, 30.0, 180.0)

        effective_green = C - total_lost
        greens = []
        for y in y_values:
            g = effective_green * (y / Y) if Y > 0 else effective_green / n_phases
            g = max(g, 5.0)
            greens.append(float(g))

        self._green_times[node_id] = greens

    def get_phase_index(
        self,
        node_id: NodeID,
        state: SignalState,
        net: CompiledNetwork,
        density: np.ndarray,
    ) -> int:
        n_phases = net.n_phases_per_node.get(node_id, 1)

        # Re-optimize periodically
        elapsed = state.time_in_phase
        total_time = self._last_optimize.get(node_id, -self.reoptimize_interval)
        if node_id not in self._green_times or (
            elapsed == 0 and state.current_phase_idx == 0
        ):
            self._optimize(node_id, net, density)
            self._last_optimize[node_id] = 0.0

        greens = self._green_times.get(node_id, [30.0] * n_phases)
        green_time = greens[state.current_phase_idx % len(greens)]

        if state.time_in_phase >= green_time:
            return (state.current_phase_idx + 1) % n_phases
        return state.current_phase_idx


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

        # Per-movement green timer tracking for capacity factor
        self._time_since_green = np.zeros(net.n_movements, dtype=FLOAT)
        self._was_green = np.zeros(net.n_movements, dtype=bool)

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
                # Look up all_red duration from the phase that just ended
                prev_idx = (state.current_phase_idx - 1) % len(phase_ids)
                prev_pid = phase_ids[prev_idx]
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

        # Update per-movement green timers
        current_mask = self.get_movement_mask()
        is_green = current_mask > 0.0
        # Detect red→green transitions (newly green)
        newly_green = is_green & ~self._was_green
        still_green = is_green & self._was_green
        # Reset timer for newly-green, increment for still-green, zero for red
        self._time_since_green[newly_green] = dt
        self._time_since_green[still_green] += dt
        self._time_since_green[~is_green] = 0.0
        self._was_green[:] = is_green

    def get_capacity_factor(self) -> np.ndarray:
        """Return per-movement capacity factor (0.0–1.0) for green ramp.

        Models start-up lost time: factor ramps from 0 to 1 over ``lost_time``
        seconds after each red→green transition. If ``lost_time == 0``, factor
        is always 1.0 (backward compatible).
        """
        factor = np.ones(self.net.n_movements, dtype=FLOAT)
        for node_id, state in self.states.items():
            phase_ids = self.net.node_phases[node_id]
            if not phase_ids:
                continue
            current_pid = phase_ids[state.current_phase_idx]
            lost = self.net.phase_lost_time[current_pid]
            if lost <= 0.0:
                continue
            node_movs = self.net.node_movements.get(node_id, [])
            for mid in node_movs:
                if self._was_green[mid]:
                    factor[mid] = min(1.0, self._time_since_green[mid] / lost)
                # Red movements already have factor=1.0 but will be zeroed by signal_mask
        return factor

    def _get_phase_yellow(self, phase_id: PhaseID) -> float:
        """Look up yellow duration for a phase from compiled arrays."""
        if len(self.net.phase_yellow) > phase_id:
            return float(self.net.phase_yellow[phase_id])
        return 3.0

    def _get_phase_all_red(self, phase_id: PhaseID) -> float:
        """Look up all-red duration for a phase from compiled arrays."""
        if len(self.net.phase_all_red) > phase_id:
            return float(self.net.phase_all_red[phase_id])
        return 2.0
