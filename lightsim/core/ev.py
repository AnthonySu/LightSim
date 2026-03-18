"""Emergency Vehicle (EV) tracking overlay for LightSim.

Provides a lightweight EV position tracker that sits on top of the CTM
simulation engine.  The EV travels along a pre-computed route of links,
with speed modulated by local congestion and blocked at red signals.

Usage::

    from lightsim.core import SimulationEngine, Network
    from lightsim.core.ev import EVTracker

    engine = SimulationEngine(network, dt=5.0)
    ev = EVTracker(engine, route=[link_0, link_1, ...])
    engine.reset()
    ev.reset()

    while not ev.arrived:
        engine.step()
        ev.step()
        print(ev.position, ev.travel_time)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .types import LinkID, NodeID

if TYPE_CHECKING:
    from .engine import SimulationEngine


@dataclass
class EVState:
    """Snapshot of the EV position and travel statistics."""

    link_idx: int = 0
    """Index into the route list (which link the EV is on)."""

    progress: float = 0.0
    """Fractional position [0, 1) on the current link."""

    speed: float = 0.0
    """EV speed (m/s) at the last step."""

    travel_time: float = 0.0
    """Cumulative travel time in seconds."""

    distance_traveled: float = 0.0
    """Cumulative distance in metres."""

    arrived: bool = False
    """True once the EV has reached the last link of its route."""

    stops: int = 0
    """Number of times the EV was blocked at a red signal."""

    _was_stopped: bool = field(default=False, repr=False)
    """Internal: whether the EV was stopped on the previous step."""


class EVTracker:
    """Tracks an emergency vehicle along a route through the network.

    Parameters
    ----------
    engine : SimulationEngine
        The simulation engine that owns the traffic state.
    route : list[LinkID]
        Ordered sequence of link IDs from origin to destination.
    speed_factor : float
        Multiplier on free-flow speed for the EV (default 1.5 for
        lights-and-sirens driving).
    min_speed_fraction : float
        Minimum EV speed as a fraction of free-flow, even in full
        congestion. Models the EV's ability to push through.
    """

    def __init__(
        self,
        engine: SimulationEngine,
        route: list[LinkID],
        speed_factor: float = 1.5,
        min_speed_fraction: float = 0.05,
    ) -> None:
        if not route:
            raise ValueError("EV route must contain at least one link")
        self.engine = engine
        self.route = list(route)
        self.speed_factor = speed_factor
        self.min_speed_fraction = min_speed_fraction
        self.state = EVState()

        # Pre-compute link lengths for fast lookup
        net = engine.net
        self._link_lengths: list[float] = []
        for lid in self.route:
            cells = net.link_cells[lid]
            total = float(net.length[cells].sum())
            self._link_lengths.append(total)

        # Pre-compute downstream node for each link (for signal checking)
        self._downstream_nodes: list[NodeID | None] = []
        for lid in self.route:
            ds_node = self._find_downstream_node(lid)
            self._downstream_nodes.append(ds_node)

    def _find_downstream_node(self, link_id: LinkID) -> NodeID | None:
        """Find the signalized node at the downstream end of a link."""
        net = self.engine.net
        # Check all movements originating from this link's last cell
        last_cell = net.link_last_cell[link_id]
        for mid in range(net.n_movements):
            if net.mov_from_cell[mid] == last_cell:
                node = net.mov_node[mid]
                if node in net.node_phases:
                    return node
        return None

    def reset(self) -> EVState:
        """Reset the EV to the start of its route."""
        self.state = EVState()
        return self.state

    def step(self) -> EVState:
        """Advance the EV by one simulation time step.

        Must be called after ``engine.step()``.

        Returns
        -------
        EVState
            Updated EV state.
        """
        s = self.state
        if s.arrived:
            return s

        engine = self.engine
        net = engine.net
        dt = engine.dt
        link_id = self.route[s.link_idx]
        link_length = self._link_lengths[s.link_idx]

        if link_length < 1e-6:
            # Degenerate link — skip to next
            s.link_idx += 1
            s.progress = 0.0
            if s.link_idx >= len(self.route):
                s.arrived = True
            return s

        # --- Compute EV speed ---
        # Get average density on the current link
        cells = net.link_cells[link_id]
        avg_density = float(engine.state.density[cells].mean())
        avg_kj = float(net.kj[cells].mean())
        avg_vf = float(net.vf[cells].mean())

        # Speed = free_flow * speed_factor * congestion_factor
        congestion_factor = max(1.0 - avg_density / avg_kj, self.min_speed_fraction)
        ev_speed = avg_vf * self.speed_factor * congestion_factor

        # Check if blocked at downstream signal (when near end of link)
        blocked = False
        if s.progress > 0.85:
            ds_node = self._downstream_nodes[s.link_idx]
            if ds_node is not None:
                if not self._is_green_for_ev(link_id, ds_node):
                    ev_speed = 0.0
                    blocked = True

        # Track stops
        if blocked and not s._was_stopped:
            s.stops += 1
        s._was_stopped = blocked

        # --- Advance position ---
        distance_this_step = ev_speed * dt
        s.speed = ev_speed
        s.travel_time += dt
        s.distance_traveled += distance_this_step
        s.progress += distance_this_step / link_length

        # Check if EV moved to next link
        while s.progress >= 1.0 and not s.arrived:
            overflow = (s.progress - 1.0) * link_length
            s.link_idx += 1
            s.progress = 0.0

            if s.link_idx >= len(self.route):
                s.arrived = True
                break

            link_length = self._link_lengths[s.link_idx]
            if link_length > 1e-6:
                s.progress = overflow / link_length
            else:
                s.progress = 1.0  # will trigger another iteration

        return s

    def _is_green_for_ev(self, link_id: LinkID, node_id: NodeID) -> bool:
        """Check if the signal at ``node_id`` is green for the EV's link."""
        net = self.engine.net
        sm = self.engine.signal_manager

        # Get current phase at this node
        current_phase_local = sm.get_node_phase(node_id)
        if current_phase_local is None:
            return True  # unsignalized node — always green

        # Get global phase index
        phases = net.node_phases.get(node_id, [])
        if not phases or current_phase_local >= len(phases):
            return True

        global_phase_idx = phases[current_phase_local]

        # Check if any movement from this link is served by this phase
        last_cell = net.link_last_cell[link_id]
        for mid in range(net.n_movements):
            if net.mov_from_cell[mid] == last_cell:
                if net.phase_mov_mask[global_phase_idx, mid]:
                    return True

        return False

    # --- Convenience properties ---

    @property
    def arrived(self) -> bool:
        """True once the EV has reached its destination."""
        return self.state.arrived

    @property
    def travel_time(self) -> float:
        """Cumulative travel time in seconds."""
        return self.state.travel_time

    @property
    def position(self) -> tuple[LinkID, float]:
        """Current (link_id, progress) pair."""
        if self.state.link_idx < len(self.route):
            return self.route[self.state.link_idx], self.state.progress
        return self.route[-1], 1.0

    @property
    def current_link(self) -> LinkID:
        """The link the EV is currently on."""
        idx = min(self.state.link_idx, len(self.route) - 1)
        return self.route[idx]

    @property
    def fraction_completed(self) -> float:
        """Fraction of route completed [0, 1]."""
        if not self.route:
            return 1.0
        total_links = len(self.route)
        completed = self.state.link_idx + self.state.progress
        return min(completed / total_links, 1.0)

    def get_ev_observation(self) -> dict:
        """Return a dict of EV state for use in observations.

        Returns
        -------
        dict
            Keys: ``link_idx``, ``progress``, ``speed``, ``travel_time``,
            ``distance_traveled``, ``arrived``, ``stops``,
            ``fraction_completed``.
        """
        s = self.state
        return {
            "link_idx": s.link_idx,
            "progress": s.progress,
            "speed": s.speed,
            "travel_time": s.travel_time,
            "distance_traveled": s.distance_traveled,
            "arrived": s.arrived,
            "stops": s.stops,
            "fraction_completed": self.fraction_completed,
        }
