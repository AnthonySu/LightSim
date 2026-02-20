"""Travel time estimation for CTM-based simulations.

Since CTM tracks aggregate density (not individual vehicles), travel time
is estimated from the speed field derived from the fundamental diagram:
  - Free-flow: speed = vf
  - Congested: speed = Q / density  (flow / density = speed)

TravelTimeTracker accumulates per-link travel time estimates over
simulation steps.
"""

from __future__ import annotations

import numpy as np

from ..core.engine import SimulationEngine
from ..core.types import FLOAT, LinkID


def estimate_link_travel_time(engine: SimulationEngine, link_id: LinkID) -> float:
    """Estimate current travel time (seconds) on a link.

    Uses the density-based speed at each cell to compute the
    traversal time: sum(cell_length / speed) across all cells.
    """
    net = engine.net
    cells = net.link_cells[link_id]
    density = engine.state.density[cells]
    vf = net.vf[cells]
    Q = net.Q[cells]
    lanes = net.lanes[cells]
    length = net.length[cells]

    k_crit = Q / vf
    # Speed from triangular FD
    speed = np.where(
        density <= k_crit,
        vf,
        np.where(density > 1e-9, Q * lanes / (density * lanes + 1e-9), vf),
    )
    speed = np.maximum(speed, 0.1)
    return float((length / speed).sum())


def estimate_link_free_flow_tt(engine: SimulationEngine, link_id: LinkID) -> float:
    """Free-flow travel time (seconds) on a link: sum(cell_length / vf)."""
    net = engine.net
    cells = net.link_cells[link_id]
    return float((net.length[cells] / net.vf[cells]).sum())


class TravelTimeTracker:
    """Accumulates travel time estimates over simulation steps.

    Call ``update(engine)`` each step to record per-link travel times.
    Query ``get_mean_travel_time(link_id)`` for the time-averaged estimate.

    Parameters
    ----------
    link_ids : list[LinkID] | None
        Links to track.  If None, tracks all links.
    window : int
        Rolling window size (number of steps) for averaging.
        Set to 0 for cumulative average.
    """

    def __init__(
        self,
        link_ids: list[LinkID] | None = None,
        window: int = 0,
    ) -> None:
        self._link_ids = link_ids
        self._window = window
        self._history: dict[LinkID, list[float]] = {}
        self._count = 0

    def update(self, engine: SimulationEngine) -> None:
        """Record travel times for the current simulation state."""
        links = self._link_ids or list(engine.net.link_cells.keys())
        for lid in links:
            tt = estimate_link_travel_time(engine, lid)
            if lid not in self._history:
                self._history[lid] = []
            self._history[lid].append(tt)
            if self._window > 0 and len(self._history[lid]) > self._window:
                self._history[lid] = self._history[lid][-self._window:]
        self._count += 1

    def get_mean_travel_time(self, link_id: LinkID) -> float:
        """Return the mean travel time for a link over the tracking window."""
        hist = self._history.get(link_id, [])
        if not hist:
            return 0.0
        return float(np.mean(hist))

    def get_current_travel_time(self, link_id: LinkID) -> float:
        """Return the most recent travel time estimate for a link."""
        hist = self._history.get(link_id, [])
        if not hist:
            return 0.0
        return hist[-1]

    def get_network_mean_travel_time(self, engine: SimulationEngine) -> float:
        """Return the length-weighted mean travel time across all tracked links."""
        total_tt = 0.0
        total_length = 0.0
        for lid, hist in self._history.items():
            if not hist:
                continue
            cells = engine.net.link_cells.get(lid, [])
            if not cells:
                continue
            link_length = float(engine.net.length[cells].sum())
            total_tt += float(np.mean(hist)) * link_length
            total_length += link_length
        if total_length < 1e-9:
            return 0.0
        return total_tt / total_length

    def get_travel_time_index(self, engine: SimulationEngine, link_id: LinkID) -> float:
        """Travel Time Index: actual_tt / free_flow_tt.  1.0 = free-flow."""
        ff_tt = estimate_link_free_flow_tt(engine, link_id)
        if ff_tt < 1e-9:
            return 1.0
        mean_tt = self.get_mean_travel_time(link_id)
        return mean_tt / ff_tt if mean_tt > 0 else 1.0

    def reset(self) -> None:
        """Clear all recorded history."""
        self._history.clear()
        self._count = 0
