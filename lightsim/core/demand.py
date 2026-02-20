"""Demand generation for LightSim.

DemandProfile defines time-varying demand for source cells.
DemandManager injects vehicles into the network at each time step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .network import CompiledNetwork
from .types import FLOAT, CellID, LinkID


@dataclass
class DemandProfile:
    """Piecewise-constant demand for one source link.

    Parameters
    ----------
    link_id : LinkID
        The origin link whose first cell receives vehicles.
    time_points : list[float]
        Breakpoints in seconds (must start with 0).
    flow_rates : list[float]
        Demand flow in veh/s for each interval.
        ``flow_rates[i]`` applies for ``time_points[i] <= t < time_points[i+1]``.
    """
    link_id: LinkID
    time_points: list[float] = field(default_factory=lambda: [0.0])
    flow_rates: list[float] = field(default_factory=lambda: [0.0])

    def __post_init__(self) -> None:
        self._tp = np.asarray(self.time_points, dtype=FLOAT)
        self._rates = np.asarray(self.flow_rates, dtype=FLOAT)

    def get_rate(self, t: float) -> float:
        """Return demand rate (veh/s) at time t (O(log n) binary search)."""
        idx = int(np.searchsorted(self._tp, t, side="right")) - 1
        idx = max(0, min(idx, len(self._rates) - 1))
        return float(self._rates[idx])


class DemandManager:
    """Injects vehicles into source cells each time step."""

    def __init__(
        self,
        net: CompiledNetwork,
        profiles: list[DemandProfile] | None = None,
        stochastic: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.net = net
        self.profiles = profiles or []
        self.stochastic = stochastic
        self.rng = rng
        # Map link_id → cell_id for source cells
        self._source_cells: dict[LinkID, CellID] = {}
        for p in self.profiles:
            if p.link_id in net.link_first_cell:
                self._source_cells[p.link_id] = net.link_first_cell[p.link_id]

        # Pre-compute per-source-cell capacity (kj * lanes * length)
        # and lane_length (lanes * length) to avoid recomputing each step
        self._cell_cap: dict[CellID, float] = {}
        self._lane_length: dict[CellID, float] = {}
        for cid in self._source_cells.values():
            self._cell_cap[cid] = float(
                net.kj[cid] * net.lanes[cid] * net.length[cid]
            )
            self._lane_length[cid] = float(net.length[cid] * net.lanes[cid])

    def get_injection(self, t: float, dt: float, density: np.ndarray) -> np.ndarray:
        """Compute vehicles to inject into source cells.

        Injection is capped by the receiving flow capacity of the source cell.
        If ``stochastic=True``, demand is drawn from a Poisson distribution
        with mean ``rate * dt`` instead of the deterministic value.

        Returns
        -------
        injection : ndarray, shape (n_cells,)
            Vehicles to add to each cell.
        """
        injection = np.zeros(self.net.n_cells, dtype=FLOAT)
        for profile in self.profiles:
            cid = self._source_cells.get(profile.link_id)
            if cid is None:
                continue
            rate = profile.get_rate(t)
            mean_veh = rate * dt
            if self.stochastic and self.rng is not None:
                demand_veh = float(self.rng.poisson(mean_veh))
            else:
                demand_veh = mean_veh
            # Cap by available space (using pre-computed cell capacity)
            current_veh = density[cid] * self._lane_length[cid]
            space = max(0.0, self._cell_cap[cid] - current_veh)
            injection[cid] = min(demand_veh, space)
        return injection
