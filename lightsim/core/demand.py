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

    def get_rate(self, t: float) -> float:
        """Return demand rate (veh/s) at time t."""
        rate = self.flow_rates[0]
        for i, tp in enumerate(self.time_points):
            if t >= tp:
                rate = self.flow_rates[min(i, len(self.flow_rates) - 1)]
            else:
                break
        return rate


class DemandManager:
    """Injects vehicles into source cells each time step."""

    def __init__(
        self,
        net: CompiledNetwork,
        profiles: list[DemandProfile] | None = None,
    ) -> None:
        self.net = net
        self.profiles = profiles or []
        # Map link_id → cell_id for source cells
        self._source_cells: dict[LinkID, CellID] = {}
        for p in self.profiles:
            if p.link_id in net.link_first_cell:
                self._source_cells[p.link_id] = net.link_first_cell[p.link_id]

    def get_injection(self, t: float, dt: float, density: np.ndarray) -> np.ndarray:
        """Compute vehicles to inject into source cells.

        Injection is capped by the receiving flow capacity of the source cell.

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
            demand_veh = rate * dt
            # Cap by available space
            cell_cap = (
                self.net.kj[cid] * self.net.lanes[cid] * self.net.length[cid]
            )
            current_veh = density[cid] * self.net.length[cid] * self.net.lanes[cid]
            space = max(0.0, cell_cap - current_veh)
            injection[cid] = min(demand_veh, space)
        return injection
