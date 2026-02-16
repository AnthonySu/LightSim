"""Network-wide metrics: throughput, delay, queue, MFD."""

from __future__ import annotations

import numpy as np

from ..core.engine import SimulationEngine
from ..core.types import FLOAT, LinkID


def compute_link_delay(engine: SimulationEngine, link_id: LinkID) -> float:
    """Estimate delay (seconds per vehicle) on a link vs free-flow travel time."""
    cells = engine.net.link_cells[link_id]
    density = engine.state.density[cells]
    vf = engine.net.vf[cells]
    length = engine.net.length[cells]
    lanes = engine.net.lanes[cells]
    Q = engine.net.Q[cells]

    # Free-flow travel time
    ff_tt = float((length / vf).sum())

    # Actual: estimate from density
    k_crit = Q / vf
    # Below critical density: travel at free-flow speed
    # Above: travel at w*(kj-k)/k  (approximate)
    w = engine.net.w[cells]
    kj = engine.net.kj[cells]

    speed = np.where(
        density < k_crit,
        vf,
        np.where(density > 1e-9, Q * lanes / (density * lanes + 1e-9), vf),
    )
    speed = np.maximum(speed, 0.1)  # avoid division by zero
    actual_tt = float((length / speed).sum())
    return max(0.0, actual_tt - ff_tt)


def compute_pressure(engine: SimulationEngine, node_id: int) -> float:
    """Compute intersection pressure: sum of upstream - downstream density differences."""
    net = engine.net
    movements = net.node_movements.get(node_id, [])
    pressure = 0.0
    for mid in movements:
        from_cell = net.mov_from_cell[mid]
        to_cell = net.mov_to_cell[mid]
        pressure += engine.state.density[from_cell] - engine.state.density[to_cell]
    return pressure


def compute_mfd(engine: SimulationEngine) -> tuple[float, float]:
    """Compute a point on the Macroscopic Fundamental Diagram.

    Returns (average_density, average_flow) across the network.
    """
    net = engine.net
    density = engine.state.density
    vf = net.vf
    Q = net.Q
    lanes = net.lanes
    length = net.length

    # Flow per cell using triangular FD
    flow = np.minimum(vf * density, Q) * lanes  # sending flow approximation

    total_length = float((length * lanes).sum())
    if total_length < 1e-9:
        return 0.0, 0.0

    avg_density = float((density * length * lanes).sum() / total_length)
    avg_flow = float((flow * length).sum() / total_length)
    return avg_density, avg_flow
