"""Network-wide metrics: throughput, delay, queue, MFD, occupancy."""

from __future__ import annotations

import numpy as np

from ..core.engine import SimulationEngine
from ..core.types import FLOAT, LinkID, NodeID


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


def compute_link_occupancy(engine: SimulationEngine, link_id: LinkID) -> float:
    """Occupancy rate (0–1) for a link: average density / jam density."""
    cells = engine.net.link_cells[link_id]
    density = engine.state.density[cells]
    kj = engine.net.kj[cells]
    length = engine.net.length[cells]
    lanes = engine.net.lanes[cells]
    weight = length * lanes
    total_weight = float(weight.sum())
    if total_weight < 1e-9:
        return 0.0
    occ = float((density / np.maximum(kj, 1e-12) * weight).sum() / total_weight)
    return min(occ, 1.0)


def compute_network_occupancy(engine: SimulationEngine) -> float:
    """Network-wide occupancy rate (0–1): weighted average of density/kj."""
    density = engine.state.density
    kj = engine.net.kj
    length = engine.net.length
    lanes = engine.net.lanes
    weight = length * lanes
    total_weight = float(weight.sum())
    if total_weight < 1e-9:
        return 0.0
    occ = float((density / np.maximum(kj, 1e-12) * weight).sum() / total_weight)
    return min(occ, 1.0)


def detect_spillback(engine: SimulationEngine, link_id: LinkID,
                     threshold: float = 0.9) -> bool:
    """Detect if congestion has spilled back to the first cell of a link.

    Returns True if the first cell's occupancy exceeds ``threshold`` (fraction
    of jam density).  This typically means the queue has backed up past the
    upstream intersection.
    """
    first_cell = engine.net.link_first_cell[link_id]
    kj = engine.net.kj[first_cell]
    if kj < 1e-12:
        return False
    return float(engine.state.density[first_cell] / kj) > threshold


def compute_link_queue_length(engine: SimulationEngine, link_id: LinkID) -> float:
    """Queue length in metres on a link (sum of congested cell lengths)."""
    cells = engine.net.link_cells[link_id]
    density = engine.state.density[cells]
    vf = engine.net.vf[cells]
    Q = engine.net.Q[cells]
    length = engine.net.length[cells]
    k_crit = Q / vf
    congested = density > k_crit
    return float(length[congested].sum())


def compute_network_delay(engine: SimulationEngine) -> float:
    """Total network delay (vehicle-seconds waiting) at current time step.

    Sums (density - k_crit) * length * lanes * dt for all congested cells.
    This gives the instantaneous vehicle-seconds of delay.
    """
    net = engine.net
    density = engine.state.density
    k_crit = net.Q / net.vf
    excess = np.maximum(density - k_crit, 0.0)
    return float((excess * net.length * net.lanes).sum()) * engine.dt


def compute_movement_counts(engine: SimulationEngine, node_id: NodeID) -> dict[int, float]:
    """Estimate turning movement flow (veh/s) at a node.

    Uses the current density and signal state to compute instantaneous flow
    for each movement.  Useful for turning movement count analysis.
    """
    net = engine.net
    density = engine.state.density
    movements = net.node_movements.get(node_id, [])
    mask = engine.signal_manager.get_movement_mask()
    counts: dict[int, float] = {}
    for mid in movements:
        from_cell = net.mov_from_cell[mid]
        k = density[from_cell]
        vf = net.vf[from_cell]
        Q = net.Q[from_cell]
        lanes = net.lanes[from_cell]
        flow = min(vf * k, Q) * lanes * net.mov_turn_ratio[mid]
        counts[mid] = float(flow * mask[mid])
    return counts
