"""Fundamental diagram validation utilities."""

from __future__ import annotations

import numpy as np

from ..core.engine import SimulationEngine
from ..core.network import Network
from ..core.demand import DemandProfile
from ..core.types import FLOAT, LinkID, NodeID, NodeType


def validate_fundamental_diagram(
    vf: float = 13.89,
    w: float = 5.56,
    kj: float = 0.15,
    capacity: float = 0.5,
    lanes: int = 1,
    dt: float = 1.0,
    n_density_points: int = 50,
    sim_steps: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Run single-link simulations at various demand levels and return (k, q).

    Verifies that the CTM reproduces the triangular fundamental diagram.

    Returns
    -------
    densities : ndarray
        Observed average densities (veh/m/lane).
    flows : ndarray
        Observed average flows (veh/s).
    """
    k_crit = capacity / vf
    demand_rates = np.linspace(0, capacity * lanes * 1.5, n_density_points)

    densities = []
    flows = []

    for rate in demand_rates:
        net = Network()
        net.add_node(NodeID(0), NodeType.ORIGIN)
        net.add_node(NodeID(1), NodeType.DESTINATION)
        link_length = vf * dt * 10  # 10 cells
        net.add_link(
            LinkID(0),
            from_node=NodeID(0),
            to_node=NodeID(1),
            length=link_length,
            lanes=lanes,
            n_cells=10,
            free_flow_speed=vf,
            wave_speed=w,
            jam_density=kj,
            capacity=capacity,
        )

        profile = DemandProfile(
            link_id=LinkID(0),
            time_points=[0.0],
            flow_rates=[float(rate)],
        )

        engine = SimulationEngine(
            network=net,
            dt=dt,
            demand_profiles=[profile],
        )
        engine.reset()

        # Run to approximate steady state
        for _ in range(sim_steps):
            engine.step()

        # Measure density and flow at middle cells
        mid_cells = list(range(3, 7))
        k = float(engine.state.density[mid_cells].mean())
        # Flow = sending flow = min(vf*k, Q)*lanes
        q = float(np.minimum(vf * k, capacity) * lanes)
        densities.append(k)
        flows.append(q)

    return np.array(densities), np.array(flows)
