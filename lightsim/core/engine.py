"""SimulationEngine: the main CTM step loop.

Orchestrates the flow model, signal manager, and demand manager to advance
the simulation one time step at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .demand import DemandManager, DemandProfile
from .flow_model import CTMFlowModel, FlowModel
from .network import CompiledNetwork, Network
from .signal import (
    FixedTimeController,
    RLController,
    SignalController,
    SignalManager,
)
from .types import FLOAT, LinkID, NodeID


@dataclass
class SimState:
    """Snapshot of the simulation state."""
    density: np.ndarray          # veh/m/lane per cell
    time: float = 0.0
    step_count: int = 0
    total_entered: float = 0.0
    total_exited: float = 0.0


class SimulationEngine:
    """Main CTM simulation engine.

    Parameters
    ----------
    network : Network
        The logical network (will be compiled).
    dt : float
        Time step in seconds.
    flow_model : FlowModel, optional
        Defaults to ``CTMFlowModel``.
    controller : SignalController, optional
        Defaults to ``FixedTimeController``.
    demand_profiles : list[DemandProfile], optional
        Time-varying demand for source links.
    """

    def __init__(
        self,
        network: Network,
        dt: float = 1.0,
        flow_model: FlowModel | None = None,
        controller: SignalController | None = None,
        demand_profiles: list[DemandProfile] | None = None,
    ) -> None:
        self.dt = dt
        self.network = network
        self.net = network.compile(dt)
        self.flow_model = flow_model or CTMFlowModel()
        self.controller = controller or FixedTimeController()
        self.signal_manager = SignalManager(self.net, self.controller)
        self.demand_manager = DemandManager(self.net, demand_profiles)
        self.state = SimState(
            density=np.zeros(self.net.n_cells, dtype=FLOAT),
        )

    def reset(self, seed: int | None = None) -> SimState:
        """Reset the simulation to time zero."""
        if seed is not None:
            np.random.seed(seed)
        self.state = SimState(
            density=np.zeros(self.net.n_cells, dtype=FLOAT),
        )
        self.signal_manager = SignalManager(self.net, self.controller)
        return self.state

    def step(self) -> SimState:
        """Advance the simulation by one time step ``dt``.

        Returns the updated SimState.
        """
        s = self.state
        net = self.net
        dt = self.dt
        density = s.density

        # 1. Compute sending and receiving flows
        sending = self.flow_model.compute_sending_flow(density, net)
        receiving = self.flow_model.compute_receiving_flow(density, net)

        # 2. Get signal mask
        signal_mask = self.signal_manager.get_movement_mask()

        # 3. Compute flows
        intra_flow, movement_flow = self.flow_model.compute_flow(
            density, sending, receiving, signal_mask, net, dt,
        )

        # 4. Update densities from intra-link flows
        # intra_flow[i] = vehicles moving from cell i to downstream_cell[i]
        vehicles = density * net.length * net.lanes  # current vehicles per cell

        # Subtract outgoing intra-link flow
        has_ds = net.downstream_cell >= 0
        vehicles[has_ds] -= intra_flow[has_ds]

        # Add incoming intra-link flow
        has_us = net.upstream_cell >= 0
        src_cells = net.upstream_cell[has_us]
        vehicles[has_us] += intra_flow[src_cells]

        # 5. Update densities from movement flows
        if net.n_movements > 0:
            # Subtract from source cells (last cell of from-link)
            np.subtract.at(vehicles, net.mov_from_cell, movement_flow)
            # Add to destination cells (first cell of to-link)
            np.add.at(vehicles, net.mov_to_cell, movement_flow)

        # 6. Inject demand at sources
        injection = self.demand_manager.get_injection(s.time, dt, density)
        vehicles += injection
        s.total_entered += injection.sum()

        # 7. Remove vehicles at sinks
        sink_flow = np.zeros(net.n_cells, dtype=FLOAT)
        sink_mask = net.is_sink
        # Sinks absorb the sending flow from their cell
        sink_sending = sending[sink_mask] * dt
        # But cap by available vehicles
        sink_flow[sink_mask] = np.minimum(sink_sending, vehicles[sink_mask])
        vehicles -= sink_flow
        s.total_exited += sink_flow.sum()

        # 8. Convert back to density, clamp non-negative
        lane_length = net.length * net.lanes
        density_new = np.maximum(vehicles / lane_length, 0.0)
        # Also clamp at jam density
        density_new = np.minimum(density_new, net.kj)

        s.density = density_new
        s.time += dt
        s.step_count += 1

        # 9. Advance signals
        self.signal_manager.step(dt, density_new)

        return s

    # --- Convenience accessors ---

    def get_link_density(self, link_id: LinkID) -> float:
        """Average density (veh/m/lane) on a link."""
        cells = self.net.link_cells[link_id]
        return float(self.state.density[cells].mean())

    def get_link_vehicles(self, link_id: LinkID) -> float:
        """Total vehicles on a link."""
        cells = self.net.link_cells[link_id]
        return float(
            (self.state.density[cells] * self.net.length[cells] * self.net.lanes[cells]).sum()
        )

    def get_link_queue(self, link_id: LinkID) -> float:
        """Approximate queue length in vehicles (cells above critical density)."""
        cells = self.net.link_cells[link_id]
        k_crit = self.net.Q[cells] / self.net.vf[cells]  # critical density
        queued = self.state.density[cells] > k_crit
        return float(
            (self.state.density[cells][queued] * self.net.length[cells][queued]
             * self.net.lanes[cells][queued]).sum()
        )

    def get_total_vehicles(self) -> float:
        """Total vehicles in the network."""
        return float(
            (self.state.density * self.net.length * self.net.lanes).sum()
        )

    def get_network_metrics(self) -> dict:
        """Return a dict of network-wide metrics."""
        density = self.state.density
        n = self.net
        vehicles = density * n.length * n.lanes
        total_veh = float(vehicles.sum())
        # Average density weighted by link length
        total_length = float((n.length * n.lanes).sum())
        avg_density = total_veh / total_length if total_length > 0 else 0.0
        return {
            "time": self.state.time,
            "total_vehicles": total_veh,
            "avg_density": avg_density,
            "total_entered": self.state.total_entered,
            "total_exited": self.state.total_exited,
        }
