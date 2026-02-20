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

    def __repr__(self) -> str:
        n_cells = len(self.density) if self.density is not None else 0
        return (
            f"SimState(t={self.time:.1f}s, step={self.step_count}, "
            f"{n_cells} cells, entered={self.total_entered:.0f}, "
            f"exited={self.total_exited:.0f})"
        )


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
        stochastic: bool = False,
    ) -> None:
        self.dt = dt
        self.network = network
        self.net = network.compile(dt)
        self.flow_model = flow_model or CTMFlowModel()
        self.controller = controller or FixedTimeController()
        self.stochastic = stochastic
        self._rng: np.random.Generator | None = None
        self.signal_manager = SignalManager(self.net, self.controller)
        self.demand_manager = DemandManager(
            self.net, demand_profiles,
            stochastic=stochastic, rng=self._rng,
        )
        self.demand_profiles = demand_profiles
        self.state = SimState(
            density=np.zeros(self.net.n_cells, dtype=FLOAT),
        )

        # Pre-compute topology masks and reusable buffers
        net = self.net
        self._lane_length = net.length * net.lanes  # constant per cell
        self._has_ds = net.downstream_cell >= 0
        self._has_us = net.upstream_cell >= 0
        self._us_src = net.upstream_cell[self._has_us]
        self._vehicles = np.zeros(net.n_cells, dtype=FLOAT)
        self._sink_flow = np.zeros(net.n_cells, dtype=FLOAT)

    def reset(self, seed: int | None = None) -> SimState:
        """Reset the simulation to time zero."""
        self._rng = np.random.default_rng(seed)
        self.state = SimState(
            density=np.zeros(self.net.n_cells, dtype=FLOAT),
        )
        self.signal_manager = SignalManager(self.net, self.controller)
        self.demand_manager = DemandManager(
            self.net, self.demand_profiles,
            stochastic=self.stochastic, rng=self._rng,
        )
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

        # 2. Get signal mask and capacity factor
        signal_mask = self.signal_manager.get_movement_mask()
        capacity_factor = self.signal_manager.get_capacity_factor()

        # 3. Compute flows
        intra_flow, movement_flow = self.flow_model.compute_flow(
            density, sending, receiving, signal_mask, net, dt,
            capacity_factor=capacity_factor,
        )

        # 4. Update densities from intra-link flows (reuse pre-allocated buffer)
        vehicles = self._vehicles
        np.multiply(density, self._lane_length, out=vehicles)

        # Subtract outgoing intra-link flow
        vehicles[self._has_ds] -= intra_flow[self._has_ds]

        # Add incoming intra-link flow
        vehicles[self._has_us] += intra_flow[self._us_src]

        # 5. Update densities from movement flows
        if net.n_movements > 0:
            np.subtract.at(vehicles, net.mov_from_cell, movement_flow)
            np.add.at(vehicles, net.mov_to_cell, movement_flow)

        # 6. Inject demand at sources
        injection = self.demand_manager.get_injection(s.time, dt, density)
        vehicles += injection
        s.total_entered += float(injection.sum())

        # 7. Remove vehicles at sinks (reuse pre-allocated buffer)
        sink_flow = self._sink_flow
        sink_flow[:] = 0.0
        sink_mask = net.is_sink
        sink_sending = sending[sink_mask] * dt
        sink_flow[sink_mask] = np.minimum(sink_sending, vehicles[sink_mask])
        vehicles -= sink_flow
        s.total_exited += float(sink_flow.sum())

        # 8. Convert back to density, clamp to [0, kj]
        np.divide(vehicles, self._lane_length, out=density)
        np.maximum(density, 0.0, out=density)
        np.minimum(density, net.kj, out=density)

        s.time += dt
        s.step_count += 1

        # 9. Advance signals
        self.signal_manager.step(dt, density)

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

    def get_link_speed(self, link_id: LinkID) -> float:
        """Average speed (m/s) on a link, weighted by cell length."""
        cells = self.net.link_cells[link_id]
        density = self.state.density[cells]
        vf = self.net.vf[cells]
        Q = self.net.Q[cells]
        lanes = self.net.lanes[cells]
        length = self.net.length[cells]
        k_crit = Q / vf
        speed = np.where(
            density <= k_crit,
            vf,
            np.where(density > 1e-9, Q * lanes / (density * lanes + 1e-9), vf),
        )
        speed = np.maximum(speed, 0.1)
        total_length = float(length.sum())
        if total_length < 1e-9:
            return 0.0
        return float((speed * length).sum() / total_length)

    def get_total_vehicles(self) -> float:
        """Total vehicles in the network."""
        return float(
            (self.state.density * self.net.length * self.net.lanes).sum()
        )

    def get_network_metrics(self) -> dict:
        """Return a dict of network-wide metrics.

        Keys: ``time``, ``total_vehicles``, ``avg_density``,
        ``total_entered``, ``total_exited``.
        """
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
