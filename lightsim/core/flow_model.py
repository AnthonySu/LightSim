"""Flow models for the CTM simulation.

``FlowModel`` is the abstract base class.  ``CTMFlowModel`` implements the
standard triangular‐fundamental‐diagram Cell Transmission Model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .network import CompiledNetwork
from .types import FLOAT


class FlowModel(ABC):
    """Abstract flow model interface."""

    @abstractmethod
    def compute_sending_flow(
        self, density: np.ndarray, net: CompiledNetwork
    ) -> np.ndarray:
        """Maximum flow each cell can *send* downstream (veh/s)."""

    @abstractmethod
    def compute_receiving_flow(
        self, density: np.ndarray, net: CompiledNetwork
    ) -> np.ndarray:
        """Maximum flow each cell can *receive* from upstream (veh/s)."""

    @abstractmethod
    def compute_flow(
        self,
        density: np.ndarray,
        sending: np.ndarray,
        receiving: np.ndarray,
        signal_mask: np.ndarray,
        net: CompiledNetwork,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute actual cell-to-cell flows.

        Returns
        -------
        intra_flow : ndarray, shape (n_cells,)
            Flow transferred from cell i to its downstream neighbour
            on the *same link* during this time step (vehicles).
        movement_flow : ndarray, shape (n_movements,)
            Flow transferred across each movement (vehicles).
        """


class CTMFlowModel(FlowModel):
    """Standard Cell Transmission Model with triangular FD.

    Sending flow:  S(k) = min(vf * k, Q) * lanes
    Receiving flow: R(k) = min(Q, w * (kj - k)) * lanes
    """

    def compute_sending_flow(
        self, density: np.ndarray, net: CompiledNetwork
    ) -> np.ndarray:
        return np.minimum(net.vf * density, net.Q) * net.lanes

    def compute_receiving_flow(
        self, density: np.ndarray, net: CompiledNetwork
    ) -> np.ndarray:
        return np.minimum(net.Q, net.w * (net.kj - density)) * net.lanes

    def compute_flow(
        self,
        density: np.ndarray,
        sending: np.ndarray,
        receiving: np.ndarray,
        signal_mask: np.ndarray,
        net: CompiledNetwork,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = net.n_cells
        intra_flow = np.zeros(n, dtype=FLOAT)
        movement_flow = np.zeros(net.n_movements, dtype=FLOAT)

        # --- 1. Intra-link flows (cell i → downstream_cell[i]) ---
        has_ds = net.downstream_cell >= 0
        src_idx = np.where(has_ds)[0]
        ds_idx = net.downstream_cell[src_idx]

        s_i = sending[src_idx]
        r_j = receiving[ds_idx]
        intra_flow[src_idx] = np.minimum(s_i, r_j) * dt

        # --- 2. Inter-link flows (movements across intersections) ---
        if net.n_movements == 0:
            return intra_flow, movement_flow

        # Sending flow from the last cell of each from-link
        mov_sending = sending[net.mov_from_cell]
        # Receiving flow of the first cell of each to-link
        mov_receiving = receiving[net.mov_to_cell]

        # Apply signal: zero flow for red movements
        effective_sending = mov_sending * signal_mask

        # Apply turn ratios: each movement gets its share of the upstream flow
        effective_sending = effective_sending * net.mov_turn_ratio

        # Cap by saturation rate
        effective_sending = np.minimum(effective_sending, net.mov_sat_rate)

        # Merge resolution: if multiple movements feed the same to-cell,
        # scale proportionally if total exceeds receiving capacity.
        # Group by to-cell
        to_cells_unique = np.unique(net.mov_to_cell)
        for tc in to_cells_unique:
            mask = net.mov_to_cell == tc
            total_demand = effective_sending[mask].sum()
            cap = receiving[tc]
            if total_demand > cap and total_demand > 1e-12:
                scale = cap / total_demand
                effective_sending[mask] *= scale

        # Diverge resolution: if multiple movements draw from the same from-cell,
        # ensure total doesn't exceed sending capacity.
        from_cells_unique = np.unique(net.mov_from_cell)
        for fc in from_cells_unique:
            mask = net.mov_from_cell == fc
            total_demand = effective_sending[mask].sum()
            cap = sending[fc]
            if total_demand > cap and total_demand > 1e-12:
                scale = cap / total_demand
                effective_sending[mask] *= scale

        movement_flow[:] = effective_sending * dt

        return intra_flow, movement_flow
