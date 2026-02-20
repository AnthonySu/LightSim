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
        capacity_factor: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute actual cell-to-cell flows.

        Parameters
        ----------
        capacity_factor : ndarray, shape (n_movements,), optional
            Per-movement capacity reduction factor (0–1) for start-up
            lost time.  If *None*, no reduction is applied.

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

    def __init__(self) -> None:
        self._intra_flow: np.ndarray | None = None
        self._movement_flow: np.ndarray | None = None
        self._merge_totals: np.ndarray | None = None
        self._diverge_totals: np.ndarray | None = None

    def _ensure_buffers(self, net: CompiledNetwork) -> None:
        """Lazy-init reusable buffers on first call or shape change."""
        n = net.n_cells
        m = net.n_movements
        if self._intra_flow is None or len(self._intra_flow) != n:
            self._intra_flow = np.zeros(n, dtype=FLOAT)
            self._merge_totals = np.zeros(n, dtype=FLOAT)
            self._diverge_totals = np.zeros(n, dtype=FLOAT)
        if self._movement_flow is None or len(self._movement_flow) != m:
            self._movement_flow = np.zeros(m, dtype=FLOAT)

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
        capacity_factor: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_buffers(net)
        n = net.n_cells
        intra_flow = self._intra_flow
        movement_flow = self._movement_flow
        intra_flow[:] = 0.0
        movement_flow[:] = 0.0

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

        # Apply signal: zero flow for red movements
        effective_sending = mov_sending * signal_mask

        # Apply capacity factor (start-up lost time ramp)
        if capacity_factor is not None:
            effective_sending *= capacity_factor

        # Apply turn ratios: each movement gets its share of the upstream flow
        effective_sending *= net.mov_turn_ratio

        # Cap by saturation rate
        np.minimum(effective_sending, net.mov_sat_rate, out=effective_sending)

        # Merge resolution: scale proportionally if total demand on a to-cell
        # exceeds its receiving capacity (fully vectorized).
        merge_totals = self._merge_totals
        merge_totals[:] = 0.0
        np.add.at(merge_totals, net.mov_to_cell, effective_sending)
        per_mov_merge = np.maximum(merge_totals[net.mov_to_cell], 1e-12)
        merge_scale = np.minimum(1.0, receiving[net.mov_to_cell] / per_mov_merge)
        effective_sending *= merge_scale

        # Diverge resolution: scale if total demand from a from-cell
        # exceeds its sending capacity (fully vectorized).
        diverge_totals = self._diverge_totals
        diverge_totals[:] = 0.0
        np.add.at(diverge_totals, net.mov_from_cell, effective_sending)
        per_mov_diverge = np.maximum(diverge_totals[net.mov_from_cell], 1e-12)
        diverge_scale = np.minimum(1.0, sending[net.mov_from_cell] / per_mov_diverge)
        effective_sending *= diverge_scale

        movement_flow[:] = effective_sending * dt

        return intra_flow, movement_flow
