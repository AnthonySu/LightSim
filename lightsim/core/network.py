"""Network topology: Node, Link, Cell, Movement, Phase, and Network.

The Network object holds the *logical* topology. Calling ``network.compile(dt)``
produces a ``CompiledNetwork`` that contains the flat NumPy index arrays the
CTM flow model needs for vectorised computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .types import (
    FLOAT,
    INT,
    CellID,
    LinkID,
    MovementID,
    NodeID,
    NodeType,
    PhaseID,
    TurnType,
)


# ---------------------------------------------------------------------------
# Logical topology dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    """One CTM cell on a link."""
    cell_id: CellID
    link_id: LinkID
    length: float          # metres
    lanes: int
    free_flow_speed: float # m/s
    wave_speed: float      # m/s  (backward wave)
    jam_density: float     # veh/m/lane
    capacity: float        # veh/s/lane  (Q)


@dataclass
class Link:
    """A directional link between two nodes, composed of cells."""
    link_id: LinkID
    from_node: NodeID
    to_node: NodeID
    cells: list[Cell] = field(default_factory=list)

    @property
    def num_cells(self) -> int:
        return len(self.cells)


@dataclass
class Movement:
    """A turning movement: from‐link → to‐link through a node."""
    movement_id: MovementID
    from_link: LinkID
    to_link: LinkID
    node_id: NodeID
    turn_type: TurnType
    turn_ratio: float = 1.0   # fraction of upstream flow
    saturation_rate: float | None = None  # veh/s, None → use cell capacity


@dataclass
class Phase:
    """A signal phase: which movements are served."""
    phase_id: PhaseID
    movements: list[MovementID]
    min_green: float = 5.0    # seconds
    max_green: float = 60.0   # seconds
    yellow: float = 3.0       # seconds
    all_red: float = 2.0      # seconds
    lost_time: float = 0.0    # seconds – start-up lost time per green onset


@dataclass
class Node:
    """An intersection or terminal node in the network.

    Signalized nodes have one or more ``Phase`` objects defining which
    movements may discharge simultaneously.
    """
    node_id: NodeID
    node_type: NodeType
    phases: list[Phase] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0


# ---------------------------------------------------------------------------
# CompiledNetwork – flat arrays for vectorised CTM
# ---------------------------------------------------------------------------

@dataclass
class CompiledNetwork:
    """Pre-computed index arrays for the CTM flow model.

    All cell-level properties are stored as 1-D arrays of length ``n_cells``.
    Connectivity is stored as index arrays into these cell arrays.
    """
    n_cells: int
    n_movements: int

    # --- per-cell properties (length n_cells) ---
    length: np.ndarray       # metres
    lanes: np.ndarray        # int
    vf: np.ndarray           # free-flow speed  m/s
    w: np.ndarray            # wave speed  m/s
    kj: np.ndarray           # jam density  veh/m/lane
    Q: np.ndarray            # capacity  veh/s/lane

    # --- intra-link connectivity ---
    # For each cell i, upstream_cell[i] is the predecessor cell on the same
    # link, or -1 if i is the first cell.  Likewise downstream_cell[i].
    upstream_cell: np.ndarray   # int, shape (n_cells,)
    downstream_cell: np.ndarray # int, shape (n_cells,)

    # --- source / sink masks ---
    is_source: np.ndarray    # bool, True for first cell of origin links
    is_sink: np.ndarray      # bool, True for last cell of destination links

    # --- movement connectivity ---
    # Movement m connects from_cell[m] → to_cell[m] with turn_ratio[m].
    mov_from_cell: np.ndarray   # int, shape (n_movements,)
    mov_to_cell: np.ndarray     # int, shape (n_movements,)
    mov_turn_ratio: np.ndarray  # float, shape (n_movements,)
    mov_sat_rate: np.ndarray    # float, shape (n_movements,)  veh/s
    mov_node: np.ndarray        # int, shape (n_movements,)

    # --- phase → movement mapping ---
    # phase_mov_mask[p] is a bool array of shape (n_movements,)
    phase_mov_mask: np.ndarray  # bool, shape (n_phases, n_movements)
    n_phases_per_node: dict[NodeID, int] = field(default_factory=dict)

    # --- per-phase timing arrays (length n_phases) ---
    phase_lost_time: np.ndarray = field(default_factory=lambda: np.empty(0))
    phase_yellow: np.ndarray = field(default_factory=lambda: np.empty(0))
    phase_all_red: np.ndarray = field(default_factory=lambda: np.empty(0))

    # --- look-ups ---
    link_first_cell: dict[LinkID, CellID] = field(default_factory=dict)
    link_last_cell: dict[LinkID, CellID] = field(default_factory=dict)
    link_cells: dict[LinkID, list[CellID]] = field(default_factory=dict)
    node_movements: dict[NodeID, list[MovementID]] = field(default_factory=dict)
    node_phases: dict[NodeID, list[PhaseID]] = field(default_factory=dict)
    global_phase_list: list[tuple[NodeID, PhaseID]] = field(default_factory=list)

    # --- pre-computed merge/diverge groups for O(M) flow resolution ---
    # merge_groups[to_cell] = array of movement indices feeding that cell
    merge_groups: dict[int, np.ndarray] = field(default_factory=dict)
    # diverge_groups[from_cell] = array of movement indices drawing from that cell
    diverge_groups: dict[int, np.ndarray] = field(default_factory=dict)

    # --- pre-computed node link maps for O(1) lookup ---
    node_incoming_links: dict[NodeID, list[LinkID]] = field(default_factory=dict)
    node_outgoing_links: dict[NodeID, list[LinkID]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class Network:
    """Mutable network builder.  Call ``compile(dt)`` to freeze."""

    def __init__(self) -> None:
        self.nodes: dict[NodeID, Node] = {}
        self.links: dict[LinkID, Link] = {}
        self.movements: dict[MovementID, Movement] = {}
        self._next_cell_id = 0
        self._next_movement_id = 0
        self._next_phase_id = 0

    # -- builder helpers -----------------------------------------------------

    def add_node(
        self,
        node_id: NodeID,
        node_type: NodeType,
        x: float = 0.0,
        y: float = 0.0,
    ) -> Node:
        """Add a node (intersection or terminal) to the network."""
        node = Node(node_id=node_id, node_type=node_type, x=x, y=y)
        self.nodes[node_id] = node
        return node

    def add_link(
        self,
        link_id: LinkID,
        from_node: NodeID,
        to_node: NodeID,
        length: float,
        lanes: int = 1,
        n_cells: int | None = None,
        free_flow_speed: float = 13.89,   # 50 km/h
        wave_speed: float = 5.56,         # 20 km/h
        jam_density: float = 0.15,        # veh/m/lane (~150 veh/km/lane)
        capacity: float = 0.5,            # veh/s/lane (~1800 veh/h/lane)
    ) -> Link:
        """Add a directional link and auto-create its CTM cells."""
        link = Link(link_id=link_id, from_node=from_node, to_node=to_node)
        if n_cells is None:
            n_cells = max(1, int(length / (free_flow_speed * 5)))  # default ~5 s cells
        cell_len = length / n_cells
        for _ in range(n_cells):
            cell = Cell(
                cell_id=CellID(self._next_cell_id),
                link_id=link_id,
                length=cell_len,
                lanes=lanes,
                free_flow_speed=free_flow_speed,
                wave_speed=wave_speed,
                jam_density=jam_density,
                capacity=capacity,
            )
            link.cells.append(cell)
            self._next_cell_id += 1
        self.links[link_id] = link
        return link

    def add_movement(
        self,
        from_link: LinkID,
        to_link: LinkID,
        node_id: NodeID,
        turn_type: TurnType = TurnType.THROUGH,
        turn_ratio: float = 1.0,
        saturation_rate: float | None = None,
    ) -> Movement:
        """Add a turning movement connecting two links through a node."""
        mid = MovementID(self._next_movement_id)
        self._next_movement_id += 1
        mov = Movement(
            movement_id=mid,
            from_link=from_link,
            to_link=to_link,
            node_id=node_id,
            turn_type=turn_type,
            turn_ratio=turn_ratio,
            saturation_rate=saturation_rate,
        )
        self.movements[mid] = mov
        return mov

    def add_phase(
        self,
        node_id: NodeID,
        movements: list[MovementID],
        min_green: float = 5.0,
        max_green: float = 60.0,
        yellow: float = 3.0,
        all_red: float = 2.0,
        lost_time: float = 0.0,
    ) -> Phase:
        """Add a signal phase to a node, grouping compatible movements."""
        pid = PhaseID(self._next_phase_id)
        self._next_phase_id += 1
        phase = Phase(
            phase_id=pid,
            movements=movements,
            min_green=min_green,
            max_green=max_green,
            yellow=yellow,
            all_red=all_red,
            lost_time=lost_time,
        )
        self.nodes[node_id].phases.append(phase)
        return phase

    # -- compile to flat arrays ----------------------------------------------

    def compile(self, dt: float) -> CompiledNetwork:
        """Compile the logical network into flat NumPy arrays.

        Parameters
        ----------
        dt : float
            Simulation time step in seconds.  Used to enforce CFL and set
            default cell lengths.
        """
        # Collect all cells in global order
        all_cells: list[Cell] = []
        for link in self.links.values():
            for cell in link.cells:
                all_cells.append(cell)
        n_cells = len(all_cells)

        # Cell property arrays
        length = np.array([c.length for c in all_cells], dtype=FLOAT)
        lanes = np.array([c.lanes for c in all_cells], dtype=INT)
        vf = np.array([c.free_flow_speed for c in all_cells], dtype=FLOAT)
        w = np.array([c.wave_speed for c in all_cells], dtype=FLOAT)
        kj = np.array([c.jam_density for c in all_cells], dtype=FLOAT)
        Q = np.array([c.capacity for c in all_cells], dtype=FLOAT)

        # CFL check
        min_cell_len = (vf * dt).max()
        violations = length < vf * dt - 1e-9
        if violations.any():
            bad = np.where(violations)[0]
            raise ValueError(
                f"CFL violation: cells {bad.tolist()} have length < vf*dt. "
                f"Min required: {min_cell_len:.2f} m"
            )

        # Intra-link connectivity
        upstream_cell = np.full(n_cells, -1, dtype=INT)
        downstream_cell = np.full(n_cells, -1, dtype=INT)
        link_first_cell: dict[LinkID, CellID] = {}
        link_last_cell: dict[LinkID, CellID] = {}
        link_cells_map: dict[LinkID, list[CellID]] = {}

        for link in self.links.values():
            cells = link.cells
            cids = [c.cell_id for c in cells]
            link_cells_map[link.link_id] = cids
            link_first_cell[link.link_id] = cids[0]
            link_last_cell[link.link_id] = cids[-1]
            for i in range(len(cids)):
                if i > 0:
                    upstream_cell[cids[i]] = cids[i - 1]
                if i < len(cids) - 1:
                    downstream_cell[cids[i]] = cids[i + 1]

        # Source / sink masks
        is_source = np.zeros(n_cells, dtype=bool)
        is_sink = np.zeros(n_cells, dtype=bool)
        for link in self.links.values():
            from_node = self.nodes[link.from_node]
            to_node = self.nodes[link.to_node]
            if from_node.node_type == NodeType.ORIGIN:
                is_source[link_first_cell[link.link_id]] = True
            if to_node.node_type == NodeType.DESTINATION:
                is_sink[link_last_cell[link.link_id]] = True

        # Movement arrays
        n_movements = len(self.movements)
        mov_from_cell = np.zeros(n_movements, dtype=INT)
        mov_to_cell = np.zeros(n_movements, dtype=INT)
        mov_turn_ratio = np.zeros(n_movements, dtype=FLOAT)
        mov_sat_rate = np.zeros(n_movements, dtype=FLOAT)
        mov_node = np.zeros(n_movements, dtype=INT)
        node_movements: dict[NodeID, list[MovementID]] = {}

        for mov in self.movements.values():
            m = mov.movement_id
            mov_from_cell[m] = link_last_cell[mov.from_link]
            mov_to_cell[m] = link_first_cell[mov.to_link]
            mov_turn_ratio[m] = mov.turn_ratio
            from_cell = all_cells[link_last_cell[mov.from_link]]
            if mov.saturation_rate is not None:
                mov_sat_rate[m] = mov.saturation_rate
            else:
                mov_sat_rate[m] = from_cell.capacity * from_cell.lanes
            mov_node[m] = mov.node_id
            node_movements.setdefault(mov.node_id, []).append(m)

        # Phase → movement mask
        all_phases: list[Phase] = []
        node_phases: dict[NodeID, list[PhaseID]] = {}
        n_phases_per_node: dict[NodeID, int] = {}
        global_phase_list: list[tuple[NodeID, PhaseID]] = []

        for node in self.nodes.values():
            if node.phases:
                pids = []
                for phase in node.phases:
                    all_phases.append(phase)
                    pids.append(phase.phase_id)
                    global_phase_list.append((node.node_id, phase.phase_id))
                node_phases[node.node_id] = pids
                n_phases_per_node[node.node_id] = len(pids)

        n_total_phases = len(all_phases)
        phase_mov_mask = np.zeros((n_total_phases, n_movements), dtype=bool)
        for phase in all_phases:
            for mid in phase.movements:
                phase_mov_mask[phase.phase_id, mid] = True

        # Per-phase timing arrays
        phase_lost_time = np.array(
            [p.lost_time for p in all_phases], dtype=FLOAT
        ) if all_phases else np.empty(0, dtype=FLOAT)
        phase_yellow = np.array(
            [p.yellow for p in all_phases], dtype=FLOAT
        ) if all_phases else np.empty(0, dtype=FLOAT)
        phase_all_red = np.array(
            [p.all_red for p in all_phases], dtype=FLOAT
        ) if all_phases else np.empty(0, dtype=FLOAT)

        # Pre-compute node → incoming/outgoing link maps
        node_incoming_links: dict[NodeID, list[LinkID]] = {}
        node_outgoing_links: dict[NodeID, list[LinkID]] = {}
        for link in self.links.values():
            node_incoming_links.setdefault(link.to_node, []).append(link.link_id)
            node_outgoing_links.setdefault(link.from_node, []).append(link.link_id)
        # Sort for deterministic ordering
        for k in node_incoming_links:
            node_incoming_links[k].sort()
        for k in node_outgoing_links:
            node_outgoing_links[k].sort()

        # Pre-compute merge/diverge groups for O(M) flow resolution
        merge_groups: dict[int, np.ndarray] = {}
        diverge_groups: dict[int, np.ndarray] = {}
        if n_movements > 0:
            for m in range(n_movements):
                tc = int(mov_to_cell[m])
                merge_groups.setdefault(tc, []).append(m)
                fc = int(mov_from_cell[m])
                diverge_groups.setdefault(fc, []).append(m)
            # Convert lists to arrays for fast indexing
            merge_groups = {k: np.array(v, dtype=INT) for k, v in merge_groups.items()}
            diverge_groups = {k: np.array(v, dtype=INT) for k, v in diverge_groups.items()}

        return CompiledNetwork(
            n_cells=n_cells,
            n_movements=n_movements,
            length=length,
            lanes=lanes,
            vf=vf,
            w=w,
            kj=kj,
            Q=Q,
            upstream_cell=upstream_cell,
            downstream_cell=downstream_cell,
            is_source=is_source,
            is_sink=is_sink,
            mov_from_cell=mov_from_cell,
            mov_to_cell=mov_to_cell,
            mov_turn_ratio=mov_turn_ratio,
            mov_sat_rate=mov_sat_rate,
            mov_node=mov_node,
            phase_mov_mask=phase_mov_mask,
            n_phases_per_node=n_phases_per_node,
            phase_lost_time=phase_lost_time,
            phase_yellow=phase_yellow,
            phase_all_red=phase_all_red,
            link_first_cell=link_first_cell,
            link_last_cell=link_last_cell,
            link_cells=link_cells_map,
            node_movements=node_movements,
            node_phases=node_phases,
            global_phase_list=global_phase_list,
            merge_groups=merge_groups,
            diverge_groups=diverge_groups,
            node_incoming_links=node_incoming_links,
            node_outgoing_links=node_outgoing_links,
        )
