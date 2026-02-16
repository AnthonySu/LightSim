"""Single intersection benchmark scenario."""

from __future__ import annotations

from ..core.demand import DemandProfile
from ..core.network import Network
from ..core.types import LinkID, NodeID, NodeType, TurnType


def create_single_intersection(
    lanes: int = 2,
    link_length: float = 300.0,
    n_cells: int = 3,
    free_flow_speed: float = 13.89,
    wave_speed: float = 5.56,
    jam_density: float = 0.15,
    capacity: float = 0.5,
) -> tuple[Network, list[DemandProfile]]:
    """Create a single 4-leg intersection with default demand.

    Layout::

            O(N)
            |
        O(W)--S--O(E)
            |
            O(S)

    Returns (network, demand_profiles).
    """
    net = Network()

    # Central signalised node
    net.add_node(NodeID(0), NodeType.SIGNALIZED, x=0, y=0)

    # Boundary origin/destination nodes
    # North
    net.add_node(NodeID(1), NodeType.ORIGIN, x=0, y=link_length)
    net.add_node(NodeID(2), NodeType.DESTINATION, x=0, y=link_length)
    # South
    net.add_node(NodeID(3), NodeType.ORIGIN, x=0, y=-link_length)
    net.add_node(NodeID(4), NodeType.DESTINATION, x=0, y=-link_length)
    # East
    net.add_node(NodeID(5), NodeType.ORIGIN, x=link_length, y=0)
    net.add_node(NodeID(6), NodeType.DESTINATION, x=link_length, y=0)
    # West
    net.add_node(NodeID(7), NodeType.ORIGIN, x=-link_length, y=0)
    net.add_node(NodeID(8), NodeType.DESTINATION, x=-link_length, y=0)

    kwargs = dict(
        length=link_length, lanes=lanes, n_cells=n_cells,
        free_flow_speed=free_flow_speed, wave_speed=wave_speed,
        jam_density=jam_density, capacity=capacity,
    )

    # Inbound links (origin → intersection)
    nb_in = net.add_link(LinkID(0), NodeID(1), NodeID(0), **kwargs)  # from north
    sb_in = net.add_link(LinkID(1), NodeID(3), NodeID(0), **kwargs)  # from south
    eb_in = net.add_link(LinkID(2), NodeID(5), NodeID(0), **kwargs)  # from east
    wb_in = net.add_link(LinkID(3), NodeID(7), NodeID(0), **kwargs)  # from west

    # Outbound links (intersection → destination)
    nb_out = net.add_link(LinkID(4), NodeID(0), NodeID(2), **kwargs)  # to north
    sb_out = net.add_link(LinkID(5), NodeID(0), NodeID(4), **kwargs)  # to south
    eb_out = net.add_link(LinkID(6), NodeID(0), NodeID(6), **kwargs)  # to east
    wb_out = net.add_link(LinkID(7), NodeID(0), NodeID(8), **kwargs)  # to west

    # Through movements
    # NS through: NB→SB, SB→NB
    ns_through = []
    m = net.add_movement(LinkID(0), LinkID(5), NodeID(0), TurnType.THROUGH, 0.5)
    ns_through.append(m.movement_id)
    m = net.add_movement(LinkID(1), LinkID(4), NodeID(0), TurnType.THROUGH, 0.5)
    ns_through.append(m.movement_id)

    # EW through: EB→WB, WB→EB
    ew_through = []
    m = net.add_movement(LinkID(2), LinkID(7), NodeID(0), TurnType.THROUGH, 0.5)
    ew_through.append(m.movement_id)
    m = net.add_movement(LinkID(3), LinkID(6), NodeID(0), TurnType.THROUGH, 0.5)
    ew_through.append(m.movement_id)

    # Right turns (always allowed — include in both phases for simplicity)
    right_turns = []
    m = net.add_movement(LinkID(0), LinkID(7), NodeID(0), TurnType.RIGHT, 0.25)
    right_turns.append(m.movement_id)
    m = net.add_movement(LinkID(1), LinkID(6), NodeID(0), TurnType.RIGHT, 0.25)
    right_turns.append(m.movement_id)
    m = net.add_movement(LinkID(2), LinkID(4), NodeID(0), TurnType.RIGHT, 0.25)
    right_turns.append(m.movement_id)
    m = net.add_movement(LinkID(3), LinkID(5), NodeID(0), TurnType.RIGHT, 0.25)
    right_turns.append(m.movement_id)

    # Left turns
    left_ns = []
    m = net.add_movement(LinkID(0), LinkID(6), NodeID(0), TurnType.LEFT, 0.25)
    left_ns.append(m.movement_id)
    m = net.add_movement(LinkID(1), LinkID(7), NodeID(0), TurnType.LEFT, 0.25)
    left_ns.append(m.movement_id)

    left_ew = []
    m = net.add_movement(LinkID(2), LinkID(5), NodeID(0), TurnType.LEFT, 0.25)
    left_ew.append(m.movement_id)
    m = net.add_movement(LinkID(3), LinkID(4), NodeID(0), TurnType.LEFT, 0.25)
    left_ew.append(m.movement_id)

    # Phase 1: NS (through + left + right)
    net.add_phase(NodeID(0), ns_through + left_ns + right_turns[:2])
    # Phase 2: EW (through + left + right)
    net.add_phase(NodeID(0), ew_through + left_ew + right_turns[2:])

    # Default demand profiles
    demand = [
        DemandProfile(LinkID(0), [0.0], [0.3]),  # North: 0.3 veh/s
        DemandProfile(LinkID(1), [0.0], [0.3]),  # South
        DemandProfile(LinkID(2), [0.0], [0.2]),  # East
        DemandProfile(LinkID(3), [0.0], [0.2]),  # West
    ]

    return net, demand
