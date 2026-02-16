"""5-intersection arterial benchmark scenario."""

from __future__ import annotations

from ..core.demand import DemandProfile
from ..core.network import Network
from ..core.types import LinkID, NodeID, NodeType
from ..networks.arterial import create_arterial_network


def create_arterial_5(
    link_length: float = 400.0,
    lanes: int = 2,
    main_demand: float = 0.4,
    side_demand: float = 0.1,
) -> tuple[Network, list[DemandProfile]]:
    """Create a 5-intersection arterial corridor with demand.

    Returns (network, demand_profiles).
    """
    net = create_arterial_network(
        n_intersections=5,
        link_length=link_length,
        lanes=lanes,
        n_cells_per_link=4,
    )

    # Add demand at all origin nodes
    demand = []
    for link in net.links.values():
        from_node = net.nodes.get(link.from_node)
        if from_node is not None and from_node.node_type == NodeType.ORIGIN:
            # Higher demand on main arterial
            node = from_node
            if node.node_id in (NodeID(0), NodeID(7)):  # main arterial origins
                rate = main_demand
            else:
                rate = side_demand
            demand.append(
                DemandProfile(link.link_id, [0.0], [rate])
            )

    return net, demand
