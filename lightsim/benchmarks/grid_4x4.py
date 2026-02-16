"""4x4 grid benchmark scenario."""

from __future__ import annotations

from ..core.demand import DemandProfile
from ..core.network import Network
from ..core.types import LinkID, NodeID, NodeType
from ..networks.grid import create_grid_network


def create_grid_4x4(
    link_length: float = 300.0,
    lanes: int = 2,
    demand_rate: float = 0.15,
) -> tuple[Network, list[DemandProfile]]:
    """Create a 4x4 grid network with uniform demand.

    Returns (network, demand_profiles).
    """
    net = create_grid_network(
        rows=4,
        cols=4,
        link_length=link_length,
        lanes=lanes,
        n_cells_per_link=3,
    )

    # Add demand at all origin nodes
    demand = []
    for link in net.links.values():
        from_node = net.nodes.get(link.from_node)
        if from_node is not None and from_node.node_type == NodeType.ORIGIN:
            demand.append(
                DemandProfile(link.link_id, [0.0], [demand_rate])
            )

    return net, demand
