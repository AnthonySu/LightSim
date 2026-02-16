"""Build a custom T-intersection and simulate it."""

from lightsim.core.demand import DemandProfile
from lightsim.core.engine import SimulationEngine
from lightsim.core.network import Network
from lightsim.core.signal import FixedTimeController
from lightsim.core.types import LinkID, NodeID, NodeType, TurnType

# Build a T-intersection: WB main street + NB side street
net = Network()

# Nodes
net.add_node(NodeID(0), NodeType.SIGNALIZED, x=0, y=0)     # intersection
net.add_node(NodeID(1), NodeType.ORIGIN, x=-500, y=0)      # west origin
net.add_node(NodeID(2), NodeType.DESTINATION, x=500, y=0)   # east dest
net.add_node(NodeID(3), NodeType.ORIGIN, x=0, y=300)       # north origin
net.add_node(NodeID(4), NodeType.DESTINATION, x=0, y=-300)  # south dest

# Links (CFL-safe: cell_length >= vf * dt = 13.89 * 1.0)
net.add_link(LinkID(0), NodeID(1), NodeID(0), length=500, lanes=2, n_cells=5)
net.add_link(LinkID(1), NodeID(0), NodeID(2), length=500, lanes=2, n_cells=5)
net.add_link(LinkID(2), NodeID(3), NodeID(0), length=300, lanes=1, n_cells=3)
net.add_link(LinkID(3), NodeID(0), NodeID(4), length=300, lanes=1, n_cells=3)

# Movements
m_wb = net.add_movement(LinkID(0), LinkID(1), NodeID(0), TurnType.THROUGH, 0.7)
m_wn = net.add_movement(LinkID(0), LinkID(3), NodeID(0), TurnType.LEFT, 0.3)
m_ns = net.add_movement(LinkID(2), LinkID(3), NodeID(0), TurnType.THROUGH, 0.5)
m_ne = net.add_movement(LinkID(2), LinkID(1), NodeID(0), TurnType.RIGHT, 0.5)

# Two-phase signal
net.add_phase(NodeID(0), [m_wb.movement_id, m_wn.movement_id])  # WB phase
net.add_phase(NodeID(0), [m_ns.movement_id, m_ne.movement_id])  # NB phase

# Demand
demand = [
    DemandProfile(LinkID(0), [0.0], [0.4]),
    DemandProfile(LinkID(2), [0.0], [0.15]),
]

# Simulate
controller = FixedTimeController({NodeID(0): [30.0, 20.0]})
engine = SimulationEngine(
    network=net, dt=1.0,
    controller=controller,
    demand_profiles=demand,
)
engine.reset(seed=42)

print("Time(s)  Vehicles  Entered  Exited")
print("-" * 40)
for step in range(600):
    engine.step()
    if (step + 1) % 100 == 0:
        m = engine.get_network_metrics()
        print(f"{m['time']:>6.0f}  {m['total_vehicles']:>8.1f}  "
              f"{m['total_entered']:>7.0f}  {m['total_exited']:>6.0f}")
