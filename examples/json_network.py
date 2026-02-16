"""Load a network from a JSON definition."""

import json
import tempfile

from lightsim.core.engine import SimulationEngine
from lightsim.core.demand import DemandProfile
from lightsim.core.signal import FixedTimeController
from lightsim.core.types import LinkID
from lightsim.networks.from_dict import from_dict

# Define a simple corridor as JSON
spec = {
    "nodes": [
        {"id": 0, "type": "origin",      "x": 0,   "y": 0},
        {"id": 1, "type": "signalized",   "x": 300, "y": 0},
        {"id": 2, "type": "signalized",   "x": 600, "y": 0},
        {"id": 3, "type": "destination",  "x": 900, "y": 0},
        {"id": 4, "type": "origin",       "x": 300, "y": 200},
        {"id": 5, "type": "destination",  "x": 300, "y": -200},
        {"id": 6, "type": "origin",       "x": 600, "y": 200},
        {"id": 7, "type": "destination",  "x": 600, "y": -200},
    ],
    "links": [
        {"id": 0, "from": 0, "to": 1, "length": 300, "lanes": 2, "n_cells": 3},
        {"id": 1, "from": 1, "to": 2, "length": 300, "lanes": 2, "n_cells": 3},
        {"id": 2, "from": 2, "to": 3, "length": 300, "lanes": 2, "n_cells": 3},
        {"id": 3, "from": 4, "to": 1, "length": 200, "lanes": 1, "n_cells": 2},
        {"id": 4, "from": 1, "to": 5, "length": 200, "lanes": 1, "n_cells": 2},
        {"id": 5, "from": 6, "to": 2, "length": 200, "lanes": 1, "n_cells": 2},
        {"id": 6, "from": 2, "to": 7, "length": 200, "lanes": 1, "n_cells": 2},
    ],
    "movements": [
        # Node 1: EB through, NB right
        {"from_link": 0, "to_link": 1, "node": 1, "turn_type": "through", "turn_ratio": 0.7},
        {"from_link": 3, "to_link": 4, "node": 1, "turn_type": "through", "turn_ratio": 0.5},
        {"from_link": 3, "to_link": 1, "node": 1, "turn_type": "right",   "turn_ratio": 0.5},
        # Node 2: EB through, NB right
        {"from_link": 1, "to_link": 2, "node": 2, "turn_type": "through", "turn_ratio": 0.7},
        {"from_link": 5, "to_link": 6, "node": 2, "turn_type": "through", "turn_ratio": 0.5},
        {"from_link": 5, "to_link": 2, "node": 2, "turn_type": "right",   "turn_ratio": 0.5},
    ],
    "phases": [
        {"node": 1, "movements": [0],    "min_green": 5, "max_green": 60},
        {"node": 1, "movements": [1, 2], "min_green": 5, "max_green": 60},
        {"node": 2, "movements": [3],    "min_green": 5, "max_green": 60},
        {"node": 2, "movements": [4, 5], "min_green": 5, "max_green": 60},
    ],
}

network = from_dict(spec)
print(f"Network: {len(network.nodes)} nodes, {len(network.links)} links, "
      f"{len(network.movements)} movements")

# Simulate
demand = [
    DemandProfile(LinkID(0), [0.0], [0.3]),
    DemandProfile(LinkID(3), [0.0], [0.1]),
    DemandProfile(LinkID(5), [0.0], [0.1]),
]

engine = SimulationEngine(
    network=network, dt=1.0,
    controller=FixedTimeController(),
    demand_profiles=demand,
)
engine.reset(seed=42)

for _ in range(600):
    engine.step()

m = engine.get_network_metrics()
print(f"After 600s: {m['total_vehicles']:.1f} vehicles, "
      f"{m['total_exited']:.0f} exited")
