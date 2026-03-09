"""Create a custom network from a dict, validate it, and simulate.

Shows how network.validate() catches common mistakes before simulation.
"""

from lightsim.core.demand import DemandProfile
from lightsim.core.engine import SimulationEngine
from lightsim.core.signal import MaxPressureController
from lightsim.core.types import LinkID
from lightsim.networks.from_dict import from_dict

# --- Step 1: A network with intentional mistakes ---
broken_spec = {
    "nodes": [
        {"id": 0, "type": "origin",     "x": 0,   "y": 0},
        {"id": 1, "type": "signalized",  "x": 300, "y": 0},
        {"id": 2, "type": "destination", "x": 600, "y": 0},
        {"id": 3, "type": "origin",      "x": 300, "y": 200},
        # Mistake: no destination for side-street traffic
    ],
    "links": [
        {"id": 0, "from": 0, "to": 1, "length": 300, "lanes": 2, "n_cells": 3},
        {"id": 1, "from": 1, "to": 2, "length": 300, "lanes": 2, "n_cells": 3},
        {"id": 2, "from": 3, "to": 1, "length": 200, "lanes": 1, "n_cells": 2},
    ],
    "movements": [
        {"from_link": 0, "to_link": 1, "node": 1, "turn_type": "through", "turn_ratio": 1.0},
        {"from_link": 2, "to_link": 1, "node": 1, "turn_type": "right",   "turn_ratio": 1.0},
    ],
    "phases": [],  # Mistake: no phases for signalized node 1
}

network = from_dict(broken_spec)
errors = network.validate()
print("=== Validation errors (broken network) ===")
for err in errors:
    print(f"  - {err}")

# --- Step 2: Fixed network ---
fixed_spec = {
    "nodes": [
        {"id": 0, "type": "origin",      "x": 0,   "y": 0},
        {"id": 1, "type": "signalized",   "x": 300, "y": 0},
        {"id": 2, "type": "destination",  "x": 600, "y": 0},
        {"id": 3, "type": "origin",       "x": 300, "y": 200},
        {"id": 4, "type": "destination",  "x": 300, "y": -200},  # added
    ],
    "links": [
        {"id": 0, "from": 0, "to": 1, "length": 300, "lanes": 2, "n_cells": 3},
        {"id": 1, "from": 1, "to": 2, "length": 300, "lanes": 2, "n_cells": 3},
        {"id": 2, "from": 3, "to": 1, "length": 200, "lanes": 1, "n_cells": 2},
        {"id": 3, "from": 1, "to": 4, "length": 200, "lanes": 1, "n_cells": 2},  # added
    ],
    "movements": [
        {"from_link": 0, "to_link": 1, "node": 1, "turn_type": "through", "turn_ratio": 0.7},
        {"from_link": 0, "to_link": 3, "node": 1, "turn_type": "left",    "turn_ratio": 0.3},
        {"from_link": 2, "to_link": 1, "node": 1, "turn_type": "right",   "turn_ratio": 0.5},
        {"from_link": 2, "to_link": 3, "node": 1, "turn_type": "through", "turn_ratio": 0.5},
    ],
    "phases": [
        {"node": 1, "movements": [0, 1]},  # EB phase
        {"node": 1, "movements": [2, 3]},  # NB phase
    ],
}

network = from_dict(fixed_spec)
errors = network.validate()
print(f"\n=== Fixed network: {'PASS' if not errors else 'FAIL'} ===")

# --- Step 3: Simulate ---
demand = [
    DemandProfile(LinkID(0), [0.0], [0.3]),
    DemandProfile(LinkID(2), [0.0], [0.1]),
]
engine = SimulationEngine(
    network=network, dt=1.0,
    controller=MaxPressureController(min_green=5.0),
    demand_profiles=demand,
)
engine.reset(seed=42)

print(f"\n{'Time':>6}  {'Vehicles':>8}  {'Entered':>7}  {'Exited':>6}")
print("-" * 35)
for step in range(600):
    engine.step()
    if (step + 1) % 200 == 0:
        m = engine.get_network_metrics()
        print(f"{m['time']:>6.0f}  {m['total_vehicles']:>8.1f}  "
              f"{m['total_entered']:>7.0f}  {m['total_exited']:>6.0f}")
