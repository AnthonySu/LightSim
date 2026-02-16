"""Validate the full triangular fundamental diagram (free-flow + congested).

Tests both branches of the triangular FD:
  Free-flow: q = vf * k  (demand-driven, k < k_crit)
  Congested: q = w * (kj - k)  (initialized at high density, k > k_crit)
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, r"C:/Users/admin/Projects/lightsim")

from lightsim.core.engine import SimulationEngine
from lightsim.core.network import Network
from lightsim.core.demand import DemandProfile
from lightsim.core.types import FLOAT, LinkID, NodeID, NodeType


def run_fd_validation():
    vf = 13.89
    w = 5.56
    kj = 0.15
    capacity = 0.5
    lanes = 1
    dt = 1.0
    n_cells = 10
    link_length = vf * dt * n_cells

    k_crit = capacity / vf

    print("Parameters:")
    print(f"  vf={vf}, w={w}, kj={kj}, Q={capacity}, k_crit={k_crit:.6f}")
    print()

    all_densities = []
    all_flows = []

    # === FREE-FLOW BRANCH ===
    n_ff = 60
    demand_rates_ff = np.linspace(0, capacity * lanes, n_ff, endpoint=True)

    print("--- Free-flow branch ---")
    for rate in demand_rates_ff:
        net = Network()
        net.add_node(NodeID(0), NodeType.ORIGIN)
        net.add_node(NodeID(1), NodeType.DESTINATION)
        net.add_link(
            LinkID(0), from_node=NodeID(0), to_node=NodeID(1),
            length=link_length, lanes=lanes, n_cells=n_cells,
            free_flow_speed=vf, wave_speed=w, jam_density=kj, capacity=capacity,
        )
        profile = DemandProfile(
            link_id=LinkID(0), time_points=[0.0], flow_rates=[float(rate)],
        )
        engine = SimulationEngine(network=net, dt=dt, demand_profiles=[profile])
        engine.reset()
        for _ in range(300):
            engine.step()

        mid_cells = list(range(3, 7))
        k = float(engine.state.density[mid_cells].mean())
        q = float(min(vf * k * lanes, capacity * lanes))
        all_densities.append(k)
        all_flows.append(q)

    print(f"  {n_ff} points, k in [{min(all_densities):.6f}, {max(all_densities):.6f}]")

    # === CONGESTED BRANCH ===
    n_cong = 60
    target_densities = np.linspace(k_crit, kj, n_cong, endpoint=True)

    print("--- Congested branch ---")
    for k_target in target_densities:
        net = Network()
        net.add_node(NodeID(0), NodeType.ORIGIN)
        net.add_node(NodeID(1), NodeType.DESTINATION)
        net.add_link(
            LinkID(0), from_node=NodeID(0), to_node=NodeID(1),
            length=link_length, lanes=lanes, n_cells=n_cells,
            free_flow_speed=vf, wave_speed=w, jam_density=kj, capacity=capacity,
        )
        engine = SimulationEngine(network=net, dt=dt, demand_profiles=[])
        engine.reset()

        # Set all cells to target density
        engine.state.density[:] = k_target

        # Take one step to let CTM compute flows
        engine.step()

        # Interior cells (2-7) should maintain density (uniform => flow in = flow out)
        interior = list(range(2, 8))
        k_measured = float(engine.state.density[interior].mean())

        # Flow from CTM at this density
        S = min(vf * k_measured, capacity) * lanes
        R = min(capacity, w * (kj - k_measured)) * lanes
        q_measured = max(min(S, R), 0.0)

        all_densities.append(k_measured)
        all_flows.append(q_measured)

    densities = np.array(all_densities)
    flows = np.array(all_flows)

    n_congested = (densities > k_crit + 1e-9).sum()
    n_freeflow = (densities <= k_crit + 1e-9).sum()
    print(f"  {n_cong} points, k in [{densities[n_ff:].min():.6f}, {densities[n_ff:].max():.6f}]")
    print(f"Total: {len(densities)} points")
    print(f"Free-flow (k <= k_crit): {n_freeflow}")
    print(f"Congested (k > k_crit):  {n_congested}")

    # === THEORY CURVE ===
    k_theory_ff = np.linspace(0, k_crit, 100)
    q_theory_ff = vf * k_theory_ff * lanes
    k_theory_cg = np.linspace(k_crit, kj, 100)
    q_theory_cg = w * (kj - k_theory_cg) * lanes
    k_theory = np.concatenate([k_theory_ff, k_theory_cg])
    q_theory = np.concatenate([q_theory_ff, q_theory_cg])

    # === R-SQUARED ===
    q_predicted = np.zeros_like(densities)
    for i, k in enumerate(densities):
        if k <= k_crit:
            q_predicted[i] = vf * k * lanes
        else:
            q_predicted[i] = w * (kj - k) * lanes
        q_predicted[i] = max(min(q_predicted[i], capacity * lanes), 0.0)

    ss_res = np.sum((flows - q_predicted) ** 2)
    ss_tot = np.sum((flows - flows.mean()) ** 2)
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0

    print("")
    print(f"R-squared = {r_squared:.10f}")

    # === SAVE ===
    results = {
        "densities": densities.tolist(),
        "flows": flows.tolist(),
        "k_theory": k_theory.tolist(),
        "q_theory": q_theory.tolist(),
        "r_squared": r_squared,
    }
    output_path = r"C:/Users/admin/Projects/lightsim/results/fundamental_diagram_full.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")
    print(f"Keys: {list(results.keys())}")


if __name__ == "__main__":
    run_fd_validation()
