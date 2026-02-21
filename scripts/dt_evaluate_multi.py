#!/usr/bin/env python
"""End-to-end Multi-Agent Decision Transformer evaluation on grid-4x4.

Collects per-agent trajectories from baseline controllers on the PettingZoo
multi-agent env, trains DT variants (Expert and Mixed) with parameter sharing,
then evaluates all policies on the same scenario.

Two DT variants:
- DT-Expert: trained on expert controllers only (MaxPressure + GreenWave)
- DT-Mixed: trained on all 6 controllers with RTG conditioning

Usage::

    python scripts/dt_evaluate_multi.py
    python scripts/dt_evaluate_multi.py --device cuda --epochs 10 --eval-episodes 10

"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import lightsim
from lightsim.core.signal import (
    FixedTimeController,
    GreenWaveController,
    MaxPressureController,
    SOTLController,
    SignalState,
    WebsterController,
)
from lightsim.dt.dataset import (
    Trajectory,
    collect_multi_agent_trajectories,
    pad_obs,
)
from lightsim.dt.model import DTConfig
from lightsim.dt.train import get_device, train_dt
from lightsim.dt.controller import MultiAgentDTPolicy

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class EvalResult:
    policy: str
    avg_reward: float
    total_throughput: float
    avg_vehicles: float
    wall_time: float


# ------------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------------

def evaluate_baseline_multi(
    scenario: str,
    controller,
    policy_name: str,
    episodes: int = 5,
    max_steps: int = 720,
    pad_dim: int = 14,
    **scenario_kwargs,
) -> EvalResult:
    """Evaluate a SignalController on the PettingZoo multi-agent env."""
    rewards_all, throughputs, vehicles = [], [], []
    t0 = time.perf_counter()

    for ep in range(episodes):
        env = lightsim.parallel_env(
            scenario, max_steps=max_steps, **scenario_kwargs,
        )
        obs_dict, _ = env.reset(seed=ep + 100)
        agent_names = list(env.possible_agents)
        agent_to_node = {a: int(a.split("_")[1]) for a in agent_names}
        n_actions = env.action_space(agent_names[0]).n

        ep_rewards = {a: 0.0 for a in agent_names}
        while env.agents:
            actions = {}
            for agent in env.agents:
                node_id = agent_to_node[agent]
                sig_state = env.engine.signal_manager.states.get(
                    node_id, SignalState(),
                )
                action = controller.get_phase_index(
                    node_id, sig_state,
                    env.engine.net, env.engine.state.density,
                )
                actions[agent] = action % n_actions

            obs_dict, rews, terms, truncs, infos = env.step(actions)
            for agent, r in rews.items():
                ep_rewards[agent] += r

        # Mean reward across agents, normalised by episode length
        mean_rew = float(np.mean(list(ep_rewards.values()))) / max_steps
        rewards_all.append(mean_rew)

        metrics = infos.get(agent_names[0], {}) if infos else {}
        throughputs.append(metrics.get("total_exited", 0))
        vehicles.append(metrics.get("total_vehicles", 0))

    return EvalResult(
        policy=policy_name,
        avg_reward=float(np.mean(rewards_all)),
        total_throughput=float(np.mean(throughputs)),
        avg_vehicles=float(np.mean(vehicles)),
        wall_time=time.perf_counter() - t0,
    )


def evaluate_dt_multi(
    scenario: str,
    dt_policy: MultiAgentDTPolicy,
    policy_name: str,
    target_return: float,
    episodes: int = 5,
    max_steps: int = 720,
    **scenario_kwargs,
) -> EvalResult:
    """Evaluate a MultiAgentDTPolicy on the PettingZoo env."""
    rewards_all, throughputs, vehicles = [], [], []
    t0 = time.perf_counter()

    for ep in range(episodes):
        env = lightsim.parallel_env(
            scenario, max_steps=max_steps, **scenario_kwargs,
        )
        obs_dict, _ = env.reset(seed=ep + 100)
        dt_policy.reset(target_return=target_return)

        ep_rewards = {a: 0.0 for a in env.possible_agents}
        while env.agents:
            actions = dt_policy.predict(obs_dict, deterministic=True)
            obs_dict, rews, terms, truncs, infos = env.step(actions)
            dt_policy.update_rtg(rews)
            for agent, r in rews.items():
                ep_rewards[agent] += r

        mean_rew = float(np.mean(list(ep_rewards.values()))) / max_steps
        rewards_all.append(mean_rew)

        agent_names = list(env.possible_agents)
        metrics = infos.get(agent_names[0], {}) if infos else {}
        throughputs.append(metrics.get("total_exited", 0))
        vehicles.append(metrics.get("total_vehicles", 0))

    return EvalResult(
        policy=policy_name,
        avg_reward=float(np.mean(rewards_all)),
        total_throughput=float(np.mean(throughputs)),
        avg_vehicles=float(np.mean(vehicles)),
        wall_time=time.perf_counter() - t0,
    )


def print_results(results: list[EvalResult], scenario: str) -> None:
    header = (
        f"{'Policy':<20} {'Reward/step':>11} {'Throughput':>11} "
        f"{'Vehicles':>9} {'Time(s)':>8}"
    )
    print()
    print("=" * len(header))
    print(f"Multi-Agent DT Evaluation — {scenario}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.policy:<20} {r.avg_reward:>11.2f} {r.total_throughput:>11.1f} "
            f"{r.avg_vehicles:>9.1f} {r.wall_time:>8.2f}"
        )
    print("-" * len(header))


# ------------------------------------------------------------------
# GreenWave offset helper
# ------------------------------------------------------------------

def _build_greenwave_offsets(env) -> dict[int, float]:
    """Compute row-wise GreenWave offsets from node coordinates."""
    agent_names = list(env.possible_agents)
    agent_to_node = {a: int(a.split("_")[1]) for a in agent_names}

    node_coords = {}
    for a in agent_names:
        nid = agent_to_node[a]
        node = env.network.nodes.get(nid)
        if node is not None:
            node_coords[nid] = (node.x, node.y)

    # Group by row (same y)
    rows: dict[float, list[int]] = {}
    for nid, (x, y) in node_coords.items():
        rows.setdefault(y, []).append(nid)

    offsets: dict[int, float] = {}
    for y_val in sorted(rows.keys()):
        row_nodes = sorted(rows[y_val], key=lambda n: node_coords[n][0])
        if len(row_nodes) < 2:
            offsets[row_nodes[0]] = 0.0
            continue
        distances = [
            abs(node_coords[row_nodes[i]][0] - node_coords[row_nodes[i - 1]][0])
            for i in range(1, len(row_nodes))
        ]
        row_offsets = GreenWaveController.compute_offsets(
            row_nodes, distances, speed=13.89,
        )
        offsets.update(row_offsets)
    return offsets


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Decision Transformer evaluation on grid-4x4",
    )
    parser.add_argument("--scenario", default="grid-4x4-v0")
    parser.add_argument("--episodes-collect", type=int, default=20,
                        help="Trajectory episodes per controller")
    parser.add_argument("--max-steps", type=int, default=720)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--context-len", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-results", default=None)
    args = parser.parse_args()

    dev = get_device(args.device)
    print(f"Device: {dev}")

    # ------------------------------------------------------------------
    # Step 0: Discover environment
    # ------------------------------------------------------------------
    print(f"\n[0/5] Discovering multi-agent environment...")
    pz_env = lightsim.parallel_env(args.scenario, max_steps=10)
    agent_names = list(pz_env.possible_agents)
    obs_dims = {a: pz_env.observation_space(a).shape[0] for a in agent_names}
    pad_dim = max(obs_dims.values())
    n_agents = len(agent_names)

    dim_counts: dict[int, int] = {}
    for d in obs_dims.values():
        dim_counts[d] = dim_counts.get(d, 0) + 1
    print(f"  Agents: {n_agents}")
    print(f"  Obs dims: {dict(sorted(dim_counts.items()))}")
    print(f"  Pad dim: {pad_dim}")

    # Build GreenWave offsets for later use
    gw_offsets = _build_greenwave_offsets(pz_env)

    # ------------------------------------------------------------------
    # Step 1: Collect trajectories
    # ------------------------------------------------------------------
    print(f"\n[1/5] Collecting trajectories...")
    t0 = time.perf_counter()

    # Mixed dataset: all 6 controllers
    mixed_trajs = collect_multi_agent_trajectories(
        scenario=args.scenario,
        episodes_per_controller=args.episodes_collect,
        max_steps=args.max_steps,
        seed=args.seed,
        pad_obs_dim=pad_dim,
    )

    # Expert dataset: MaxPressure + GreenWave only
    expert_trajs = collect_multi_agent_trajectories(
        scenario=args.scenario,
        controllers={
            "MaxPressure": MaxPressureController(min_green=5.0),
            "GreenWave": GreenWaveController(offsets=gw_offsets),
        },
        episodes_per_controller=args.episodes_collect * 2,  # 40 eps each
        max_steps=args.max_steps,
        seed=args.seed + 1000,
        pad_obs_dim=pad_dim,
    )

    collect_time = time.perf_counter() - t0
    mixed_steps = sum(t.length for t in mixed_trajs)
    expert_steps = sum(t.length for t in expert_trajs)
    print(f"  Mixed:  {len(mixed_trajs)} trajs, {mixed_steps:,} steps")
    print(f"  Expert: {len(expert_trajs)} trajs, {expert_steps:,} steps")
    print(f"  Collection: {collect_time:.1f}s")

    # Per-controller stats for mixed (6 controllers × eps × 16 agents)
    trajs_per_ctrl = args.episodes_collect * n_agents
    ctrl_names = ["Random", "FixedTime", "MaxPressure", "SOTL", "Webster", "GreenWave"]
    for i, name in enumerate(ctrl_names):
        sub = mixed_trajs[i * trajs_per_ctrl : (i + 1) * trajs_per_ctrl]
        if sub:
            rets = [t.total_return for t in sub]
            print(f"    {name:>12}: mean_return={np.mean(rets):.1f}")

    # ------------------------------------------------------------------
    # Step 2: Train DT variants
    # ------------------------------------------------------------------
    print(f"\n[2/5] Training DT models ({args.epochs} epochs, dropout={args.dropout})...")

    obs_dim = pad_dim
    act_dim = int(max(
        max(t.actions.max() for t in mixed_trajs),
        max(t.actions.max() for t in expert_trajs),
    )) + 1

    config = DTConfig(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=64,
        n_layers=3,
        n_heads=4,
        ffn_dim=256,
        context_len=args.context_len,
        dropout=args.dropout,
    )
    print(f"  Config: obs_dim={obs_dim}, act_dim={act_dim}, "
          f"hidden={config.hidden_dim}, layers={config.n_layers}")

    # Variant A: Expert (MaxPressure + GreenWave)
    print("  DT-Expert (MaxPressure + GreenWave)...")
    model_expert, losses_e, rtg_expert = train_dt(
        expert_trajs, config=config, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, device=args.device,
    )
    print(f"    Final loss: {losses_e[-1]:.4f}")

    # Variant B: Mixed (all controllers, RTG-conditioned)
    print("  DT-Mixed (all 6 controllers, RTG-conditioned)...")
    model_mixed, losses_m, rtg_mixed = train_dt(
        mixed_trajs, config=config, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, device=args.device,
    )
    print(f"    Final loss: {losses_m[-1]:.4f}")

    # ------------------------------------------------------------------
    # Step 3: Evaluate baselines
    # ------------------------------------------------------------------
    print(f"\n[3/5] Evaluating baselines ({args.eval_episodes} episodes, "
          f"{args.max_steps} steps)...")
    results: list[EvalResult] = []

    baselines = [
        ("FixedTime", FixedTimeController()),
        ("Webster", WebsterController()),
        ("SOTL", SOTLController()),
        ("MaxPressure", MaxPressureController(min_green=5.0)),
        ("GreenWave", GreenWaveController(offsets=gw_offsets)),
    ]

    for name, ctrl in baselines:
        print(f"  {name}...", end=" ", flush=True)
        r = evaluate_baseline_multi(
            args.scenario, ctrl, name,
            episodes=args.eval_episodes, max_steps=args.max_steps,
        )
        results.append(r)
        print(f"reward={r.avg_reward:.2f}  throughput={r.total_throughput:.0f}")

    # ------------------------------------------------------------------
    # Step 4: Evaluate DT variants
    # ------------------------------------------------------------------
    print(f"\n[4/5] Evaluating Decision Transformer variants...")

    # DT-Expert
    expert_ret = float(np.mean([t.total_return for t in expert_trajs]))
    dt_expert = MultiAgentDTPolicy(
        model=model_expert,
        agent_names=agent_names,
        target_return=expert_ret,
        rtg_stats=rtg_expert,
        pad_obs_dim=pad_dim,
        device="cpu",
    )
    print(f"  DT-Expert (target_return={expert_ret:.1f})...", end=" ", flush=True)
    r = evaluate_dt_multi(
        args.scenario, dt_expert, "DT-Expert",
        target_return=expert_ret,
        episodes=args.eval_episodes, max_steps=args.max_steps,
    )
    results.append(r)
    print(f"reward={r.avg_reward:.2f}  throughput={r.total_throughput:.0f}")

    # DT-Mixed with high RTG (aim for expert-level)
    # Use MaxPressure block as reference for high return
    mp_start = 2 * trajs_per_ctrl  # index of MaxPressure block in mixed
    mp_end = 3 * trajs_per_ctrl
    mp_trajs = mixed_trajs[mp_start:mp_end]
    high_ret = float(np.mean([t.total_return for t in mp_trajs])) if mp_trajs else expert_ret

    dt_mixed_hi = MultiAgentDTPolicy(
        model=model_mixed,
        agent_names=agent_names,
        target_return=high_ret,
        rtg_stats=rtg_mixed,
        pad_obs_dim=pad_dim,
        device="cpu",
    )
    print(f"  DT-Mixed(hi) (target_return={high_ret:.1f})...", end=" ", flush=True)
    r = evaluate_dt_multi(
        args.scenario, dt_mixed_hi, "DT-Mixed(hi)",
        target_return=high_ret,
        episodes=args.eval_episodes, max_steps=args.max_steps,
    )
    results.append(r)
    print(f"reward={r.avg_reward:.2f}  throughput={r.total_throughput:.0f}")

    # DT-Mixed with low RTG (Random block)
    random_trajs = mixed_trajs[:trajs_per_ctrl]
    low_ret = float(np.mean([t.total_return for t in random_trajs]))

    dt_mixed_lo = MultiAgentDTPolicy(
        model=model_mixed,
        agent_names=agent_names,
        target_return=low_ret,
        rtg_stats=rtg_mixed,
        pad_obs_dim=pad_dim,
        device="cpu",
    )
    print(f"  DT-Mixed(lo) (target_return={low_ret:.1f})...", end=" ", flush=True)
    r = evaluate_dt_multi(
        args.scenario, dt_mixed_lo, "DT-Mixed(lo)",
        target_return=low_ret,
        episodes=args.eval_episodes, max_steps=args.max_steps,
    )
    results.append(r)
    print(f"reward={r.avg_reward:.2f}  throughput={r.total_throughput:.0f}")

    # ------------------------------------------------------------------
    # Step 5: Print and save
    # ------------------------------------------------------------------
    print_results(results, args.scenario)

    save_path = args.save_results or str(RESULTS_DIR / "multi_agent_dt.json")
    out = {
        "scenario": args.scenario,
        "num_agents": n_agents,
        "pad_obs_dim": pad_dim,
        "obs_dimensions": dict(sorted(dim_counts.items())),
        "collection": {
            "mixed_trajectories": len(mixed_trajs),
            "mixed_steps": mixed_steps,
            "expert_trajectories": len(expert_trajs),
            "expert_steps": expert_steps,
            "collection_time_s": round(collect_time, 1),
        },
        "training": {
            "config": {
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "hidden_dim": config.hidden_dim,
                "n_layers": config.n_layers,
                "context_len": config.context_len,
                "dropout": config.dropout,
            },
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "expert_final_loss": losses_e[-1],
            "mixed_final_loss": losses_m[-1],
        },
        "results": [asdict(r) for r in results],
    }
    Path(save_path).write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
