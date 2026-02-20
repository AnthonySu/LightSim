#!/usr/bin/env python
"""End-to-end Decision Transformer evaluation.

Collects trajectories from baseline controllers, trains DT variants on GPU,
then evaluates DT vs all baselines on the same scenario.

Two DT variants are trained:
- DT-Expert: behavioral cloning on expert controllers only (MaxPressure + SOTL)
- DT-Mixed: trained on all controllers with RTG conditioning

Usage::

    python scripts/dt_evaluate.py
    python scripts/dt_evaluate.py --device cuda --epochs 5 --episodes 30
    python scripts/dt_evaluate.py --scenario grid-4x4-v0

"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import lightsim
from lightsim.core.engine import SimulationEngine
from lightsim.core.signal import (
    FixedTimeController,
    MaxPressureController,
    SOTLController,
    SignalState,
    WebsterController,
)
from lightsim.core.types import NodeType
from lightsim.dt.dataset import (
    Trajectory,
    _RandomController,
    collect_trajectories,
    save_trajectories,
)
from lightsim.dt.model import DTConfig
from lightsim.dt.train import get_device, save_dt_model, train_dt
from lightsim.dt.controller import DTPolicy
from lightsim.utils.metrics import compute_link_delay


@dataclass
class EvalResult:
    policy: str
    avg_reward: float
    total_throughput: float
    avg_delay: float
    avg_vehicles: float
    wall_time: float


def evaluate_via_env(
    scenario: str,
    controller_or_policy,
    policy_name: str,
    episodes: int = 5,
    max_steps: int = 720,
    is_dt: bool = False,
    target_return: float | None = None,
) -> EvalResult:
    """Evaluate a controller or DT policy through the Gym env.

    All controllers are evaluated through the same env interface for
    fair, apples-to-apples comparison.
    """
    rewards, throughputs, delays, vehicles = [], [], [], []
    t0 = time.perf_counter()

    for ep in range(episodes):
        env = lightsim.make(scenario, max_steps=max_steps)
        obs, info = env.reset(seed=ep + 100)
        agent_node = env.agent_node

        if is_dt:
            controller_or_policy.reset(target_return=target_return)

        ep_reward = 0.0
        done = False
        while not done:
            if is_dt:
                action = controller_or_policy.predict(obs, deterministic=True)
            else:
                sig_state = env.engine.signal_manager.states.get(
                    agent_node, SignalState(),
                )
                action = controller_or_policy.get_phase_index(
                    agent_node, sig_state,
                    env.engine.net, env.engine.state.density,
                ) % env.action_space.n

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if is_dt:
                controller_or_policy.update_rtg(reward)
            done = terminated or truncated

        rewards.append(ep_reward / max_steps)
        throughputs.append(info.get("total_exited", 0))
        vehicles.append(info.get("total_vehicles", 0))

        total_delay, n_links = 0.0, 0
        for link in env.network.links.values():
            to_node = env.network.nodes.get(link.to_node)
            if to_node and to_node.node_type == NodeType.SIGNALIZED:
                total_delay += compute_link_delay(env.engine, link.link_id)
                n_links += 1
        delays.append(total_delay / max(n_links, 1))
        env.close()

    return EvalResult(
        policy=policy_name,
        avg_reward=float(np.mean(rewards)),
        total_throughput=float(np.mean(throughputs)),
        avg_delay=float(np.mean(delays)),
        avg_vehicles=float(np.mean(vehicles)),
        wall_time=time.perf_counter() - t0,
    )


def print_results(results: list[EvalResult], scenario: str) -> None:
    header = (
        f"{'Policy':<20} {'Reward/step':>11} {'Throughput':>11} "
        f"{'Delay':>8} {'Vehicles':>9} {'Time(s)':>8}"
    )
    print()
    print("=" * len(header))
    print(f"Decision Transformer Evaluation — {scenario}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.policy:<20} {r.avg_reward:>11.2f} {r.total_throughput:>11.1f} "
            f"{r.avg_delay:>8.2f} {r.avg_vehicles:>9.1f} {r.wall_time:>8.2f}"
        )
    print("-" * len(header))


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end Decision Transformer evaluation",
    )
    parser.add_argument("--scenario", default="single-intersection-v0")
    parser.add_argument("--episodes-collect", type=int, default=20,
                        help="Trajectory episodes per controller")
    parser.add_argument("--max-steps", type=int, default=720,
                        help="Episode length in RL steps")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs (fewer = less overfitting)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-len", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", default="auto",
                        help="Training device: auto, cpu, cuda")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-model", default=None,
                        help="Save expert DT model to path")
    parser.add_argument("--save-data", default=None,
                        help="Save trajectories to .npz")
    parser.add_argument("--save-results", default=None,
                        help="Save results to .json")
    args = parser.parse_args()

    dev = get_device(args.device)
    print(f"Device: {dev}")

    # ------------------------------------------------------------------
    # Step 1: Collect trajectories
    # ------------------------------------------------------------------
    print(f"\n[1/4] Collecting trajectories...")
    t0 = time.perf_counter()

    # Mixed dataset (all controllers)
    mixed_trajs = collect_trajectories(
        scenario=args.scenario,
        episodes_per_controller=args.episodes_collect,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    # Expert dataset (good controllers only)
    expert_trajs = collect_trajectories(
        scenario=args.scenario,
        controllers={
            "MaxPressure": MaxPressureController(min_green=5.0),
            "SOTL": SOTLController(),
        },
        episodes_per_controller=args.episodes_collect * 2,  # more expert data
        max_steps=args.max_steps,
        seed=args.seed + 1000,
    )

    collect_time = time.perf_counter() - t0
    print(f"  Mixed: {len(mixed_trajs)} trajs, "
          f"{sum(t.length for t in mixed_trajs):,} steps")
    print(f"  Expert: {len(expert_trajs)} trajs, "
          f"{sum(t.length for t in expert_trajs):,} steps")
    print(f"  Collection: {collect_time:.1f}s")

    # Per-controller stats for mixed
    eps = args.episodes_collect
    for i, name in enumerate(["Random", "FixedTime", "MaxPressure", "SOTL"]):
        sub = mixed_trajs[i * eps:(i + 1) * eps]
        rets = [t.total_return for t in sub]
        print(f"    {name:>12}: mean_return={np.mean(rets):.1f}")

    if args.save_data:
        save_trajectories(expert_trajs, args.save_data)
        print(f"  Saved expert trajectories to {args.save_data}")

    # ------------------------------------------------------------------
    # Step 2: Train DT variants
    # ------------------------------------------------------------------
    print(f"\n[2/4] Training DT models ({args.epochs} epochs, dropout={args.dropout})...")

    obs_dim = expert_trajs[0].observations.shape[1]
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

    # Variant A: Expert behavioral cloning
    print("  DT-Expert (MaxPressure + SOTL only)...")
    model_expert, losses_e, rtg_expert = train_dt(
        expert_trajs, config=config, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, device=args.device,
    )
    print(f"    Loss: {losses_e[-1]:.4f}")

    # Variant B: Mixed data with RTG conditioning
    print("  DT-Mixed (all controllers, RTG-conditioned)...")
    model_mixed, losses_m, rtg_mixed = train_dt(
        mixed_trajs, config=config, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, device=args.device,
    )
    print(f"    Loss: {losses_m[-1]:.4f}")

    if args.save_model:
        save_dt_model(model_expert, args.save_model, rtg_stats=rtg_expert)
        print(f"  Saved expert model to {args.save_model}")

    # ------------------------------------------------------------------
    # Step 3: Evaluate baselines
    # ------------------------------------------------------------------
    print(f"\n[3/4] Evaluating baselines ({args.eval_episodes} episodes, "
          f"{args.max_steps} steps)...")
    results: list[EvalResult] = []

    baselines = [
        ("FixedTime", FixedTimeController()),
        ("Webster", WebsterController()),
        ("SOTL", SOTLController()),
        ("MaxPressure", MaxPressureController(min_green=5.0)),
    ]

    for name, ctrl in baselines:
        print(f"  {name}...", end=" ", flush=True)
        r = evaluate_via_env(
            args.scenario, ctrl, name,
            episodes=args.eval_episodes, max_steps=args.max_steps,
        )
        results.append(r)
        print(f"reward={r.avg_reward:.2f}  throughput={r.total_throughput:.0f}")

    # ------------------------------------------------------------------
    # Step 4: Evaluate DT variants
    # ------------------------------------------------------------------
    print(f"\n[4/4] Evaluating Decision Transformer variants...")

    # Expert DT
    expert_ret = float(np.mean([t.total_return for t in expert_trajs]))
    policy_e = DTPolicy(model_expert, target_return=expert_ret,
                        rtg_stats=rtg_expert, device="cpu")
    print(f"  DT-Expert...", end=" ", flush=True)
    r = evaluate_via_env(
        args.scenario, policy_e, "DT-Expert",
        episodes=args.eval_episodes, max_steps=args.max_steps,
        is_dt=True, target_return=expert_ret,
    )
    results.append(r)
    print(f"reward={r.avg_reward:.2f}  throughput={r.total_throughput:.0f}")

    # Mixed DT with high RTG (aim for expert-level performance)
    mp_trajs = mixed_trajs[2 * eps:3 * eps]  # MaxPressure block
    high_ret = float(np.mean([t.total_return for t in mp_trajs]))
    policy_m = DTPolicy(model_mixed, target_return=high_ret,
                        rtg_stats=rtg_mixed, device="cpu")
    print(f"  DT-Mixed (high RTG)...", end=" ", flush=True)
    r = evaluate_via_env(
        args.scenario, policy_m, "DT-Mixed(hi)",
        episodes=args.eval_episodes, max_steps=args.max_steps,
        is_dt=True, target_return=high_ret,
    )
    results.append(r)
    print(f"reward={r.avg_reward:.2f}  throughput={r.total_throughput:.0f}")

    # Mixed DT with low RTG
    low_ret = float(np.mean([t.total_return for t in mixed_trajs[:eps]]))  # Random
    policy_m2 = DTPolicy(model_mixed, target_return=low_ret,
                         rtg_stats=rtg_mixed, device="cpu")
    print(f"  DT-Mixed (low RTG)...", end=" ", flush=True)
    r = evaluate_via_env(
        args.scenario, policy_m2, "DT-Mixed(lo)",
        episodes=args.eval_episodes, max_steps=args.max_steps,
        is_dt=True, target_return=low_ret,
    )
    results.append(r)
    print(f"reward={r.avg_reward:.2f}  throughput={r.total_throughput:.0f}")

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print_results(results, args.scenario)

    if args.save_results:
        out = [asdict(r) for r in results]
        Path(args.save_results).write_text(json.dumps(out, indent=2))
        print(f"\nResults saved to {args.save_results}")


if __name__ == "__main__":
    main()
