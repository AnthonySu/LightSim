"""Evaluate pretrained model checkpoints and reproduce paper Table 3 results.

Usage::
    python scripts/evaluate_pretrained.py
    python scripts/evaluate_pretrained.py --model dqn_single_intersection
    python scripts/evaluate_pretrained.py --episodes 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import lightsim
from lightsim.pretrained import list_pretrained, load_pretrained
from lightsim.core.engine import SimulationEngine
from lightsim.core.signal import FixedTimeController, MaxPressureController
from lightsim.benchmarks.scenarios import get_scenario


def evaluate_rl_model(name: str, episodes: int = 10) -> dict:
    """Evaluate a pretrained RL model."""
    # Detect reward function from name
    reward_fn = "pressure" if "pressure" in name else "queue"
    env = lightsim.make("single-intersection-v0", reward_fn=reward_fn, max_steps=720)
    model = load_pretrained(name, env=env)

    rewards, throughputs = [], []
    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
        metrics = env.unwrapped.engine.get_network_metrics()
        throughputs.append(metrics.get("total_exited", 0))
    env.close()

    return {
        "name": name,
        "reward_fn": reward_fn,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "per_step_reward": float(np.mean(rewards)) / 720,
        "mean_throughput": float(np.mean(throughputs)),
    }


def evaluate_baseline(controller_name: str, episodes: int = 10) -> dict:
    """Evaluate a classical controller baseline."""
    network, demand = get_scenario("single-intersection-v0")()
    controllers = {
        "FixedTime": FixedTimeController(),
        "MaxPressure": MaxPressureController(min_green=15.0),
    }
    ctrl = controllers[controller_name]

    throughputs, rewards = [], []
    for _ in range(episodes):
        engine = SimulationEngine(
            network=network, dt=1.0, controller=ctrl, demand_profiles=demand
        )
        engine.reset()
        total_reward = 0.0
        for step in range(3600):
            engine.step()
            metrics = engine.get_network_metrics()
            total_reward += -metrics.get("total_vehicles", 0)
        final_metrics = engine.get_network_metrics()
        throughputs.append(final_metrics.get("total_exited", 0))
        rewards.append(total_reward)

    return {
        "name": controller_name,
        "reward_fn": "queue",
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "per_step_reward": float(np.mean(rewards)) / 720,
        "mean_throughput": float(np.mean(throughputs)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained LightSim models")
    parser.add_argument("--model", type=str, default=None, help="Specific model to evaluate")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--no-baselines", action="store_true", help="Skip baseline evaluation")
    args = parser.parse_args()

    available = list_pretrained()
    if not available:
        print("No pretrained models found in weights/. Run train_pretrained.py first.")
        return

    models = [args.model] if args.model else available
    results = []

    # Evaluate baselines
    if not args.no_baselines:
        print("Evaluating baselines...")
        for ctrl_name in ["FixedTime", "MaxPressure"]:
            r = evaluate_baseline(ctrl_name, episodes=args.episodes)
            results.append(r)
            print(f"  {ctrl_name:25s}  reward/step={r['per_step_reward']:8.2f}  "
                  f"tput={r['mean_throughput']:8.0f}")

    # Evaluate pretrained models
    print("\nEvaluating pretrained models...")
    for name in models:
        if name not in available:
            print(f"  WARNING: {name} not found, skipping")
            continue
        r = evaluate_rl_model(name, episodes=args.episodes)
        results.append(r)
        print(f"  {name:25s}  reward/step={r['per_step_reward']:8.2f}  "
              f"tput={r['mean_throughput']:8.0f}")

    # Summary table
    sorted_results = sorted(results, key=lambda x: x["per_step_reward"], reverse=True)
    print(f"\n{'='*70}")
    print(f"{'Controller':35s} {'Reward/step':>12s} {'Throughput':>12s}")
    print(f"{'-'*70}")
    for r in sorted_results:
        label = r["name"]
        if r["reward_fn"] != "queue":
            label += f" [{r['reward_fn']}]"
        std_str = f" +/- {r['std_reward']/720:.2f}" if r["std_reward"] > 0 else ""
        print(f"{label:35s} {r['per_step_reward']:8.2f}{std_str:>12s} "
              f"{r['mean_throughput']:8.0f}")

    # Markdown table
    print(f"\n### Markdown Table\n")
    print("| Controller | Reward/step | Throughput |")
    print("|------------|-------------|------------|")
    for r in sorted_results:
        label = r["name"]
        if r["reward_fn"] != "queue":
            label += f" [{r['reward_fn']}]"
        print(f"| {label} | {r['per_step_reward']:.2f} | {r['mean_throughput']:,.0f} |")


if __name__ == "__main__":
    main()
