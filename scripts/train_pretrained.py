"""Train and save pretrained RL model checkpoints.

Trains DQN and PPO on single-intersection-v0 with both queue and pressure
rewards, saving checkpoints to weights/ for distribution with the repo.

Usage::
    python scripts/train_pretrained.py
    python scripts/train_pretrained.py --timesteps 200000
    python scripts/train_pretrained.py --algo dqn  # train only DQN
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import lightsim
from lightsim.core.engine import SimulationEngine
from lightsim.core.signal import FixedTimeController, MaxPressureController
from lightsim.benchmarks.scenarios import get_scenario

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor

WEIGHTS_DIR = PROJECT_ROOT / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

MODELS = [
    {"algo": "DQN", "reward": "queue", "name": "dqn_single_intersection"},
    {"algo": "PPO", "reward": "queue", "name": "ppo_single_intersection"},
    {"algo": "DQN", "reward": "pressure", "name": "dqn_single_intersection_pressure"},
    {"algo": "PPO", "reward": "pressure", "name": "ppo_single_intersection_pressure"},
]


def make_env(reward_fn: str = "queue", max_steps: int = 720):
    """Create env with max_steps=720 (720 decision steps * 5 = 3600 sim seconds = 1 hour)."""
    env = lightsim.make("single-intersection-v0", reward_fn=reward_fn, max_steps=max_steps)
    return Monitor(env)


def evaluate(model, reward_fn: str, episodes: int = 10) -> dict:
    """Evaluate a trained model and return metrics."""
    env = lightsim.make("single-intersection-v0", reward_fn=reward_fn, max_steps=720)
    rewards, throughputs, delays = [], [], []

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
        delays.append(metrics.get("avg_density", 0.0))
    env.close()

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "reward_per_step": float(np.mean(rewards)) / 720,
        "mean_throughput": float(np.mean(throughputs)),
        "mean_delay": float(np.mean(delays)),
    }


def train_and_save(spec: dict, timesteps: int, seed: int = 42) -> dict:
    """Train one model and save it."""
    algo_name = spec["algo"]
    reward_fn = spec["reward"]
    name = spec["name"]

    print(f"\n{'=' * 60}")
    print(f"Training {algo_name} | reward={reward_fn} | {timesteps} steps")
    print(f"{'=' * 60}")

    env = make_env(reward_fn=reward_fn)
    AlgoCls = DQN if algo_name == "DQN" else PPO

    kwargs = {"seed": seed, "verbose": 1}
    if algo_name == "DQN":
        kwargs.update({"learning_starts": 1000, "batch_size": 64})

    model = AlgoCls("MlpPolicy", env, **kwargs)

    t0 = time.time()
    model.learn(total_timesteps=timesteps)
    train_time = time.time() - t0

    save_path = WEIGHTS_DIR / name
    model.save(str(save_path))
    print(f"Saved to {save_path}.zip ({train_time:.1f}s)")

    # Evaluate
    print("Evaluating...")
    metrics = evaluate(model, reward_fn=reward_fn)
    metrics["train_time_s"] = train_time
    metrics["timesteps"] = timesteps
    metrics["algo"] = algo_name
    metrics["reward_fn"] = reward_fn
    metrics["seed"] = seed

    print(f"  Reward: {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")
    print(f"  Throughput: {metrics['mean_throughput']:.0f}")
    print(f"  Delay: {metrics['mean_delay']:.2f}s")

    env.close()
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train pretrained LightSim RL checkpoints")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--algo", type=str, default=None, help="Train only this algo (dqn/ppo)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    models_to_train = MODELS
    if args.algo:
        models_to_train = [m for m in MODELS if m["algo"].lower() == args.algo.lower()]

    all_metrics = {}
    for spec in models_to_train:
        metrics = train_and_save(spec, timesteps=args.timesteps, seed=args.seed)
        all_metrics[spec["name"]] = metrics

    # Save evaluation results alongside weights
    results_path = WEIGHTS_DIR / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll results saved to {results_path}")

    # Update weights README with actual performance
    update_readme(all_metrics)
    print("Updated weights/README.md with performance numbers")


def update_readme(metrics: dict):
    """Update the weights README with actual evaluation results."""
    rows = []
    for name, m in metrics.items():
        rows.append(
            f"| `{name}.zip` | {m['algo']} | single-intersection-v0 "
            f"| {m['reward_fn']} | {m['timesteps']//1000}k "
            f"| reward={m['mean_reward']:.1f}, tput={m['mean_throughput']:.0f} |"
        )

    readme = f"""# Pretrained Model Weights

Pretrained Stable-Baselines3 checkpoints for reproducing paper results.

## Available Models

| File | Algorithm | Scenario | Reward | Steps | Performance |
|------|-----------|----------|--------|-------|-------------|
{chr(10).join(rows)}

## Usage

```python
import lightsim
from lightsim.pretrained import load_pretrained

# Load and evaluate a pretrained DQN
env = lightsim.make("single-intersection-v0")
model = load_pretrained("dqn_single_intersection", env=env)

obs, info = env.reset()
for _ in range(3600):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
```

## Reproducing

To retrain these models from scratch:
```bash
python scripts/train_pretrained.py
python scripts/train_pretrained.py --timesteps 200000  # more training
```
"""
    with open(WEIGHTS_DIR / "README.md", "w") as f:
        f.write(readme)


if __name__ == "__main__":
    main()
