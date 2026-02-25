"""Train pretrained RL checkpoints and save to weights/ for distribution.

Trains single-agent DQN/PPO on single-intersection-v0 with queue and pressure
rewards, plus multi-agent DQN on grid-4x4 with parameter sharing.

Usage::
    python scripts/train_pretrained.py                    # train all
    python scripts/train_pretrained.py --single-only      # single-agent only
    python scripts/train_pretrained.py --multi-only       # multi-agent only
    python scripts/train_pretrained.py --timesteps 200000 # more training
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import lightsim

WEIGHTS_DIR = PROJECT_ROOT / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

MAX_OBS_DIM = 14

SINGLE_AGENT_MODELS = [
    {"algo": "DQN", "reward": "queue", "name": "dqn_single_intersection"},
    {"algo": "PPO", "reward": "queue", "name": "ppo_single_intersection"},
    {"algo": "DQN", "reward": "pressure", "name": "dqn_single_intersection_pressure"},
    {"algo": "PPO", "reward": "pressure", "name": "ppo_single_intersection_pressure"},
]


# ---------------------------------------------------------------------------
# Multi-agent wrapper (parameter sharing across 16 heterogeneous agents)
# ---------------------------------------------------------------------------

def pad_obs(obs, target_dim=MAX_OBS_DIM):
    """Pad observation to target_dim with trailing zeros."""
    if len(obs) >= target_dim:
        return obs[:target_dim].astype(np.float32)
    padded = np.zeros(target_dim, dtype=np.float32)
    padded[: len(obs)] = obs
    return padded


class SharedPolicyMultiAgentWrapper(gym.Env):
    """Wraps PettingZoo grid-4x4 as single-agent Gym env with parameter sharing.

    Each env step cycles through all 16 agents: the policy sees one padded
    observation at a time, outputs one action, and only after all 16 actions
    are collected does the underlying PettingZoo env advance.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps=720):
        super().__init__()
        self.max_steps = max_steps
        self.pz_env = lightsim.parallel_env("grid-4x4-v0", max_steps=max_steps)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(MAX_OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)
        self._agents: list = []
        self._agent_idx = 0
        self._actions: dict = {}
        self._obs: dict = {}
        self._done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._obs, _ = self.pz_env.reset(seed=seed)
        self._agents = list(self.pz_env.agents)
        self._agent_idx = 0
        self._actions = {}
        self._done = False
        return pad_obs(self._obs[self._agents[0]]), {}

    def step(self, action):
        if self._done:
            return pad_obs(np.zeros(10)), 0.0, False, True, {}
        agent_name = self._agents[self._agent_idx]
        self._actions[agent_name] = int(action)
        self._agent_idx += 1
        if self._agent_idx < len(self._agents):
            return pad_obs(self._obs[self._agents[self._agent_idx]]), 0.0, False, False, {}
        self._obs, rewards, terms, truncs, infos = self.pz_env.step(self._actions)
        mean_reward = float(np.mean(list(rewards.values())))
        if not self.pz_env.agents:
            self._done = True
            return pad_obs(np.zeros(10)), mean_reward, False, True, {}
        self._agents = list(self.pz_env.agents)
        self._agent_idx = 0
        self._actions = {}
        return pad_obs(self._obs[self._agents[0]]), mean_reward, False, False, {}


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_single_agent(spec: dict, timesteps: int, seed: int = 42) -> dict:
    """Train one single-agent model and save to weights/."""
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.monitor import Monitor

    algo_name = spec["algo"]
    reward_fn = spec["reward"]
    name = spec["name"]

    print(f"\n  Training {name} ({algo_name}, reward={reward_fn}, {timesteps:,} steps)...")
    env = Monitor(lightsim.make("single-intersection-v0", reward_fn=reward_fn, max_steps=720))

    cls = {"DQN": DQN, "PPO": PPO}[algo_name]
    kwargs = {"seed": seed, "verbose": 0}
    if algo_name == "DQN":
        kwargs.update(
            learning_rate=1e-3,
            buffer_size=50_000,
            learning_starts=1_000,
            batch_size=64,
            gamma=0.99,
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            target_update_interval=500,
        )
    else:
        kwargs.update(
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
        )

    model = cls("MlpPolicy", env, **kwargs)
    t0 = time.perf_counter()
    model.learn(total_timesteps=timesteps)
    train_time = time.perf_counter() - t0

    save_path = WEIGHTS_DIR / name
    model.save(str(save_path))
    print(f"    Saved to {save_path}.zip ({train_time:.1f}s)")

    # Evaluate
    eval_env = lightsim.make("single-intersection-v0", reward_fn=reward_fn, max_steps=720)
    rewards, throughputs = [], []
    for ep in range(10):
        obs, _ = eval_env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
        throughputs.append(eval_env.unwrapped.engine.get_network_metrics().get("total_exited", 0))
    eval_env.close()
    env.close()

    rps = float(np.mean(rewards)) / 720
    tput = float(np.mean(throughputs))
    print(f"    Eval (10 ep): reward/step={rps:.2f}, throughput={tput:.0f}")

    return {
        "name": name,
        "algo": algo_name,
        "scenario": "single-intersection-v0",
        "reward_fn": reward_fn,
        "timesteps": timesteps,
        "train_time_s": round(train_time, 1),
        "reward_per_step": round(rps, 2),
        "mean_throughput": round(tput),
        "seed": seed,
    }


def train_multi_agent(timesteps: int, seed: int = 42) -> dict:
    """Train multi-agent DQN on grid-4x4 with parameter sharing."""
    from stable_baselines3 import DQN

    name = "dqn_grid4x4_multi"
    print(f"\n  Training {name} (DQN shared-parameter, {timesteps:,} steps)...")

    train_env = SharedPolicyMultiAgentWrapper(max_steps=720)
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=2_000,
        batch_size=128,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        target_update_interval=500,
        train_freq=4,
        seed=seed,
        verbose=0,
    )

    t0 = time.perf_counter()
    checkpoint_interval = max(timesteps // 5, 1)
    for i in range(5):
        model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False)
        elapsed = time.perf_counter() - t0
        steps_done = (i + 1) * checkpoint_interval
        print(f"    {steps_done:>7,} / {timesteps:,} steps  ({elapsed:.1f}s)")

    train_time = time.perf_counter() - t0
    save_path = WEIGHTS_DIR / name
    model.save(str(save_path))
    print(f"    Saved to {save_path}.zip ({train_time:.1f}s)")

    return {
        "name": name,
        "algo": "DQN",
        "scenario": "grid-4x4-v0",
        "reward_fn": "queue",
        "timesteps": timesteps,
        "train_time_s": round(train_time, 1),
        "seed": seed,
        "notes": "shared-parameter across 16 agents with obs padding to dim 14",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train pretrained LightSim checkpoints")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Training timesteps per model")
    parser.add_argument("--multi-timesteps", type=int, default=100_000, help="Multi-agent timesteps")
    parser.add_argument("--single-only", action="store_true", help="Train only single-agent models")
    parser.add_argument("--multi-only", action="store_true", help="Train only multi-agent model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from stable_baselines3 import DQN  # noqa: F401
    except ImportError:
        print("ERROR: stable-baselines3 is required. Install with: pip install stable-baselines3")
        sys.exit(1)

    results = []
    print("=" * 60)
    print("  LightSim: Training Pretrained Checkpoints")
    print("=" * 60)

    if not args.multi_only:
        print("\n[Single-Agent Models]")
        for spec in SINGLE_AGENT_MODELS:
            r = train_single_agent(spec, args.timesteps, seed=args.seed)
            results.append(r)

    if not args.single_only:
        print("\n[Multi-Agent Model]")
        r = train_multi_agent(args.multi_timesteps, seed=args.seed)
        results.append(r)

    # Save eval results
    eval_path = WEIGHTS_DIR / "eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {eval_path}")

    # Summary
    print(f"\n{'Model':45s} {'Time':>8s} {'Reward/step':>12s} {'Throughput':>11s}")
    print("-" * 80)
    for r in results:
        rps = r.get("reward_per_step", "N/A")
        rps_str = f"{rps:.2f}" if isinstance(rps, (int, float)) else rps
        tput = r.get("mean_throughput", "N/A")
        tput_str = f"{tput:,}" if isinstance(tput, (int, float)) else tput
        print(f"  {r['name']:43s} {r['train_time_s']:7.1f}s {rps_str:>12s} {tput_str:>11s}")

    print(f"\nAll checkpoints saved to {WEIGHTS_DIR}/")
    print("Run `python scripts/evaluate_pretrained.py` for full evaluation.")


if __name__ == "__main__":
    main()
