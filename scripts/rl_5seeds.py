"""Train DQN and PPO on single-intersection-v0 with 5 seeds each.

Saves results (learning curves + final evaluations + baselines)
to results/rl_training_5seeds.json.

Usage::
    python scripts/rl_5seeds.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Ensure lightsim is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import lightsim
from lightsim.benchmarks.scenarios import get_scenario
from lightsim.core.engine import SimulationEngine
from lightsim.core.signal import FixedTimeController, MaxPressureController
from lightsim.core.types import NodeID

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]
TOTAL_TIMESTEPS = 100_000
EVAL_FREQ = 5_000          # evaluate every N training steps
N_EVAL_EPISODES = 5        # episodes per evaluation during training
FINAL_EVAL_EPISODES = 10   # episodes for final evaluation
RL_MAX_STEPS = 720         # episode length for RL training and eval
BASELINE_MAX_STEPS = 3600  # episode length for baseline evaluation
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_FILE = RESULTS_DIR / "rl_training_5seeds.json"


def make_env(max_steps=RL_MAX_STEPS):
    """Create a single-intersection env wrapped in Monitor."""
    env = lightsim.make("single-intersection-v0", max_steps=max_steps)
    env = Monitor(env)
    return env


def train_algo(algo_name, seeds):
    """Train an RL algorithm across multiple seeds, return per-seed results."""
    results = []
    for seed in seeds:
        sep = "=" * 60
        print()
        print(sep)
        print(f"  Training {algo_name} | seed={seed}")
        print(sep)
        t0 = time.time()

        train_env = make_env(max_steps=RL_MAX_STEPS)
        eval_env = make_env(max_steps=RL_MAX_STEPS)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            log_path = os.path.join(tmpdir, f"{algo_name}_s{seed}")

            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=None,
                log_path=log_path,
                eval_freq=EVAL_FREQ,
                n_eval_episodes=N_EVAL_EPISODES,
                deterministic=True,
                verbose=0,
            )

            if algo_name == "DQN":
                model = DQN(
                    "MlpPolicy",
                    train_env,
                    learning_rate=1e-3,
                    buffer_size=50_000,
                    learning_starts=1_000,
                    batch_size=64,
                    gamma=0.99,
                    exploration_fraction=0.3,
                    exploration_final_eps=0.05,
                    seed=seed,
                    verbose=0,
                )
            elif algo_name == "PPO":
                model = PPO(
                    "MlpPolicy",
                    train_env,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    seed=seed,
                    verbose=0,
                )
            else:
                raise ValueError(f"Unknown algorithm: {algo_name}")

            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                callback=eval_callback,
                progress_bar=False,
            )

            npz_path = os.path.join(log_path, "evaluations.npz")
            eval_data = np.load(npz_path)
            timesteps = eval_data["timesteps"].tolist()
            # eval_data["results"] shape: (n_evals, n_eval_episodes)
            mean_rewards_raw = eval_data["results"].mean(axis=1).tolist()
            # Convert to per-step rewards
            mean_rewards = [r / RL_MAX_STEPS for r in mean_rewards_raw]
            # Close the npz file handle to allow temp dir cleanup on Windows
            eval_data.close()

        # Final evaluation: 10 episodes
        final_env = make_env(max_steps=RL_MAX_STEPS)
        episode_rewards = []
        for ep in range(FINAL_EVAL_EPISODES):
            obs, info = final_env.reset(seed=seed * 1000 + ep)
            total_reward = 0.0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = final_env.step(action)
                total_reward += reward
                done = terminated or truncated
            episode_rewards.append(total_reward / RL_MAX_STEPS)
        final_env.close()

        final_eval_reward = float(np.mean(episode_rewards))

        elapsed = time.time() - t0
        print(f"  Seed {seed} done in {elapsed:.1f}s | "
              f"final_eval_reward={final_eval_reward:.4f}")

        results.append({
            "seed": seed,
            "timesteps": timesteps,
            "mean_rewards": mean_rewards,
            "final_eval_reward": final_eval_reward,
        })

        train_env.close()
        eval_env.close()

    return results


def evaluate_baseline(controller_name, controller, n_episodes=FINAL_EVAL_EPISODES):
    """Evaluate a baseline controller using SimulationEngine directly."""
    sep = "=" * 60
    print()
    print(sep)
    print(f"  Evaluating baseline: {controller_name}")
    print(sep)

    factory = get_scenario("single-intersection-v0")
    network, demand_profiles = factory()

    episode_rewards = []

    for ep in range(n_episodes):
        engine = SimulationEngine(
            network=network,
            dt=1.0,
            controller=controller,
            demand_profiles=demand_profiles,
        )
        engine.reset(seed=ep)

        total_reward = 0.0
        agent_node = NodeID(0)
        for step in range(BASELINE_MAX_STEPS):
            engine.step()
            queue = 0.0
            for link in engine.network.links.values():
                if link.to_node == agent_node:
                    queue += engine.get_link_queue(link.link_id)
            total_reward += -queue

        per_step_reward = total_reward / BASELINE_MAX_STEPS
        episode_rewards.append(per_step_reward)
        print(f"  Episode {ep}: per-step reward = {per_step_reward:.4f}")

    avg_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    print(f"  {controller_name}: avg={avg_reward:.4f}, std={std_reward:.4f}")

    return {"avg_reward": avg_reward, "std_reward": std_reward}


def main():
    print("=" * 60)
    print("  RL 5-Seeds Training Script")
    print(f"  Seeds: {SEEDS}")
    print(f"  Timesteps: {TOTAL_TIMESTEPS}")
    print(f"  Eval freq: {EVAL_FREQ}, Eval episodes: {N_EVAL_EPISODES}")
    print(f"  Final eval episodes: {FINAL_EVAL_EPISODES}")
    print("=" * 60)

    all_results = {"rl": {}, "baselines": {}}

    all_results["rl"]["DQN"] = train_algo("DQN", SEEDS)
    all_results["rl"]["PPO"] = train_algo("PPO", SEEDS)

    all_results["baselines"]["FixedTime"] = evaluate_baseline(
        "FixedTime", FixedTimeController()
    )
    all_results["baselines"]["MaxPressure"] = evaluate_baseline(
        "MaxPressure", MaxPressureController(min_green=5.0)
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_FILE}")
    print("Done!")


if __name__ == "__main__":
    main()
