"""Reward function ablation study for DQN on single-intersection scenario.

Trains a DQN agent with each of 4 reward functions and evaluates all on the
same queue-based metric for fair comparison.

Results saved to: ../results/reward_ablation.json
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure lightsim is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightsim
from lightsim.envs.rewards import get_reward_function
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REWARD_FUNCTIONS = ["queue", "pressure", "delay", "throughput"]
TRAIN_TIMESTEPS = 50_000
EVAL_EPISODES = 10
SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_FILE = RESULTS_DIR / "reward_ablation.json"

# ---------------------------------------------------------------------------
# Training callback to track mean episode reward
# ---------------------------------------------------------------------------
class EpisodeRewardTracker(BaseCallback):
    """Tracks episode rewards during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # SB3 stores episode info in 'infos' when episode ends (via Monitor)
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is not None:
                self.episode_rewards.append(ep_info["r"])
        return True


# ---------------------------------------------------------------------------
# Evaluation function: run trained agent and measure queue-based reward
# ---------------------------------------------------------------------------
def evaluate_agent(model, reward_fn_name, n_episodes=10, seed=42):
    """Evaluate a trained agent on the single-intersection scenario.

    Always evaluates using the QUEUE reward for fair comparison, but also
    reports throughput (total_exited).
    """
    # Create a fresh env with the SAME reward the agent was trained on
    # (so the agent gets proper observations), but we compute queue reward
    # externally for comparison.
    env = lightsim.make("single-intersection-v0", reward_fn=reward_fn_name)
    queue_reward_fn = get_reward_function("queue")

    episode_queue_rewards = []
    episode_throughputs = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        total_queue_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            # Compute queue reward from the engine state directly
            q_reward = queue_reward_fn.compute(env.engine, env.agent_node)
            total_queue_reward += q_reward
            done = terminated or truncated

        episode_queue_rewards.append(total_queue_reward)
        episode_throughputs.append(info["total_exited"])

    return {
        "mean_queue_reward": float(np.mean(episode_queue_rewards)),
        "std_queue_reward": float(np.std(episode_queue_rewards)),
        "mean_throughput": float(np.mean(episode_throughputs)),
        "std_throughput": float(np.std(episode_throughputs)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {"reward_functions": []}

    print("=" * 70)
    print("REWARD FUNCTION ABLATION STUDY")
    print(f"  Scenario: single-intersection-v0")
    print(f"  Algorithm: DQN (Stable-Baselines3)")
    print(f"  Training timesteps: {TRAIN_TIMESTEPS:,}")
    print(f"  Eval episodes: {EVAL_EPISODES}")
    print(f"  Seed: {SEED}")
    print("=" * 70)

    for reward_name in REWARD_FUNCTIONS:
        print(f"\n{'─' * 70}")
        print(f"  Reward function: {reward_name}")
        print(f"{'─' * 70}")

        # 1. Create environment
        print(f"  Creating environment with reward_fn='{reward_name}'...")
        env = lightsim.make("single-intersection-v0", reward_fn=reward_name)

        # 2. Create DQN agent
        print(f"  Initialising DQN agent (seed={SEED})...")
        model = DQN(
            "MlpPolicy",
            env,
            seed=SEED,
            verbose=0,
            learning_rate=1e-3,
            buffer_size=50_000,
            learning_starts=1_000,
            batch_size=64,
            gamma=0.99,
            target_update_interval=500,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
        )

        # 3. Train
        print(f"  Training for {TRAIN_TIMESTEPS:,} timesteps...")
        t0 = time.time()
        tracker = EpisodeRewardTracker()
        model.learn(total_timesteps=TRAIN_TIMESTEPS, callback=tracker)
        train_time = time.time() - t0
        print(f"  Training complete in {train_time:.1f}s")

        # Compute mean training reward from logged episodes
        if tracker.episode_rewards:
            mean_train_reward = float(np.mean(tracker.episode_rewards))
            print(f"  Mean training episode reward: {mean_train_reward:.2f} "
                  f"({len(tracker.episode_rewards)} episodes)")
        else:
            mean_train_reward = float("nan")
            print(f"  No full training episodes completed.")

        # 4. Evaluate
        print(f"  Evaluating for {EVAL_EPISODES} episodes (queue-based metric)...")
        eval_results = evaluate_agent(
            model, reward_name, n_episodes=EVAL_EPISODES, seed=SEED
        )

        print(f"  Eval queue reward:  {eval_results['mean_queue_reward']:.2f} "
              f"(+/- {eval_results['std_queue_reward']:.2f})")
        print(f"  Eval throughput:    {eval_results['mean_throughput']:.1f} "
              f"(+/- {eval_results['std_throughput']:.1f})")

        # 5. Store results
        entry = {
            "name": reward_name,
            "train_reward": mean_train_reward,
            "eval_queue_reward": eval_results["mean_queue_reward"],
            "eval_throughput": eval_results["mean_throughput"],
            "eval_queue_reward_std": eval_results["std_queue_reward"],
            "eval_throughput_std": eval_results["std_throughput"],
            "train_time_seconds": round(train_time, 1),
            "train_episodes": len(tracker.episode_rewards),
        }
        results["reward_functions"].append(entry)

        # Save intermediate results (in case of crash)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Reward':<14} {'Train Rew':>12} {'Eval Queue Rew':>16} {'Eval Throughput':>16}")
    print(f"{'─' * 14} {'─' * 12} {'─' * 16} {'─' * 16}")
    for entry in results["reward_functions"]:
        print(f"{entry['name']:<14} {entry['train_reward']:>12.2f} "
              f"{entry['eval_queue_reward']:>16.2f} "
              f"{entry['eval_throughput']:>16.1f}")
    print(f"{'=' * 70}")

    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
