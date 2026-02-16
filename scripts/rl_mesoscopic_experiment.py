"""Train DQN and PPO on single-intersection in default vs mesoscopic mode.

Compares RL performance across simulator configurations and against
rule-based baselines (FixedTime, MaxPressure, LT-Aware-MP).

Usage::
    python scripts/rl_mesoscopic_experiment.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import lightsim
from lightsim.benchmarks.scenarios import get_scenario
from lightsim.core.engine import SimulationEngine
from lightsim.core.signal import (
    FixedTimeController,
    LostTimeAwareMaxPressureController,
    MaxPressureController,
)
from lightsim.core.types import NodeID, NodeType

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]
TOTAL_TIMESTEPS = 100_000
EVAL_FREQ = 5_000
N_EVAL_EPISODES = 5
FINAL_EVAL_EPISODES = 10
RL_MAX_STEPS = 720
BASELINE_STEPS = 3600
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_FILE = RESULTS_DIR / "rl_mesoscopic_experiment.json"


def make_env(stochastic: bool = False, max_steps: int = RL_MAX_STEPS):
    env = lightsim.make("single-intersection-v0",
                        max_steps=max_steps, stochastic=stochastic)
    return Monitor(env)


def make_meso_env(max_steps: int = RL_MAX_STEPS):
    """Create env with mesoscopic mode (stochastic + lost_time on phases)."""
    from lightsim.benchmarks.scenarios import get_scenario
    factory = get_scenario("single-intersection-v0")
    network, demand_profiles = factory()

    # Patch lost_time
    for node in network.nodes.values():
        for phase in node.phases:
            phase.lost_time = 2.0

    from lightsim.core.signal import RLController
    from lightsim.envs.single_agent import LightSimEnv
    env = LightSimEnv(
        network=network,
        dt=1.0,
        max_steps=max_steps,
        demand_profiles=demand_profiles,
        stochastic=True,
    )
    return Monitor(env)


def train_algo(algo_name: str, mode: str, seeds: list[int]):
    """Train an RL algorithm across seeds in a given mode."""
    results = []
    for seed in seeds:
        print(f"\n{'=' * 60}")
        print(f"  {algo_name} | mode={mode} | seed={seed}")
        print("=" * 60)
        t0 = time.time()

        if mode == "mesoscopic":
            train_env = make_meso_env(max_steps=RL_MAX_STEPS)
            eval_env = make_meso_env(max_steps=RL_MAX_STEPS)
        else:
            train_env = make_env(stochastic=False, max_steps=RL_MAX_STEPS)
            eval_env = make_env(stochastic=False, max_steps=RL_MAX_STEPS)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            log_path = os.path.join(tmpdir, f"{algo_name}_{mode}_s{seed}")

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
                    "MlpPolicy", train_env,
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
            else:
                model = PPO(
                    "MlpPolicy", train_env,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    seed=seed,
                    verbose=0,
                )

            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                callback=eval_callback,
                progress_bar=False,
            )

            npz_path = os.path.join(log_path, "evaluations.npz")
            eval_data = np.load(npz_path)
            timesteps = eval_data["timesteps"].tolist()
            mean_rewards = (eval_data["results"].mean(axis=1) / RL_MAX_STEPS).tolist()
            eval_data.close()

        # Final evaluation
        if mode == "mesoscopic":
            final_env = make_meso_env(max_steps=RL_MAX_STEPS)
        else:
            final_env = make_env(stochastic=False, max_steps=RL_MAX_STEPS)

        episode_rewards = []
        episode_throughputs = []
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
            episode_throughputs.append(info.get("total_exited", 0))
        final_env.close()

        elapsed = time.time() - t0
        avg_reward = float(np.mean(episode_rewards))
        avg_throughput = float(np.mean(episode_throughputs))
        print(f"  Done in {elapsed:.1f}s | reward={avg_reward:.4f} | "
              f"throughput={avg_throughput:.0f}")

        results.append({
            "seed": seed,
            "timesteps": timesteps,
            "mean_rewards": mean_rewards,
            "final_eval_reward": avg_reward,
            "final_throughput": avg_throughput,
            "wall_time": elapsed,
        })

        train_env.close()
        eval_env.close()

    return results


def evaluate_baseline(name: str, controller, mode: str,
                      n_episodes: int = FINAL_EVAL_EPISODES):
    """Evaluate a rule-based controller."""
    print(f"\n  Baseline: {name} ({mode})")
    factory = get_scenario("single-intersection-v0")
    network, demand_profiles = factory()

    stochastic = (mode == "mesoscopic")
    if mode == "mesoscopic":
        for node in network.nodes.values():
            for phase in node.phases:
                phase.lost_time = 2.0

    episode_rewards = []
    episode_throughputs = []
    agent_node = NodeID(0)

    for ep in range(n_episodes):
        engine = SimulationEngine(
            network=network, dt=1.0,
            controller=controller,
            demand_profiles=demand_profiles,
            stochastic=stochastic,
        )
        engine.reset(seed=ep)

        total_reward = 0.0
        for step in range(BASELINE_STEPS):
            engine.step()
            queue = 0.0
            for link in engine.network.links.values():
                if link.to_node == agent_node:
                    queue += engine.get_link_queue(link.link_id)
            total_reward += -queue

        per_step = total_reward / BASELINE_STEPS
        episode_rewards.append(per_step)
        episode_throughputs.append(engine.state.total_exited)

    avg_r = float(np.mean(episode_rewards))
    avg_t = float(np.mean(episode_throughputs))
    print(f"    reward={avg_r:.4f}, throughput={avg_t:.0f}")
    return {
        "avg_reward": avg_r,
        "std_reward": float(np.std(episode_rewards)),
        "avg_throughput": avg_t,
    }


def main():
    print("=" * 60)
    print("  RL Mesoscopic Experiment")
    print(f"  Modes: default, mesoscopic")
    print(f"  Algos: DQN, PPO × {len(SEEDS)} seeds × {TOTAL_TIMESTEPS} steps")
    print("=" * 60)

    all_results = {}

    for mode in ["default", "mesoscopic"]:
        print(f"\n{'#' * 60}")
        print(f"  MODE: {mode.upper()}")
        print("#" * 60)

        mode_results = {"rl": {}, "baselines": {}}

        # RL training
        for algo in ["DQN", "PPO"]:
            mode_results["rl"][algo] = train_algo(algo, mode, SEEDS)

        # Baselines
        baselines = [
            ("FixedTime", FixedTimeController()),
            ("MaxPressure-mg5", MaxPressureController(min_green=5.0)),
            ("MaxPressure-mg15", MaxPressureController(min_green=15.0)),
            ("LT-Aware-MP-mg5", LostTimeAwareMaxPressureController(min_green=5.0)),
        ]
        print(f"\n  --- Baselines ({mode}) ---")
        for name, ctrl in baselines:
            mode_results["baselines"][name] = evaluate_baseline(name, ctrl, mode)

        all_results[mode] = mode_results

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)

    header = f"{'Mode':<12} {'Controller':<22} {'Reward':>10} {'Throughput':>12}"
    print(header)
    print("-" * 60)

    for mode in ["default", "mesoscopic"]:
        mr = all_results[mode]
        # RL
        for algo in ["DQN", "PPO"]:
            rewards = [s["final_eval_reward"] for s in mr["rl"][algo]]
            throughputs = [s["final_throughput"] for s in mr["rl"][algo]]
            avg_r = np.mean(rewards)
            avg_t = np.mean(throughputs)
            std_r = np.std(rewards)
            print(f"{mode:<12} {algo:<22} {avg_r:>9.4f}± {std_r:.3f} {avg_t:>8.0f}")
        # Baselines
        for name, bl in mr["baselines"].items():
            print(f"{mode:<12} {name:<22} {bl['avg_reward']:>10.4f} {bl['avg_throughput']:>12.0f}")
        print()

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
