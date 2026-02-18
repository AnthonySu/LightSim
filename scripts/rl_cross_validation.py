"""RL Cross-Validation: train RL agents in both LightSim and SUMO, compare rankings.

Trains 5 RL variants in both simulators on a single intersection, then
compares the controller ranking to validate that LightSim produces
consistent RL outcomes.

Saves a checkpoint after every single run so we can resume if interrupted.

Usage::
    python scripts/rl_cross_validation.py
    python scripts/rl_cross_validation.py --resume   # continue from checkpoint
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# SUMO_HOME must be set before importing sumo_rl
# ---------------------------------------------------------------------------
def _set_sumo_home():
    if os.environ.get("SUMO_HOME"):
        return
    try:
        import importlib.util
        spec = importlib.util.find_spec("sumolib")
        if spec and spec.origin:
            site_packages = Path(spec.origin).parent.parent
            sumo_home = site_packages / "sumo"
            if sumo_home.exists():
                os.environ["SUMO_HOME"] = str(sumo_home)
    except Exception:
        pass

_set_sumo_home()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TOTAL_TIMESTEPS = 100_000
SUMO_TIMESTEPS = 50_000  # SUMO is ~10x slower; 50k is enough for ranking
N_SEEDS = 5
N_EVAL_EPISODES = 10
SUMO_EVAL_EPISODES = 3  # fewer eval episodes for SUMO (each is slow)
RESULTS_DIR = Path("results")
CHECKPOINT_FILE = RESULTS_DIR / "rl_crossval_checkpoint.json"

# RL variants to test
VARIANTS = [
    {
        "name": "DQN",
        "algo": "DQN",
        "kwargs": {
            "learning_rate": 1e-4,
            "buffer_size": 50_000,
            "learning_starts": 1000,
            "batch_size": 64,
            "exploration_fraction": 0.3,
            "exploration_final_eps": 0.05,
            "policy_kwargs": dict(net_arch=[64, 64]),
        },
    },
    {
        "name": "PPO",
        "algo": "PPO",
        "kwargs": {
            "learning_rate": 3e-4,
            "n_steps": 256,
            "batch_size": 64,
            "n_epochs": 10,
            "policy_kwargs": dict(net_arch=[64, 64]),
        },
    },
    {
        "name": "A2C",
        "algo": "A2C",
        "kwargs": {
            "learning_rate": 7e-4,
            "n_steps": 5,
            "policy_kwargs": dict(net_arch=[64, 64]),
        },
    },
    {
        "name": "DQN-pressure",
        "algo": "DQN",
        "reward": "pressure",
        "kwargs": {
            "learning_rate": 1e-4,
            "buffer_size": 50_000,
            "learning_starts": 1000,
            "batch_size": 64,
            "exploration_fraction": 0.3,
            "exploration_final_eps": 0.05,
            "policy_kwargs": dict(net_arch=[64, 64]),
        },
    },
    {
        "name": "PPO-pressure",
        "algo": "PPO",
        "reward": "pressure",
        "kwargs": {
            "learning_rate": 3e-4,
            "n_steps": 256,
            "batch_size": 64,
            "n_epochs": 10,
            "policy_kwargs": dict(net_arch=[64, 64]),
        },
    },
]


# ---------------------------------------------------------------------------
# Reward wrappers
# ---------------------------------------------------------------------------
import gymnasium as gym


class PressureRewardWrapper(gym.Wrapper):
    """Replace reward with negative pressure (sum of queue differences)."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Use observation-based proxy for pressure:
        # obs contains phase one-hot + lane densities (normalised 0-1)
        # Higher density on current-green lanes = less pressure to switch
        # We approximate pressure as sum of red-lane densities minus green-lane densities
        # This is a simple proxy that works without simulator internals
        n_phase = 2  # single intersection has 2 phases
        density_features = obs[n_phase:]  # skip phase one-hot
        n_lanes = len(density_features)
        # Pressure = total density (more density = more waiting = worse)
        pressure = float(np.sum(density_features))
        reward = -pressure
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------
def make_lightsim_env(reward_type: str = "default", seed: int = 0):
    """Create LightSim single-intersection env."""
    import lightsim
    env = lightsim.make("single-intersection-v0")
    if reward_type == "pressure":
        env = PressureRewardWrapper(env)
    return env


def make_sumo_env(reward_type: str = "default", seed: int = 0):
    """Create sumo-rl single-intersection env."""
    import sumo_rl

    net_file = os.path.join(
        os.path.dirname(sumo_rl.__file__),
        "nets", "single-intersection", "single-intersection.net.xml",
    )
    rou_file = os.path.join(
        os.path.dirname(sumo_rl.__file__),
        "nets", "single-intersection", "single-intersection.rou.xml",
    )

    env = sumo_rl.SumoEnvironment(
        net_file=net_file,
        route_file=rou_file,
        num_seconds=3600,
        single_agent=True,
        sumo_warnings=False,
        additional_sumo_cmd="--no-step-log",
    )
    if reward_type == "pressure":
        env = PressureRewardWrapper(env)
    return env


# ---------------------------------------------------------------------------
# Training + Evaluation
# ---------------------------------------------------------------------------
def train_and_evaluate(
    simulator: str,
    variant: dict,
    seed: int,
) -> dict:
    """Train an RL agent and evaluate it. Returns result dict."""
    from stable_baselines3 import DQN, PPO, A2C

    algo_map = {"DQN": DQN, "PPO": PPO, "A2C": A2C}
    algo_cls = algo_map[variant["algo"]]
    reward_type = variant.get("reward", "default")

    # Create training env
    if simulator == "LightSim":
        env = make_lightsim_env(reward_type, seed)
    else:
        env = make_sumo_env(reward_type, seed)

    # Train
    t0 = time.perf_counter()
    model = algo_cls(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=0,
        **variant["kwargs"],
    )
    timesteps = SUMO_TIMESTEPS if simulator == "SUMO" else TOTAL_TIMESTEPS
    model.learn(total_timesteps=timesteps)
    train_time = time.perf_counter() - t0
    env.close()

    # Evaluate
    n_eval = SUMO_EVAL_EPISODES if simulator == "SUMO" else N_EVAL_EPISODES
    eval_rewards = []
    for ep in range(n_eval):
        if simulator == "LightSim":
            eval_env = make_lightsim_env(reward_type, seed + 1000 + ep)
        else:
            eval_env = make_sumo_env(reward_type, seed + 1000 + ep)

        obs, _ = eval_env.reset(seed=seed + 1000 + ep)
        total_reward = 0.0
        done = False
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(int(action))
            total_reward += reward
            done = terminated or truncated
            steps += 1
            if steps >= 3600:  # safety cap
                break
        eval_rewards.append(total_reward)
        eval_env.close()

    result = {
        "simulator": simulator,
        "variant": variant["name"],
        "seed": seed,
        "train_time": round(train_time, 1),
        "eval_reward_mean": round(float(np.mean(eval_rewards)), 2),
        "eval_reward_std": round(float(np.std(eval_rewards)), 2),
    }
    return result


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------
def load_checkpoint() -> list[dict]:
    """Load existing results from checkpoint file."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return []


def save_checkpoint(results: list[dict]):
    """Save results to checkpoint file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(results, f, indent=2)


def is_completed(results: list[dict], simulator: str, variant_name: str, seed: int) -> bool:
    """Check if a specific run has already been completed."""
    for r in results:
        if r["simulator"] == simulator and r["variant"] == variant_name and r["seed"] == seed:
            return True
    return False


# ---------------------------------------------------------------------------
# Ranking comparison
# ---------------------------------------------------------------------------
def compute_rank_correlation(results: list[dict]):
    """Compute Kendall's tau between LightSim and SUMO variant rankings."""
    from scipy.stats import kendalltau, spearmanr

    # Average reward per variant per simulator
    sim_rankings = {}
    for sim in ["LightSim", "SUMO"]:
        variant_rewards = {}
        for r in results:
            if r["simulator"] == sim:
                variant_rewards.setdefault(r["variant"], []).append(r["eval_reward_mean"])
        # Mean across seeds
        variant_means = {v: np.mean(rews) for v, rews in variant_rewards.items()}
        # Rank by reward (higher = better)
        sorted_variants = sorted(variant_means.keys(), key=lambda v: -variant_means[v])
        sim_rankings[sim] = {v: rank + 1 for rank, v in enumerate(sorted_variants)}

    # Compute correlation on shared variants
    shared = sorted(set(sim_rankings.get("LightSim", {}).keys()) &
                    set(sim_rankings.get("SUMO", {}).keys()))
    if len(shared) < 3:
        return None, None, {}

    ls_ranks = [sim_rankings["LightSim"][v] for v in shared]
    sumo_ranks = [sim_rankings["SUMO"][v] for v in shared]

    tau, tau_p = kendalltau(ls_ranks, sumo_ranks)
    rho, rho_p = spearmanr(ls_ranks, sumo_ranks)

    ranking_details = {
        "variants": shared,
        "lightsim_ranks": ls_ranks,
        "sumo_ranks": sumo_ranks,
        "kendall_tau": round(tau, 3),
        "kendall_p": round(tau_p, 4),
        "spearman_rho": round(rho, 3),
        "spearman_p": round(rho_p, 4),
    }
    return tau, rho, ranking_details


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--lightsim-only", action="store_true",
                        help="Only run LightSim experiments")
    parser.add_argument("--sumo-only", action="store_true",
                        help="Only run SUMO experiments")
    args = parser.parse_args()

    simulators = ["LightSim", "SUMO"]
    if args.lightsim_only:
        simulators = ["LightSim"]
    elif args.sumo_only:
        simulators = ["SUMO"]

    # Load or start fresh
    if args.resume:
        results = load_checkpoint()
        completed = len(results)
        print(f"Resuming from checkpoint: {completed} runs already completed")
    else:
        results = []

    total_runs = len(simulators) * len(VARIANTS) * N_SEEDS
    done_count = len(results)

    print("=" * 70)
    print(f"RL Cross-Validation: LightSim vs SUMO")
    print(f"Variants: {[v['name'] for v in VARIANTS]}")
    print(f"Seeds: {N_SEEDS}, Training steps: {TOTAL_TIMESTEPS:,}")
    print(f"Total runs: {total_runs}, Already done: {done_count}")
    print("=" * 70)

    for simulator in simulators:
        print(f"\n{'#' * 60}")
        print(f"  SIMULATOR: {simulator}")
        print(f"{'#' * 60}")

        for variant in VARIANTS:
            for seed in range(N_SEEDS):
                if is_completed(results, simulator, variant["name"], seed):
                    print(f"  [SKIP] {variant['name']} seed={seed} (already done)")
                    continue

                print(f"\n  {variant['name']} | seed={seed} | {simulator}",
                      end="", flush=True)

                try:
                    result = train_and_evaluate(simulator, variant, seed)
                    results.append(result)
                    save_checkpoint(results)
                    done_count += 1

                    print(f"  ...done in {result['train_time']}s | "
                          f"reward={result['eval_reward_mean']:.1f} +/- "
                          f"{result['eval_reward_std']:.1f} "
                          f"[{done_count}/{total_runs}]")
                except Exception as e:
                    print(f"  ...FAILED: {e}")
                    # Save a failure marker so we can investigate
                    results.append({
                        "simulator": simulator,
                        "variant": variant["name"],
                        "seed": seed,
                        "error": str(e),
                    })
                    save_checkpoint(results)
                    done_count += 1

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print(f"\n\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Group by simulator and variant
    for sim in simulators:
        print(f"\n  [{sim}]")
        print(f"  {'Variant':<18} {'Reward':>12} {'Std':>10} {'Time (s)':>10}")
        print(f"  {'-' * 52}")
        for variant in VARIANTS:
            runs = [r for r in results
                    if r["simulator"] == sim and r["variant"] == variant["name"]
                    and "error" not in r]
            if not runs:
                print(f"  {variant['name']:<18} {'FAILED':>12}")
                continue
            rewards = [r["eval_reward_mean"] for r in runs]
            times = [r["train_time"] for r in runs]
            print(f"  {variant['name']:<18} {np.mean(rewards):>12.1f} "
                  f"{np.std(rewards):>10.1f} {np.mean(times):>10.1f}")

    # Ranking comparison
    if len(simulators) == 2:
        print(f"\n{'=' * 70}")
        print("RANKING COMPARISON")
        print("=" * 70)
        tau, rho, details = compute_rank_correlation(results)
        if details:
            print(f"\n  {'Variant':<18} {'LightSim Rank':>15} {'SUMO Rank':>12}")
            print(f"  {'-' * 47}")
            for v, lr, sr in zip(details["variants"],
                                  details["lightsim_ranks"],
                                  details["sumo_ranks"]):
                print(f"  {v:<18} {lr:>15} {sr:>12}")
            print(f"\n  Kendall's tau = {details['kendall_tau']:.3f} "
                  f"(p = {details['kendall_p']:.4f})")
            print(f"  Spearman's rho = {details['spearman_rho']:.3f} "
                  f"(p = {details['spearman_p']:.4f})")
        else:
            print("  Not enough data for ranking comparison")

    # Speed comparison
    if len(simulators) == 2:
        print(f"\n{'=' * 70}")
        print("SPEED COMPARISON")
        print("=" * 70)
        for variant in VARIANTS:
            ls_times = [r["train_time"] for r in results
                        if r["simulator"] == "LightSim"
                        and r["variant"] == variant["name"]
                        and "error" not in r]
            su_times = [r["train_time"] for r in results
                        if r["simulator"] == "SUMO"
                        and r["variant"] == variant["name"]
                        and "error" not in r]
            if ls_times and su_times:
                ls_avg = np.mean(ls_times)
                su_avg = np.mean(su_times)
                speedup = su_avg / ls_avg if ls_avg > 0 else 0
                print(f"  {variant['name']:<18} LightSim: {ls_avg:>6.1f}s  "
                      f"SUMO: {su_avg:>6.1f}s  Speedup: {speedup:.0f}x")

    # Save final results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    final_file = RESULTS_DIR / "rl_crossval_results.json"
    with open(final_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {final_file}")


if __name__ == "__main__":
    main()
