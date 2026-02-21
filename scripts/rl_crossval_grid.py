"""Multi-Agent RL Cross-Validation on Grid-4x4: LightSim vs SUMO.

Trains DQN and PPO with parameter-sharing on grid-4x4 in both simulators,
then compares RL algorithm rankings and training time.

Usage::
    python scripts/rl_crossval_grid.py
    python scripts/rl_crossval_grid.py --lightsim-only
    python scripts/rl_crossval_grid.py --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# SUMO_HOME setup
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
TOTAL_TIMESTEPS = 50_000
N_SEEDS = 3
SEEDS = [0, 1, 2]
N_EVAL_EPISODES = 3
MAX_OBS_DIM = 14

RESULTS_DIR = Path("results")
RESULTS_FILE = RESULTS_DIR / "rl_crossval_grid.json"
CHECKPOINT_FILE = RESULTS_DIR / "rl_crossval_grid_checkpoint.json"

VARIANTS = [
    {
        "name": "DQN",
        "algo": "DQN",
        "kwargs": {
            "learning_rate": 1e-3,
            "buffer_size": 50_000,
            "learning_starts": 2_000,
            "batch_size": 128,
            "exploration_fraction": 0.3,
            "exploration_final_eps": 0.05,
            "target_update_interval": 500,
            "train_freq": 4,
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
]


# ---------------------------------------------------------------------------
# Observation padding (reused from multi_agent_rl.py)
# ---------------------------------------------------------------------------
def pad_obs(obs, target_dim=MAX_OBS_DIM):
    """Pad observation vector to target_dim with trailing zeros."""
    if len(obs) >= target_dim:
        return obs[:target_dim].astype(np.float32)
    padded = np.zeros(target_dim, dtype=np.float32)
    padded[:len(obs)] = obs
    return padded


# ---------------------------------------------------------------------------
# LightSim multi-agent wrapper (from multi_agent_rl.py)
# ---------------------------------------------------------------------------
class LightSimMultiAgentWrapper(gym.Env):
    """PettingZoo grid-4x4 wrapped as single-agent Gym with parameter sharing."""

    metadata = {"render_modes": []}

    def __init__(self, max_steps=720):
        super().__init__()
        import lightsim
        self.max_steps = max_steps
        self.pz_env = lightsim.parallel_env("grid-4x4-v0", max_steps=max_steps)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(MAX_OBS_DIM,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(2)
        self._agents = []
        self._agent_idx = 0
        self._actions = {}
        self._obs = {}
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
            return pad_obs(np.zeros(MAX_OBS_DIM)), 0.0, False, True, {}

        agent_name = self._agents[self._agent_idx]
        self._actions[agent_name] = int(action)
        self._agent_idx += 1

        if self._agent_idx < len(self._agents):
            next_obs = pad_obs(self._obs[self._agents[self._agent_idx]])
            return next_obs, 0.0, False, False, {}

        # All agents acted — step PZ env
        self._obs, rewards, terms, truncs, infos = self.pz_env.step(self._actions)
        mean_reward = float(np.mean(list(rewards.values())))

        if not self.pz_env.agents:
            self._done = True
            return pad_obs(np.zeros(MAX_OBS_DIM)), mean_reward, False, True, {}

        self._agents = list(self.pz_env.agents)
        self._agent_idx = 0
        self._actions = {}
        return pad_obs(self._obs[self._agents[0]]), mean_reward, False, False, {}


# ---------------------------------------------------------------------------
# SUMO multi-agent wrapper
# ---------------------------------------------------------------------------
class SUMOMultiAgentWrapper(gym.Env):
    """Wraps sumo_rl grid env as single-agent Gym with parameter sharing.

    sumo_rl.parallel_env for grid networks provides per-agent obs/actions.
    We normalize obs to MAX_OBS_DIM by padding and cycle through agents.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps=720):
        super().__init__()
        self.max_steps = max_steps
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(MAX_OBS_DIM,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(2)
        self._pz_env = None
        self._agents = []
        self._agent_idx = 0
        self._actions = {}
        self._obs = {}
        self._done = False

    def _make_env(self, seed=None):
        """Create sumo_rl grid env."""
        import sumo_rl
        env = sumo_rl.parallel_env(
            net_file=sumo_rl.grid4x4_net,
            route_file=sumo_rl.grid4x4_rou1,
            num_seconds=self.max_steps,
            reward_fn="diff-waiting-time",
            sumo_warnings=False,
            additional_sumo_cmd="--no-step-log",
        )
        return env

    def _normalize_obs(self, obs):
        """Normalize SUMO obs to [0,1] range and pad to MAX_OBS_DIM."""
        obs = np.array(obs, dtype=np.float32)
        # Clip extreme values and normalize
        obs = np.clip(obs, -100, 100)
        obs = (obs - obs.min()) / (obs.max() - obs.min() + 1e-8)
        return pad_obs(obs)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._pz_env is not None:
            try:
                self._pz_env.close()
            except Exception:
                pass
        self._pz_env = self._make_env(seed)
        self._obs, _ = self._pz_env.reset(seed=seed)
        self._agents = list(self._pz_env.agents)
        self._agent_idx = 0
        self._actions = {}
        self._done = False
        if not self._agents:
            self._done = True
            return pad_obs(np.zeros(MAX_OBS_DIM)), {}
        return self._normalize_obs(self._obs[self._agents[0]]), {}

    def step(self, action):
        if self._done:
            return pad_obs(np.zeros(MAX_OBS_DIM)), 0.0, False, True, {}

        agent_name = self._agents[self._agent_idx]
        self._actions[agent_name] = int(action)
        self._agent_idx += 1

        if self._agent_idx < len(self._agents):
            next_obs = self._normalize_obs(
                self._obs[self._agents[self._agent_idx]])
            return next_obs, 0.0, False, False, {}

        # All agents acted — step PZ env
        self._obs, rewards, terms, truncs, infos = self._pz_env.step(self._actions)
        mean_reward = float(np.mean(list(rewards.values())))

        if not self._pz_env.agents:
            self._done = True
            return pad_obs(np.zeros(MAX_OBS_DIM)), mean_reward, False, True, {}

        self._agents = list(self._pz_env.agents)
        self._agent_idx = 0
        self._actions = {}
        return self._normalize_obs(self._obs[self._agents[0]]), mean_reward, False, False, {}

    def close(self):
        if self._pz_env is not None:
            try:
                self._pz_env.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Training + Evaluation
# ---------------------------------------------------------------------------
def evaluate_multi_agent(model, simulator: str, n_episodes: int = N_EVAL_EPISODES,
                         seed_offset: int = 1000) -> list[float]:
    """Evaluate model on multi-agent env, return per-episode rewards."""
    rewards = []
    for ep in range(n_episodes):
        if simulator == "LightSim":
            env = LightSimMultiAgentWrapper(max_steps=720)
        else:
            env = SUMOMultiAgentWrapper(max_steps=720)

        obs, _ = env.reset(seed=seed_offset + ep)
        total_reward = 0.0
        done = False
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward
            done = terminated or truncated
            steps += 1
            if steps >= 720 * 20:  # safety cap (720 steps * ~16 agents)
                break
        rewards.append(total_reward)
        env.close()

    return rewards


def train_and_evaluate(simulator: str, variant: dict, seed: int) -> dict:
    """Train RL agent on grid-4x4 and evaluate."""
    from stable_baselines3 import DQN, PPO

    algo_map = {"DQN": DQN, "PPO": PPO}
    algo_cls = algo_map[variant["algo"]]

    if simulator == "LightSim":
        env = LightSimMultiAgentWrapper(max_steps=720)
    else:
        env = SUMOMultiAgentWrapper(max_steps=720)

    t0 = time.perf_counter()
    model = algo_cls(
        "MlpPolicy", env, seed=seed, verbose=0, **variant["kwargs"])
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    train_time = time.perf_counter() - t0
    env.close()

    # Evaluate
    eval_rewards = evaluate_multi_agent(
        model, simulator, n_episodes=N_EVAL_EPISODES,
        seed_offset=seed * 1000)

    return {
        "simulator": simulator,
        "variant": variant["name"],
        "seed": seed,
        "train_time": round(train_time, 1),
        "eval_reward_mean": round(float(np.mean(eval_rewards)), 2),
        "eval_reward_std": round(float(np.std(eval_rewards)), 2),
    }


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------
def load_checkpoint() -> list[dict]:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return []


def save_checkpoint(results: list[dict]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(results, f, indent=2)


def is_completed(results: list[dict], simulator: str,
                 variant_name: str, seed: int) -> bool:
    return any(
        r["simulator"] == simulator and r["variant"] == variant_name
        and r["seed"] == seed
        for r in results
    )


# ---------------------------------------------------------------------------
# Ranking comparison
# ---------------------------------------------------------------------------
def compute_rank_correlation(results: list[dict]) -> dict | None:
    from scipy.stats import kendalltau, spearmanr

    sim_rankings = {}
    for sim in ["LightSim", "SUMO"]:
        variant_rewards = {}
        for r in results:
            if r["simulator"] == sim and "error" not in r:
                variant_rewards.setdefault(r["variant"], []).append(
                    r["eval_reward_mean"])
        if not variant_rewards:
            continue
        variant_means = {v: np.mean(rews) for v, rews in variant_rewards.items()}
        sorted_variants = sorted(variant_means.keys(),
                                 key=lambda v: -variant_means[v])
        sim_rankings[sim] = {v: rank + 1 for rank, v in enumerate(sorted_variants)}

    if "LightSim" not in sim_rankings or "SUMO" not in sim_rankings:
        return None

    shared = sorted(set(sim_rankings["LightSim"].keys()) &
                    set(sim_rankings["SUMO"].keys()))
    if len(shared) < 2:
        return None

    ls_ranks = [sim_rankings["LightSim"][v] for v in shared]
    sumo_ranks = [sim_rankings["SUMO"][v] for v in shared]

    tau, tau_p = kendalltau(ls_ranks, sumo_ranks)
    rho, rho_p = spearmanr(ls_ranks, sumo_ranks)

    return {
        "variants": shared,
        "lightsim_ranks": ls_ranks,
        "sumo_ranks": sumo_ranks,
        "kendall_tau": round(float(tau), 3),
        "kendall_p": round(float(tau_p), 4),
        "spearman_rho": round(float(rho), 3),
        "spearman_p": round(float(rho_p), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lightsim-only", action="store_true")
    parser.add_argument("--sumo-only", action="store_true")
    args = parser.parse_args()

    simulators = ["LightSim", "SUMO"]
    if args.lightsim_only:
        simulators = ["LightSim"]
    elif args.sumo_only:
        simulators = ["SUMO"]

    results = load_checkpoint() if args.resume else []
    if results:
        print(f"Resuming: {len(results)} runs already completed")

    total_runs = len(simulators) * len(VARIANTS) * N_SEEDS
    done = len(results)

    print("=" * 70)
    print("Multi-Agent RL Cross-Validation: Grid-4x4")
    print(f"Variants: {[v['name'] for v in VARIANTS]}")
    print(f"Seeds: {N_SEEDS}, Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Total runs: {total_runs}, Already done: {done}")
    print("=" * 70)

    for simulator in simulators:
        print(f"\n{'#' * 60}")
        print(f"  SIMULATOR: {simulator}")
        print(f"{'#' * 60}")

        for variant in VARIANTS:
            for seed in SEEDS:
                if is_completed(results, simulator, variant["name"], seed):
                    print(f"  [SKIP] {variant['name']} seed={seed}")
                    continue

                print(f"\n  {variant['name']} | seed={seed} | {simulator}",
                      end="", flush=True)
                try:
                    result = train_and_evaluate(simulator, variant, seed)
                    results.append(result)
                    save_checkpoint(results)
                    done += 1
                    print(f"  ...done in {result['train_time']}s | "
                          f"reward={result['eval_reward_mean']:.1f} +/- "
                          f"{result['eval_reward_std']:.1f} [{done}/{total_runs}]")
                except Exception as e:
                    print(f"  ...FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    results.append({
                        "simulator": simulator,
                        "variant": variant["name"],
                        "seed": seed,
                        "error": str(e),
                    })
                    save_checkpoint(results)
                    done += 1

    # Summary
    print(f"\n\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print("=" * 70)

    for sim in simulators:
        print(f"\n  [{sim}]")
        print(f"  {'Variant':<12} {'Reward':>12} {'Std':>10} {'Time (s)':>10}")
        print(f"  {'-' * 46}")
        for variant in VARIANTS:
            runs = [r for r in results
                    if r["simulator"] == sim and r["variant"] == variant["name"]
                    and "error" not in r]
            if not runs:
                print(f"  {variant['name']:<12} {'FAILED':>12}")
                continue
            rewards = [r["eval_reward_mean"] for r in runs]
            times = [r["train_time"] for r in runs]
            print(f"  {variant['name']:<12} {np.mean(rewards):>12.1f} "
                  f"{np.std(rewards):>10.1f} {np.mean(times):>10.1f}")

    # Ranking comparison
    if len(simulators) == 2:
        print(f"\n{'=' * 70}")
        print("RANKING COMPARISON")
        print("=" * 70)
        ranking = compute_rank_correlation(results)
        if ranking:
            print(f"\n  {'Variant':<12} {'LightSim Rank':>15} {'SUMO Rank':>12}")
            print(f"  {'-' * 41}")
            for v, lr, sr in zip(ranking["variants"],
                                  ranking["lightsim_ranks"],
                                  ranking["sumo_ranks"]):
                print(f"  {v:<12} {lr:>15} {sr:>12}")
            print(f"\n  Kendall's tau = {ranking['kendall_tau']:.3f} "
                  f"(p = {ranking['kendall_p']:.4f})")
            print(f"  Spearman's rho = {ranking['spearman_rho']:.3f} "
                  f"(p = {ranking['spearman_p']:.4f})")
        else:
            print("  Not enough data for ranking comparison")

    # Speed comparison
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
            print(f"  {variant['name']:<12} LightSim: {ls_avg:>6.1f}s  "
                  f"SUMO: {su_avg:>6.1f}s  Speedup: {speedup:.1f}x")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "rl_crossval_grid",
        "scenario": "grid-4x4-v0",
        "num_agents": 16,
        "total_timesteps": TOTAL_TIMESTEPS,
        "n_seeds": N_SEEDS,
        "variants": [v["name"] for v in VARIANTS],
        "results": [r for r in results if "error" not in r],
    }
    ranking = compute_rank_correlation(results)
    if ranking:
        output["ranking_correlation"] = ranking

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
