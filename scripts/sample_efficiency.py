"""Sample Efficiency: wall-clock convergence of RL in LightSim vs SUMO.

Trains DQN on single-intersection in both simulators, logging eval reward
at regular timestep intervals. Shows LightSim reaches target performance
faster in wall-clock time.

Usage::
    python scripts/sample_efficiency.py
    python scripts/sample_efficiency.py --lightsim-only
    python scripts/sample_efficiency.py --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

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
TOTAL_TIMESTEPS = 100_000
EVAL_INTERVAL = 10_000      # evaluate every 10k steps
N_EVAL_EPISODES = 5
N_SEEDS = 3
SEEDS = [0, 1, 2]

RESULTS_DIR = Path("results")
RESULTS_FILE = RESULTS_DIR / "sample_efficiency.json"
CHECKPOINT_FILE = RESULTS_DIR / "sample_efficiency_checkpoint.json"

DQN_KWARGS = {
    "learning_rate": 1e-4,
    "buffer_size": 50_000,
    "learning_starts": 1000,
    "batch_size": 64,
    "exploration_fraction": 0.3,
    "exploration_final_eps": 0.05,
    "policy_kwargs": dict(net_arch=[64, 64]),
}

# Baseline rewards (will be computed at runtime for reference lines)
BASELINES = {}


# ---------------------------------------------------------------------------
# Environment factories (from rl_cross_validation.py)
# ---------------------------------------------------------------------------
def make_lightsim_env(seed: int = 0):
    import lightsim
    return lightsim.make("single-intersection-v0")


def make_sumo_env(seed: int = 0):
    import sumo_rl
    net_file = os.path.join(
        os.path.dirname(sumo_rl.__file__),
        "nets", "single-intersection", "single-intersection.net.xml",
    )
    rou_file = os.path.join(
        os.path.dirname(sumo_rl.__file__),
        "nets", "single-intersection", "single-intersection.rou.xml",
    )
    return sumo_rl.SumoEnvironment(
        net_file=net_file,
        route_file=rou_file,
        num_seconds=3600,
        single_agent=True,
        sumo_warnings=False,
        additional_sumo_cmd="--no-step-log",
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, simulator: str, n_episodes: int = N_EVAL_EPISODES,
                   seed_offset: int = 1000) -> float:
    """Evaluate model and return mean reward."""
    rewards = []
    for ep in range(n_episodes):
        if simulator == "LightSim":
            env = make_lightsim_env(seed_offset + ep)
        else:
            env = make_sumo_env(seed_offset + ep)

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
            if steps >= 3600:
                break
        rewards.append(total_reward)
        env.close()

    return float(np.mean(rewards))


# ---------------------------------------------------------------------------
# Baseline controllers
# ---------------------------------------------------------------------------
def compute_baselines():
    """Compute baseline rewards for reference lines on convergence plot.

    Runs the RL env with simple policies:
    - FixedTime: always action 0 (keep current phase = cycle through phases)
    - MaxPressure-like: alternate actions every min_green steps
    """
    baselines = {}

    for name, policy_fn in [
        ("FixedTime", lambda obs, step: 0),
        ("MaxPressure-proxy", lambda obs, step: 1 if (step % 30) < 15 else 0),
    ]:
        rewards = []
        for ep in range(5):
            env = make_lightsim_env(42 + ep)
            obs, _ = env.reset(seed=42 + ep)
            total_reward = 0.0
            done = False
            step = 0
            while not done:
                action = policy_fn(obs, step)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
                step += 1
            rewards.append(total_reward)
            env.close()
        baselines[name] = float(np.mean(rewards))

    return baselines


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------
def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"curves": [], "baselines": {}}


def save_checkpoint(data: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)


def is_completed(data: dict, simulator: str, seed: int) -> bool:
    return any(
        c["simulator"] == simulator and c["seed"] == seed
        for c in data["curves"]
    )


# ---------------------------------------------------------------------------
# Train with periodic evaluation
# ---------------------------------------------------------------------------
def train_with_checkpoints(simulator: str, seed: int) -> dict:
    """Train DQN with periodic evaluation, return convergence curve."""
    from stable_baselines3 import DQN

    if simulator == "LightSim":
        env = make_lightsim_env(seed)
    else:
        env = make_sumo_env(seed)

    model = DQN("MlpPolicy", env, seed=seed, verbose=0, **DQN_KWARGS)

    curve = []
    t_start = time.perf_counter()

    n_checkpoints = TOTAL_TIMESTEPS // EVAL_INTERVAL
    for i in range(n_checkpoints):
        model.learn(total_timesteps=EVAL_INTERVAL, reset_num_timesteps=False)
        wall_time = time.perf_counter() - t_start
        timesteps_done = (i + 1) * EVAL_INTERVAL

        eval_reward = evaluate_model(model, simulator, n_episodes=N_EVAL_EPISODES,
                                     seed_offset=seed * 1000 + i * 100)

        curve.append({
            "timesteps": timesteps_done,
            "wall_seconds": round(wall_time, 2),
            "eval_reward": round(eval_reward, 2),
        })
        print(f"      {timesteps_done:>7,} steps | {wall_time:>6.1f}s | "
              f"reward={eval_reward:.1f}")

    env.close()
    total_time = time.perf_counter() - t_start

    return {
        "simulator": simulator,
        "seed": seed,
        "total_train_time": round(total_time, 2),
        "curve": curve,
    }


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------
def generate_figure(data: dict):
    """Generate sample efficiency figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    OVERLEAF = Path(r"C:\Users\admin\Projects\69927a89543379cbbfcbc218\figures")
    OVERLEAF.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 0.6,
        'legend.fontsize': 8,
        'figure.dpi': 150,
        'savefig.bbox': 'tight',
    })

    BLUE = '#4472C4'
    RED = '#C0504D'
    GRAY = '#78909C'
    GREEN = '#548235'

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for sim, color, label in [("LightSim", BLUE, "LightSim"),
                               ("SUMO", RED, "SUMO")]:
        sim_curves = [c for c in data["curves"] if c["simulator"] == sim]
        if not sim_curves:
            continue

        # Collect all curves: each has list of (wall_seconds, eval_reward)
        all_times = []
        all_rewards = []
        for c in sim_curves:
            times = [p["wall_seconds"] for p in c["curve"]]
            rewards = [p["eval_reward"] for p in c["curve"]]
            all_times.append(times)
            all_rewards.append(rewards)

        # Plot individual runs faintly
        for times, rewards in zip(all_times, all_rewards):
            ax.plot(times, rewards, color=color, alpha=0.2, linewidth=0.8)

        # Plot mean
        if len(all_rewards) > 1:
            # Interpolate to common time grid
            max_time = max(t[-1] for t in all_times)
            time_grid = np.linspace(0, max_time, 50)
            interp_rewards = []
            for times, rewards in zip(all_times, all_rewards):
                interp_rewards.append(np.interp(time_grid, times, rewards))
            mean_r = np.mean(interp_rewards, axis=0)
            std_r = np.std(interp_rewards, axis=0)
            ax.plot(time_grid, mean_r, color=color, linewidth=2.0, label=label)
            ax.fill_between(time_grid, mean_r - std_r, mean_r + std_r,
                            color=color, alpha=0.15)
        else:
            ax.plot(all_times[0], all_rewards[0], color=color,
                    linewidth=2.0, label=label)

    # Baseline horizontal lines
    baselines = data.get("baselines", {})
    if "MaxPressure" in baselines:
        ax.axhline(y=baselines["MaxPressure"], color=GREEN, linestyle='--',
                   linewidth=1.0, alpha=0.7, label='MaxPressure baseline')
    if "FixedTime" in baselines:
        ax.axhline(y=baselines["FixedTime"], color=GRAY, linestyle=':',
                   linewidth=1.0, alpha=0.7, label='FixedTime baseline')

    ax.set_xlabel("Wall-Clock Time (seconds)")
    ax.set_ylabel("Eval Reward (mean over episodes)")
    ax.set_title("DQN Training Convergence: LightSim vs SUMO")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OVERLEAF / "sample_efficiency.pdf")
    fig.savefig(RESULTS_DIR / "sample_efficiency.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved sample_efficiency.pdf to {OVERLEAF}")
    print(f"  Saved sample_efficiency.png to {RESULTS_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lightsim-only", action="store_true")
    parser.add_argument("--sumo-only", action="store_true")
    parser.add_argument("--skip-figure", action="store_true")
    args = parser.parse_args()

    simulators = ["LightSim", "SUMO"]
    if args.lightsim_only:
        simulators = ["LightSim"]
    elif args.sumo_only:
        simulators = ["SUMO"]

    data = load_checkpoint() if args.resume else {"curves": [], "baselines": {}}
    if data["curves"]:
        print(f"Resuming: {len(data['curves'])} runs already completed")

    print("=" * 70)
    print("Sample Efficiency: DQN Wall-Clock Convergence")
    print(f"Simulators: {simulators}, Seeds: {SEEDS}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}, Eval every {EVAL_INTERVAL:,}")
    print("=" * 70)

    # Compute baselines (fast, LightSim-only)
    if not data["baselines"]:
        print("\nComputing baseline rewards...")
        data["baselines"] = compute_baselines()
        for name, reward in data["baselines"].items():
            print(f"  {name}: {reward:.1f}")
        save_checkpoint(data)

    for simulator in simulators:
        print(f"\n{'#' * 60}")
        print(f"  SIMULATOR: {simulator}")
        print(f"{'#' * 60}")

        for seed in SEEDS:
            if is_completed(data, simulator, seed):
                print(f"\n  [SKIP] seed={seed} (already done)")
                continue

            print(f"\n  Training DQN | seed={seed} | {simulator}")
            try:
                result = train_with_checkpoints(simulator, seed)
                data["curves"].append(result)
                save_checkpoint(data)
                print(f"    Done in {result['total_train_time']:.1f}s")
            except Exception as e:
                print(f"    FAILED: {e}")
                import traceback
                traceback.print_exc()

    # Summary
    print(f"\n\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    for sim in simulators:
        sim_curves = [c for c in data["curves"] if c["simulator"] == sim]
        if not sim_curves:
            continue
        times = [c["total_train_time"] for c in sim_curves]
        final_rewards = [c["curve"][-1]["eval_reward"] for c in sim_curves
                         if c["curve"]]
        print(f"\n  [{sim}]")
        print(f"    Train time: {np.mean(times):.1f} +/- {np.std(times):.1f}s")
        if final_rewards:
            print(f"    Final reward: {np.mean(final_rewards):.1f} +/- "
                  f"{np.std(final_rewards):.1f}")

    # Speed comparison
    ls_times = [c["total_train_time"] for c in data["curves"]
                if c["simulator"] == "LightSim"]
    su_times = [c["total_train_time"] for c in data["curves"]
                if c["simulator"] == "SUMO"]
    if ls_times and su_times:
        speedup = np.mean(su_times) / np.mean(ls_times)
        print(f"\n  Speedup: {speedup:.1f}x faster in LightSim")

    # Save final
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")

    # Generate figure
    if not args.skip_figure and data["curves"]:
        print("\nGenerating figure...")
        generate_figure(data)


if __name__ == "__main__":
    main()
