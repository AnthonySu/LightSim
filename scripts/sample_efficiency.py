"""Sample Efficiency: wall-clock convergence of PPO in LightSim vs SUMO.

Trains PPO on single-intersection in both simulators, logging eval reward
at regular timestep intervals. Shows LightSim reaches target performance
faster in wall-clock time.

Rewards are normalized as % improvement over FixedTime baseline within
each simulator, enabling direct comparison despite different reward scales.

Usage::
    python scripts/sample_efficiency.py
    python scripts/sample_efficiency.py --lightsim-only
    python scripts/sample_efficiency.py --sumo-only
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
SUMO_TIMESTEPS = 50_000     # SUMO is ~10x slower; 50k is enough
EVAL_INTERVAL = 5_000       # evaluate every 5k steps
N_EVAL_EPISODES = 5
SUMO_EVAL_EPISODES = 3      # fewer for SUMO (each episode is slow)
N_SEEDS = 3
SEEDS = [0, 1, 2]

RESULTS_DIR = Path("results")
RESULTS_FILE = RESULTS_DIR / "sample_efficiency.json"
CHECKPOINT_FILE = RESULTS_DIR / "sample_efficiency_checkpoint.json"

PPO_KWARGS = {
    "learning_rate": 3e-4,
    "n_steps": 256,
    "batch_size": 64,
    "n_epochs": 10,
    "policy_kwargs": dict(net_arch=[64, 64]),
}


# ---------------------------------------------------------------------------
# Environment factories
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
def evaluate_model(model, simulator: str, n_episodes: int = None,
                   seed_offset: int = 1000) -> float:
    """Evaluate model and return mean reward."""
    if n_episodes is None:
        n_episodes = SUMO_EVAL_EPISODES if simulator == "SUMO" else N_EVAL_EPISODES
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
def compute_baselines(simulator: str) -> dict:
    """Compute FixedTime baseline reward for a simulator.

    FixedTime = always action 0 (keep current phase / default cycling).
    """
    n_episodes = SUMO_EVAL_EPISODES if simulator == "SUMO" else N_EVAL_EPISODES
    rewards = []
    for ep in range(n_episodes):
        if simulator == "LightSim":
            env = make_lightsim_env(42 + ep)
        else:
            env = make_sumo_env(42 + ep)

        obs, _ = env.reset(seed=42 + ep)
        total_reward = 0.0
        done = False
        step = 0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(0)
            total_reward += reward
            done = terminated or truncated
            step += 1
            if step >= 3600:
                break
        rewards.append(total_reward)
        env.close()

    return {"FixedTime": float(np.mean(rewards))}


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
    """Train PPO with periodic evaluation, return convergence curve."""
    from stable_baselines3 import PPO

    if simulator == "LightSim":
        env = make_lightsim_env(seed)
    else:
        env = make_sumo_env(seed)

    model = PPO("MlpPolicy", env, seed=seed, verbose=0, device="cpu",
                **PPO_KWARGS)

    curve = []
    t_start = time.perf_counter()

    total = SUMO_TIMESTEPS if simulator == "SUMO" else TOTAL_TIMESTEPS
    n_checkpoints = total // EVAL_INTERVAL
    for i in range(n_checkpoints):
        model.learn(total_timesteps=EVAL_INTERVAL, reset_num_timesteps=False)
        wall_time = time.perf_counter() - t_start
        timesteps_done = (i + 1) * EVAL_INTERVAL

        n_eval = SUMO_EVAL_EPISODES if simulator == "SUMO" else N_EVAL_EPISODES
        eval_reward = evaluate_model(model, simulator, n_episodes=n_eval,
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
    """Generate sample efficiency figure: training speed + convergence."""
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    # --- Panel A: Training timesteps vs wall-clock time ---
    for sim, color, label in [("LightSim", BLUE, "LightSim"),
                               ("SUMO", RED, "SUMO")]:
        sim_curves = [c for c in data["curves"] if c["simulator"] == sim]
        if not sim_curves:
            continue

        all_times = []
        all_steps = []
        for c in sim_curves:
            times = [0] + [p["wall_seconds"] for p in c["curve"]]
            steps = [0] + [p["timesteps"] for p in c["curve"]]
            all_times.append(times)
            all_steps.append(steps)

        for times, steps in zip(all_times, all_steps):
            ax1.plot(times, [s / 1000 for s in steps], color=color,
                     alpha=0.2, linewidth=0.8)

        if len(all_times) > 1:
            max_time = max(t[-1] for t in all_times)
            time_grid = np.linspace(0, max_time, 50)
            interp_steps = []
            for times, steps in zip(all_times, all_steps):
                interp_steps.append(np.interp(time_grid, times, steps))
            mean_s = np.mean(interp_steps, axis=0) / 1000
            ax1.plot(time_grid, mean_s, color=color, linewidth=2.0,
                     label=label)
        else:
            ax1.plot(all_times[0], [s / 1000 for s in all_steps[0]],
                     color=color, linewidth=2.0, label=label)

    ax1.set_xlabel("Wall-Clock Time (seconds)")
    ax1.set_ylabel("Training Timesteps (k)")
    ax1.set_title("(a) Training Speed")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Annotate throughput
    for sim, color, y_pos in [("LightSim", BLUE, 0.85), ("SUMO", RED, 0.15)]:
        sim_curves = [c for c in data["curves"] if c["simulator"] == sim]
        if sim_curves:
            total_steps = [c["curve"][-1]["timesteps"] for c in sim_curves]
            total_times = [c["total_train_time"] for c in sim_curves]
            throughput = np.mean(total_steps) / np.mean(total_times)
            ax1.annotate(f"{sim}: {throughput:.0f} steps/s",
                        xy=(0.98, y_pos), xycoords='axes fraction',
                        ha='right', va='center', fontsize=7.5,
                        color=color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', edgecolor=color,
                                  alpha=0.8))

    # --- Panel B: Convergence curves (dual y-axis) ---
    ls_curves = [c for c in data["curves"] if c["simulator"] == "LightSim"]
    su_curves = [c for c in data["curves"] if c["simulator"] == "SUMO"]

    if ls_curves:
        all_times, all_rewards = [], []
        for c in ls_curves:
            times = [p["wall_seconds"] for p in c["curve"]]
            rewards = [p["eval_reward"] / 1000 for p in c["curve"]]
            all_times.append(times)
            all_rewards.append(rewards)

        for times, rewards in zip(all_times, all_rewards):
            ax2.plot(times, rewards, color=BLUE, alpha=0.15, linewidth=0.8)

        if len(all_rewards) > 1:
            max_time = max(t[-1] for t in all_times)
            time_grid = np.linspace(0, max_time, 50)
            interp_r = [np.interp(time_grid, t, r)
                        for t, r in zip(all_times, all_rewards)]
            mean_r = np.mean(interp_r, axis=0)
            std_r = np.std(interp_r, axis=0)
            ax2.plot(time_grid, mean_r, color=BLUE, linewidth=2.0,
                     label='LightSim')
            ax2.fill_between(time_grid, mean_r - std_r, mean_r + std_r,
                             color=BLUE, alpha=0.15)

    ax2.set_xlabel("Wall-Clock Time (seconds)")
    ax2.set_ylabel("LightSim Reward (×1000)", color=BLUE)
    ax2.tick_params(axis='y', labelcolor=BLUE)

    if su_curves:
        ax2r = ax2.twinx()
        all_times, all_rewards = [], []
        for c in su_curves:
            times = [p["wall_seconds"] for p in c["curve"]]
            rewards = [p["eval_reward"] for p in c["curve"]]
            all_times.append(times)
            all_rewards.append(rewards)

        for times, rewards in zip(all_times, all_rewards):
            ax2r.plot(times, rewards, color=RED, alpha=0.15, linewidth=0.8)

        if len(all_rewards) > 1:
            max_time = max(t[-1] for t in all_times)
            time_grid = np.linspace(0, max_time, 50)
            interp_r = [np.interp(time_grid, t, r)
                        for t, r in zip(all_times, all_rewards)]
            mean_r = np.mean(interp_r, axis=0)
            std_r = np.std(interp_r, axis=0)
            ax2r.plot(time_grid, mean_r, color=RED, linewidth=2.0,
                      label='SUMO')
            ax2r.fill_between(time_grid, mean_r - std_r, mean_r + std_r,
                              color=RED, alpha=0.15)

        ax2r.set_ylabel("SUMO Reward", color=RED)
        ax2r.tick_params(axis='y', labelcolor=RED)

    ax2.set_title("(b) Eval Reward Convergence")
    lines1, labels1 = ax2.get_legend_handles_labels()
    if su_curves:
        lines2, labels2 = ax2r.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2,
                   loc="lower right", framealpha=0.9)
    else:
        ax2.legend(loc="lower right", framealpha=0.9)
    ax2.grid(True, alpha=0.3)

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
    print("Sample Efficiency: PPO Wall-Clock Convergence")
    print(f"Simulators: {simulators}, Seeds: {SEEDS}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}, Eval every {EVAL_INTERVAL:,}")
    print("=" * 70)

    # Compute baselines per simulator
    for sim in simulators:
        if sim not in data.get("baselines", {}):
            print(f"\nComputing {sim} baselines...")
            if "baselines" not in data:
                data["baselines"] = {}
            data["baselines"][sim] = compute_baselines(sim)
            for name, reward in data["baselines"][sim].items():
                print(f"  {name}: {reward:.2f}")
            save_checkpoint(data)

    for simulator in simulators:
        print(f"\n{'#' * 60}")
        print(f"  SIMULATOR: {simulator}")
        print(f"{'#' * 60}")

        for seed in SEEDS:
            if is_completed(data, simulator, seed):
                print(f"\n  [SKIP] seed={seed} (already done)")
                continue

            print(f"\n  Training PPO | seed={seed} | {simulator}")
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
