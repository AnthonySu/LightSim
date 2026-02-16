"""Oversaturated demand experiment for LightSim single-intersection scenario.

Tests three controllers (FixedTime, MaxPressure, DQN) under three demand
levels (v/c = 0.7, 1.0, 1.3) to study performance degradation under
oversaturated conditions.

Results saved to: ../results/oversaturated_experiment.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure lightsim is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightsim
from lightsim import DemandProfile, LinkID, NodeID
from lightsim.benchmarks.single_intersection import create_single_intersection
from lightsim.core.engine import SimulationEngine
from lightsim.core.signal import FixedTimeController, MaxPressureController
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEMAND_LEVELS = {
    "vc_0.7": {"label": "v/c ~ 0.7 (undersaturated)", "ns": 0.35, "ew": 0.25},
    "vc_1.0": {"label": "v/c ~ 1.0 (at capacity)",     "ns": 0.50, "ew": 0.35},
    "vc_1.3": {"label": "v/c ~ 1.3 (oversaturated)",   "ns": 0.65, "ew": 0.45},
}

DQN_TRAIN_STEPS = 50_000
EVAL_EPISODES = 5
EVAL_MAX_STEPS = 720       # RL steps per eval episode
SIM_STEPS_PER_ACTION = 5   # env default
DT = 1.0                   # seconds
SEED = 42

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_FILE = RESULTS_DIR / "oversaturated_experiment.json"


# ---------------------------------------------------------------------------
# Helper: build demand profiles with custom flow rates
# ---------------------------------------------------------------------------
def make_demand(ns_rate: float, ew_rate: float) -> list:
    """Create demand profiles for the 4 approaches of the single intersection."""
    return [
        DemandProfile(LinkID(0), [0.0], [ns_rate]),  # North inbound
        DemandProfile(LinkID(1), [0.0], [ns_rate]),  # South inbound
        DemandProfile(LinkID(2), [0.0], [ew_rate]),  # East inbound
        DemandProfile(LinkID(3), [0.0], [ew_rate]),  # West inbound
    ]


# ---------------------------------------------------------------------------
# Helper: make a LightSimEnv with custom demand
# ---------------------------------------------------------------------------
def make_env(ns_rate: float, ew_rate: float, max_steps: int = 3600):
    """Create a LightSimEnv with custom demand for DQN training/eval."""
    from lightsim.benchmarks.scenarios import get_scenario

    factory = get_scenario("single-intersection-v0")
    network, _ = factory()  # ignore default demand
    demand = make_demand(ns_rate, ew_rate)

    return lightsim.LightSimEnv(
        network=network,
        dt=DT,
        sim_steps_per_action=SIM_STEPS_PER_ACTION,
        max_steps=max_steps,
        obs_builder="default",
        action_handler="phase_select",
        reward_fn="queue",
        demand_profiles=demand,
    )


# ---------------------------------------------------------------------------
# Evaluate FixedTime controller via direct SimulationEngine
# ---------------------------------------------------------------------------
def evaluate_fixed_time(ns_rate: float, ew_rate: float) -> dict:
    """Run FixedTime controller and collect metrics over eval episodes."""
    network, _ = create_single_intersection()
    demand = make_demand(ns_rate, ew_rate)
    controller = FixedTimeController()

    total_steps = EVAL_MAX_STEPS * SIM_STEPS_PER_ACTION  # total sim steps
    all_rewards = []
    all_throughputs = []
    all_final_queues = []

    for ep in range(EVAL_EPISODES):
        engine = SimulationEngine(
            network=network, dt=DT,
            controller=controller,
            demand_profiles=demand,
        )
        engine.reset(seed=SEED + ep)

        ep_reward = 0.0
        for step in range(total_steps):
            engine.step()
            # Accumulate queue-based reward every SIM_STEPS_PER_ACTION steps
            if (step + 1) % SIM_STEPS_PER_ACTION == 0:
                total_queue = 0.0
                for link in network.links.values():
                    if link.to_node == NodeID(0):
                        total_queue += engine.get_link_queue(link.link_id)
                ep_reward += -total_queue

        throughput = engine.state.total_exited
        final_queue = 0.0
        for link in network.links.values():
            if link.to_node == NodeID(0):
                final_queue += engine.get_link_queue(link.link_id)

        all_rewards.append(ep_reward)
        all_throughputs.append(throughput)
        all_final_queues.append(final_queue)

    return {
        "avg_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "throughput": float(np.mean(all_throughputs)),
        "std_throughput": float(np.std(all_throughputs)),
        "final_queue": float(np.mean(all_final_queues)),
        "std_final_queue": float(np.std(all_final_queues)),
    }


# ---------------------------------------------------------------------------
# Evaluate MaxPressure controller via direct SimulationEngine
# ---------------------------------------------------------------------------
def evaluate_max_pressure(ns_rate: float, ew_rate: float) -> dict:
    """Run MaxPressure controller and collect metrics over eval episodes."""
    network, _ = create_single_intersection()
    demand = make_demand(ns_rate, ew_rate)
    controller = MaxPressureController(min_green=5.0)

    total_steps = EVAL_MAX_STEPS * SIM_STEPS_PER_ACTION
    all_rewards = []
    all_throughputs = []
    all_final_queues = []

    for ep in range(EVAL_EPISODES):
        engine = SimulationEngine(
            network=network, dt=DT,
            controller=controller,
            demand_profiles=demand,
        )
        engine.reset(seed=SEED + ep)

        ep_reward = 0.0
        for step in range(total_steps):
            engine.step()
            if (step + 1) % SIM_STEPS_PER_ACTION == 0:
                total_queue = 0.0
                for link in network.links.values():
                    if link.to_node == NodeID(0):
                        total_queue += engine.get_link_queue(link.link_id)
                ep_reward += -total_queue

        throughput = engine.state.total_exited
        final_queue = 0.0
        for link in network.links.values():
            if link.to_node == NodeID(0):
                final_queue += engine.get_link_queue(link.link_id)

        all_rewards.append(ep_reward)
        all_throughputs.append(throughput)
        all_final_queues.append(final_queue)

    return {
        "avg_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "throughput": float(np.mean(all_throughputs)),
        "std_throughput": float(np.std(all_throughputs)),
        "final_queue": float(np.mean(all_final_queues)),
        "std_final_queue": float(np.std(all_final_queues)),
    }


# ---------------------------------------------------------------------------
# Train & evaluate DQN agent
# ---------------------------------------------------------------------------
class EpisodeRewardTracker(BaseCallback):
    """Tracks episode rewards during training."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is not None:
                self.episode_rewards.append(ep_info["r"])
        return True


def train_and_evaluate_dqn(ns_rate: float, ew_rate: float) -> dict:
    """Train a fresh DQN agent and evaluate it."""
    train_env = make_env(ns_rate, ew_rate, max_steps=EVAL_MAX_STEPS)

    model = DQN(
        "MlpPolicy",
        train_env,
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

    print(f"    Training DQN for {DQN_TRAIN_STEPS:,} timesteps...")
    t0 = time.time()
    tracker = EpisodeRewardTracker()
    model.learn(total_timesteps=DQN_TRAIN_STEPS, callback=tracker)
    train_time = time.time() - t0
    print(f"    Training complete in {train_time:.1f}s "
          f"({len(tracker.episode_rewards)} episodes)")

    eval_env = make_env(ns_rate, ew_rate, max_steps=EVAL_MAX_STEPS)
    all_rewards = []
    all_throughputs = []
    all_final_queues = []

    for ep in range(EVAL_EPISODES):
        obs, info = eval_env.reset(seed=SEED + ep)
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(int(action))
            ep_reward += reward
            done = terminated or truncated

        throughput = info["total_exited"]
        final_queue = 0.0
        for link in eval_env.network.links.values():
            if link.to_node == eval_env.agent_node:
                final_queue += eval_env.engine.get_link_queue(link.link_id)

        all_rewards.append(ep_reward)
        all_throughputs.append(throughput)
        all_final_queues.append(final_queue)

    return {
        "avg_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "throughput": float(np.mean(all_throughputs)),
        "std_throughput": float(np.std(all_throughputs)),
        "final_queue": float(np.mean(all_final_queues)),
        "std_final_queue": float(np.std(all_final_queues)),
        "train_time_seconds": round(train_time, 1),
        "train_episodes": len(tracker.episode_rewards),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": "oversaturated_demand",
        "config": {
            "dqn_train_steps": DQN_TRAIN_STEPS,
            "eval_episodes": EVAL_EPISODES,
            "eval_max_steps": EVAL_MAX_STEPS,
            "sim_steps_per_action": SIM_STEPS_PER_ACTION,
            "dt": DT,
            "seed": SEED,
        },
        "demand_levels": {},
    }

    print("=" * 74)
    print("OVERSATURATED DEMAND EXPERIMENT")
    print("  Scenario: single-intersection (custom demand)")
    print("  Controllers: FixedTime, MaxPressure, DQN")
    print(f"  DQN training: {DQN_TRAIN_STEPS:,} steps")
    print(f"  Eval: {EVAL_EPISODES} episodes x {EVAL_MAX_STEPS} RL steps")
    print(f"  Seed: {SEED}")
    print("=" * 74)

    for level_key, level_cfg in DEMAND_LEVELS.items():
        ns_rate = level_cfg["ns"]
        ew_rate = level_cfg["ew"]

        sep = "=" * 74
        print()
        print(sep)
        print("  DEMAND LEVEL: " + level_cfg["label"])
        print(f"  NS = {ns_rate} veh/s, EW = {ew_rate} veh/s")
        print(sep)

        level_results = {
            "label": level_cfg["label"],
            "ns_rate": ns_rate,
            "ew_rate": ew_rate,
            "controllers": {},
        }

        # --- FixedTime ---
        print()
        print("  [1/3] FixedTime controller...")
        t0 = time.time()
        ft_results = evaluate_fixed_time(ns_rate, ew_rate)
        ft_time = time.time() - t0
        print(f"    avg_reward:  {ft_results['avg_reward']:.2f}")
        print(f"    throughput:  {ft_results['throughput']:.1f}")
        print(f"    final_queue: {ft_results['final_queue']:.1f}")
        print(f"    time: {ft_time:.1f}s")
        level_results["controllers"]["FixedTime"] = ft_results

        # --- MaxPressure ---
        print()
        print("  [2/3] MaxPressure controller...")
        t0 = time.time()
        mp_results = evaluate_max_pressure(ns_rate, ew_rate)
        mp_time = time.time() - t0
        print(f"    avg_reward:  {mp_results['avg_reward']:.2f}")
        print(f"    throughput:  {mp_results['throughput']:.1f}")
        print(f"    final_queue: {mp_results['final_queue']:.1f}")
        print(f"    time: {mp_time:.1f}s")
        level_results["controllers"]["MaxPressure"] = mp_results

        # --- DQN ---
        print()
        print("  [3/3] DQN (train + eval)...")
        dqn_results = train_and_evaluate_dqn(ns_rate, ew_rate)
        print(f"    avg_reward:  {dqn_results['avg_reward']:.2f}")
        print(f"    throughput:  {dqn_results['throughput']:.1f}")
        print(f"    final_queue: {dqn_results['final_queue']:.1f}")
        level_results["controllers"]["DQN"] = dqn_results

        results["demand_levels"][level_key] = level_results

        # Save intermediate results
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print()
        print("  (intermediate results saved)")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    sep = "=" * 74
    print()
    print()
    print(sep)
    print("SUMMARY")
    print(sep)
    print(f"{'Demand':<20} {'Controller':<14} {'Avg Reward':>12} {'Throughput':>12} {'Final Queue':>12}")
    print("-" * 74)

    for level_key, level_cfg in DEMAND_LEVELS.items():
        level_data = results["demand_levels"][level_key]
        for ctrl_name in ["FixedTime", "MaxPressure", "DQN"]:
            ctrl = level_data["controllers"][ctrl_name]
            label = level_cfg["label"].split("(")[0].strip()
            ar = ctrl["avg_reward"]
            tp = ctrl["throughput"]
            fq = ctrl["final_queue"]
            print(f"{label:<20} {ctrl_name:<14} {ar:>12.2f} {tp:>12.1f} {fq:>12.1f}")
        print()

    print(sep)
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
