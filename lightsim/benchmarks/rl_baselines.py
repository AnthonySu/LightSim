"""RL baseline benchmarks: MaxPressure, DQN, PPO on LightSim scenarios.

Evaluates control policies on the single-intersection and grid-4x4 scenarios,
reporting average reward, throughput, and delay.

Usage::

    python -m lightsim.benchmarks.rl_baselines                  # MaxPressure only
    python -m lightsim.benchmarks.rl_baselines --train-rl       # + DQN/PPO (needs SB3)
    python -m lightsim.benchmarks.rl_baselines --timesteps 50000
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import numpy as np

import lightsim
from ..core.demand import DemandProfile
from ..core.engine import SimulationEngine
from ..core.signal import FixedTimeController, MaxPressureController
from ..core.types import NodeID, NodeType


@dataclass
class BaselineResult:
    policy: str
    scenario: str
    episodes: int
    avg_reward: float
    avg_throughput: float
    avg_delay: float
    avg_vehicles: float
    wall_time: float


# ---------------------------------------------------------------------------
# Evaluate a non-RL controller by running the raw engine
# ---------------------------------------------------------------------------

def evaluate_controller(
    scenario: str,
    controller,
    episodes: int = 5,
    episode_steps: int = 3600,
    dt: float = 1.0,
) -> BaselineResult:
    """Evaluate a signal controller on a scenario."""
    from ..benchmarks.scenarios import get_scenario
    from ..utils.metrics import compute_link_delay

    factory = get_scenario(scenario)

    rewards_all = []
    throughputs = []
    delays = []
    vehicles_all = []

    t0 = time.perf_counter()

    for ep in range(episodes):
        network, demand = factory()
        engine = SimulationEngine(
            network=network, dt=dt,
            controller=controller,
            demand_profiles=demand,
        )
        engine.reset(seed=ep)

        ep_reward = 0.0
        for step in range(episode_steps):
            engine.step()
            # Queue-based reward (negative queue)
            total_queue = 0.0
            for link in network.links.values():
                if any(
                    n.node_type == NodeType.SIGNALIZED
                    for n in [network.nodes[link.to_node]]
                    if n.node_id in network.nodes
                ):
                    total_queue += engine.get_link_queue(link.link_id)
            ep_reward -= total_queue

        metrics = engine.get_network_metrics()
        rewards_all.append(ep_reward / episode_steps)
        throughputs.append(metrics["total_exited"])
        vehicles_all.append(metrics["total_vehicles"])

        # Average delay across incoming links
        total_delay = 0.0
        n_links = 0
        for link in network.links.values():
            to_node = network.nodes.get(link.to_node)
            if to_node and to_node.node_type == NodeType.SIGNALIZED:
                total_delay += compute_link_delay(engine, link.link_id)
                n_links += 1
        delays.append(total_delay / max(n_links, 1))

    wall = time.perf_counter() - t0
    policy_name = type(controller).__name__

    return BaselineResult(
        policy=policy_name,
        scenario=scenario,
        episodes=episodes,
        avg_reward=float(np.mean(rewards_all)),
        avg_throughput=float(np.mean(throughputs)),
        avg_delay=float(np.mean(delays)),
        avg_vehicles=float(np.mean(vehicles_all)),
        wall_time=wall,
    )


# ---------------------------------------------------------------------------
# Evaluate an RL policy via the Gymnasium env
# ---------------------------------------------------------------------------

def evaluate_rl_policy(
    model,
    scenario: str,
    episodes: int = 5,
    max_steps: int = 720,
) -> BaselineResult:
    """Evaluate a trained SB3 model on a LightSim env."""
    rewards_all = []
    throughputs = []
    delays = []
    vehicles_all = []

    t0 = time.perf_counter()

    for ep in range(episodes):
        env = lightsim.make(scenario, max_steps=max_steps)
        obs, info = env.reset(seed=ep + 100)
        ep_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        rewards_all.append(ep_reward / max_steps)
        throughputs.append(info.get("total_exited", 0))
        vehicles_all.append(info.get("total_vehicles", 0))
        delays.append(0.0)  # would need engine access for exact delay
        env.close()

    wall = time.perf_counter() - t0

    return BaselineResult(
        policy=type(model).__name__,
        scenario=scenario,
        episodes=episodes,
        avg_reward=float(np.mean(rewards_all)),
        avg_throughput=float(np.mean(throughputs)),
        avg_delay=float(np.mean(delays)),
        avg_vehicles=float(np.mean(vehicles_all)),
        wall_time=wall,
    )


# ---------------------------------------------------------------------------
# Train + evaluate DQN / PPO
# ---------------------------------------------------------------------------

def train_and_evaluate_sb3(
    algo: str,
    scenario: str,
    total_timesteps: int = 20000,
    eval_episodes: int = 5,
    max_steps: int = 720,
) -> BaselineResult:
    """Train an SB3 agent and evaluate it."""
    try:
        from stable_baselines3 import DQN, PPO
    except ImportError:
        raise ImportError(
            "stable-baselines3 is required for RL baselines. "
            "Install with: pip install stable-baselines3"
        )

    env = lightsim.make(scenario, max_steps=max_steps)

    t0 = time.perf_counter()

    if algo.upper() == "DQN":
        model = DQN(
            "MlpPolicy", env,
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            exploration_fraction=0.3,
            verbose=0,
        )
    elif algo.upper() == "PPO":
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=0,
        )
    else:
        raise ValueError(f"Unknown algo: {algo}")

    model.learn(total_timesteps=total_timesteps)
    env.close()

    result = evaluate_rl_policy(model, scenario, episodes=eval_episodes, max_steps=max_steps)
    result.policy = algo.upper()
    result.wall_time = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_results(results: list[BaselineResult]) -> None:
    header = (
        f"{'Policy':<20} {'Scenario':<25} {'Ep':>3} "
        f"{'Avg Reward':>11} {'Throughput':>11} {'Delay':>8} "
        f"{'Vehicles':>9} {'Wall(s)':>8}"
    )
    print("=" * len(header))
    print("LightSim RL Baselines")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.policy:<20} {r.scenario:<25} {r.episodes:>3} "
            f"{r.avg_reward:>11.2f} {r.avg_throughput:>11.1f} {r.avg_delay:>8.2f} "
            f"{r.avg_vehicles:>9.1f} {r.wall_time:>8.2f}"
        )
    print("-" * len(header))


def main():
    parser = argparse.ArgumentParser(description="LightSim RL Baselines")
    parser.add_argument("--scenarios", nargs="+",
                        default=["single-intersection-v0"],
                        help="Scenarios to benchmark")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-steps", type=int, default=3600)
    parser.add_argument("--train-rl", action="store_true",
                        help="Also train and evaluate DQN/PPO (needs SB3)")
    parser.add_argument("--timesteps", type=int, default=20000,
                        help="SB3 training timesteps")
    args = parser.parse_args()

    results: list[BaselineResult] = []

    for scenario in args.scenarios:
        print(f"\nEvaluating on: {scenario}")

        # Fixed-time
        print("  Running FixedTime...")
        r = evaluate_controller(
            scenario, FixedTimeController(),
            episodes=args.episodes, episode_steps=args.episode_steps,
        )
        results.append(r)

        # MaxPressure
        print("  Running MaxPressure...")
        r = evaluate_controller(
            scenario, MaxPressureController(min_green=5.0),
            episodes=args.episodes, episode_steps=args.episode_steps,
        )
        results.append(r)

        # RL baselines
        if args.train_rl:
            for algo in ["DQN", "PPO"]:
                print(f"  Training {algo}...")
                try:
                    r = train_and_evaluate_sb3(
                        algo, scenario,
                        total_timesteps=args.timesteps,
                        eval_episodes=args.episodes,
                    )
                    results.append(r)
                except ImportError as e:
                    print(f"    Skipped {algo}: {e}")

    print()
    print_results(results)


if __name__ == "__main__":
    main()
