"""Multi-agent RL experiment on grid-4x4 using PettingZoo + independent learners.

Trains independent DQN agents (parameter sharing) on the grid-4x4 scenario
and compares against FixedTime and MaxPressure baselines.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightsim
from lightsim.core.signal import FixedTimeController, MaxPressureController
from lightsim.core.engine import SimulationEngine
from lightsim.core.types import NodeType
from lightsim.benchmarks.scenarios import get_scenario

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def evaluate_multi_agent_baseline(scenario_name, controller, episodes=5, episode_steps=3600):
    """Evaluate a non-RL controller on a multi-intersection scenario."""
    factory = get_scenario(scenario_name)

    all_rewards = []
    all_throughputs = []
    all_queues = []

    for ep in range(episodes):
        network, demand = factory()
        engine = SimulationEngine(
            network=network, dt=1.0,
            controller=controller,
            demand_profiles=demand,
        )
        engine.reset(seed=42 + ep)

        total_reward = 0.0
        for step in range(episode_steps):
            engine.step()
            # Compute total queue across all signalized nodes
            for node in network.nodes.values():
                if node.node_type == NodeType.SIGNALIZED:
                    for link in network.links.values():
                        if link.to_node == node.node_id:
                            total_reward -= engine.get_link_queue(link.link_id)

        metrics = engine.get_network_metrics()
        avg_reward = total_reward / episode_steps
        all_rewards.append(avg_reward)
        all_throughputs.append(metrics["total_exited"])

        # Total queue at end
        total_q = 0.0
        for node in network.nodes.values():
            if node.node_type == NodeType.SIGNALIZED:
                for link in network.links.values():
                    if link.to_node == node.node_id:
                        total_q += engine.get_link_queue(link.link_id)
        all_queues.append(total_q)

    return {
        "avg_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "avg_throughput": float(np.mean(all_throughputs)),
        "avg_queue": float(np.mean(all_queues)),
    }


def evaluate_multi_agent_rl(scenario_name, episodes=10, episode_steps=720):
    """Evaluate independent RL agents using parameter-shared DQN."""
    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        print("SB3 not installed, skipping RL training")
        return None

    # Train a shared DQN policy on a single intersection from the grid.
    # We use an individual grid intersection (grid-4x4 obs space) so the
    # policy can be deployed across all 16 intersections independently.
    print("  Training shared DQN policy on grid-4x4 (single agent view)...", flush=True)

    # Use grid-4x4 single-agent env (controls node_id=0 by default)
    train_env = lightsim.make("grid-4x4-v0", max_steps=720)
    eval_env = lightsim.make("grid-4x4-v0", max_steps=720)

    model = DQN(
        "MlpPolicy", train_env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        seed=42,
        verbose=0,
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=50000)
    train_time = time.perf_counter() - t0
    print(f"  Training done in {train_time:.1f}s", flush=True)

    # Now deploy on grid-4x4 using PettingZoo
    print("  Deploying on grid-4x4 with independent learners...", flush=True)

    env = lightsim.parallel_env("grid-4x4-v0", max_steps=episode_steps)

    all_rewards = []
    all_throughputs = []

    for ep in range(episodes):
        obs, infos = env.reset(seed=42 + ep)
        ep_rewards = {agent: 0.0 for agent in env.possible_agents}

        while env.agents:
            actions = {}
            for agent in env.agents:
                # Use the trained policy to select actions
                action, _ = model.predict(obs[agent], deterministic=True)
                actions[agent] = int(action)

            obs, rewards, terms, truncs, infos = env.step(actions)
            for agent, r in rewards.items():
                ep_rewards[agent] += r

        # Average reward across agents
        avg_reward = np.mean(list(ep_rewards.values())) / episode_steps
        all_rewards.append(avg_reward)

        # Get throughput from last info
        # After truncation, agents list is empty, use infos from last available step
        all_throughputs.append(0)  # We'll estimate from reward

    return {
        "avg_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "train_time": train_time,
        "approach": "independent_DQN_parameter_sharing",
    }


def main():
    print("=" * 60)
    print("Multi-Agent Experiment: grid-4x4")
    print("=" * 60)

    results = {}

    # Baseline: FixedTime
    print("\n1. FixedTime baseline on grid-4x4...")
    ft_result = evaluate_multi_agent_baseline(
        "grid-4x4-v0", FixedTimeController(), episodes=3, episode_steps=3600
    )
    results["FixedTime"] = ft_result
    print(f"   Reward: {ft_result['avg_reward']:.2f} +/- {ft_result['std_reward']:.2f}")
    print(f"   Throughput: {ft_result['avg_throughput']:.0f}")
    print(f"   Queue: {ft_result['avg_queue']:.1f}")

    # Baseline: MaxPressure
    print("\n2. MaxPressure baseline on grid-4x4...")
    mp_result = evaluate_multi_agent_baseline(
        "grid-4x4-v0", MaxPressureController(min_green=5.0), episodes=3, episode_steps=3600
    )
    results["MaxPressure"] = mp_result
    print(f"   Reward: {mp_result['avg_reward']:.2f} +/- {mp_result['std_reward']:.2f}")
    print(f"   Throughput: {mp_result['avg_throughput']:.0f}")
    print(f"   Queue: {mp_result['avg_queue']:.1f}")

    # RL: Independent DQN
    print("\n3. Independent DQN (parameter sharing) on grid-4x4...")
    rl_result = evaluate_multi_agent_rl("grid-4x4-v0", episodes=5, episode_steps=720)
    if rl_result:
        results["IndependentDQN"] = rl_result
        print(f"   Reward: {rl_result['avg_reward']:.2f} +/- {rl_result['std_reward']:.2f}")
        print(f"   Train time: {rl_result['train_time']:.1f}s")

    # Save
    out = RESULTS_DIR / "multi_agent_grid.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
