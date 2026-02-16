"""Train and evaluate RL on grid-4x4 multi-agent scenario.

Challenge: grid-4x4 has heterogeneous observation spaces across agents:
  - Corner intersections (4 nodes): 14 dimensions (2 phases + 6 incoming links)
  - Edge intersections (8 nodes): 12 dimensions (2 phases + 5 incoming links)
  - Center intersections (4 nodes): 10 dimensions (2 phases + 4 incoming links)

SB3 cannot handle variable-size observations directly, so this script:
  1. Creates a Gymnasium wrapper that pads all observations to max dim (14)
     and iterates over all 16 agents per environment step (parameter sharing).
  2. Trains a single DQN policy on this wrapper.
  3. Evaluates the trained policy on the PettingZoo parallel env for 5 episodes.
  4. Compares against FixedTime and MaxPressure baselines.
  5. Saves results to results/multi_agent_rl.json.
"""

import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightsim
from lightsim.core.signal import FixedTimeController, MaxPressureController
from lightsim.core.engine import SimulationEngine
from lightsim.core.types import NodeType
from lightsim.benchmarks.scenarios import get_scenario

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MAX_OBS_DIM = 14


def pad_obs(obs, target_dim=MAX_OBS_DIM):
    """Pad an observation vector to target_dim with trailing zeros."""
    if len(obs) >= target_dim:
        return obs[:target_dim].astype(np.float32)
    padded = np.zeros(target_dim, dtype=np.float32)
    padded[:len(obs)] = obs
    return padded


class SharedPolicyMultiAgentWrapper(gym.Env):
    """Wraps PettingZoo grid-4x4 as single-agent Gym env with parameter sharing.

    Each environment step cycles through ALL 16 agents:
    - The wrapper presents one agent observation (padded) at a time.
    - The policy outputs one action for that agent.
    - After all 16 actions are collected, the underlying PettingZoo env steps.
    - Reward returned is the mean reward across all agents for that step.

    SB3 sees 16 micro-steps per real env step. Only the 16th micro-step
    advances the simulation and returns the real mean reward.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps=720):
        super().__init__()
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
        first_obs = pad_obs(self._obs[self._agents[0]])
        return first_obs, {}

    def step(self, action):
        if self._done:
            return pad_obs(np.zeros(10)), 0.0, False, True, {}

        agent_name = self._agents[self._agent_idx]
        self._actions[agent_name] = int(action)
        self._agent_idx += 1

        if self._agent_idx < len(self._agents):
            next_obs = pad_obs(self._obs[self._agents[self._agent_idx]])
            return next_obs, 0.0, False, False, {}

        # All 16 actions collected -- step the PettingZoo env
        self._obs, rewards, terms, truncs, infos = self.pz_env.step(self._actions)
        mean_reward = float(np.mean(list(rewards.values())))

        if not self.pz_env.agents:
            self._done = True
            return pad_obs(np.zeros(10)), mean_reward, False, True, {}

        self._agents = list(self.pz_env.agents)
        self._agent_idx = 0
        self._actions = {}
        first_obs = pad_obs(self._obs[self._agents[0]])
        return first_obs, mean_reward, False, False, {}


def evaluate_baseline(controller, episodes=5, episode_steps=3600):
    """Evaluate a non-RL controller on grid-4x4."""
    factory = get_scenario("grid-4x4-v0")
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
            for node in network.nodes.values():
                if node.node_type == NodeType.SIGNALIZED:
                    for link in network.links.values():
                        if link.to_node == node.node_id:
                            total_reward -= engine.get_link_queue(link.link_id)

        metrics = engine.get_network_metrics()
        avg_reward = total_reward / episode_steps
        all_rewards.append(avg_reward)
        all_throughputs.append(metrics["total_exited"])

        total_q = 0.0
        for node in network.nodes.values():
            if node.node_type == NodeType.SIGNALIZED:
                for link in network.links.values():
                    if link.to_node == node.node_id:
                        total_q += engine.get_link_queue(link.link_id)
        all_queues.append(total_q)

    return {
        "avg_reward_per_step": float(np.mean(all_rewards)),
        "std_reward_per_step": float(np.std(all_rewards)),
        "avg_throughput": float(np.mean(all_throughputs)),
        "avg_final_queue": float(np.mean(all_queues)),
    }


def evaluate_rl_pettingzoo(model, episodes=5, episode_steps=720):
    """Deploy trained DQN on full PettingZoo grid-4x4 with obs padding."""
    env = lightsim.parallel_env("grid-4x4-v0", max_steps=episode_steps)

    all_rewards = []
    all_throughputs = []

    for ep in range(episodes):
        obs, infos = env.reset(seed=100 + ep)
        ep_rewards = {agent: 0.0 for agent in env.possible_agents}
        last_metrics = None

        while env.agents:
            actions = {}
            for agent in env.agents:
                padded = pad_obs(obs[agent])
                action, _ = model.predict(padded, deterministic=True)
                actions[agent] = int(action)

            obs, rewards, terms, truncs, infos = env.step(actions)
            for agent, r in rewards.items():
                ep_rewards[agent] += r
            if infos:
                last_metrics = next(iter(infos.values()))

        mean_ep_reward = float(np.mean(list(ep_rewards.values()))) / episode_steps
        all_rewards.append(mean_ep_reward)
        if last_metrics:
            all_throughputs.append(last_metrics.get("total_exited", 0.0))

    return {
        "avg_reward_per_step": float(np.mean(all_rewards)),
        "std_reward_per_step": float(np.std(all_rewards)),
        "avg_throughput": float(np.mean(all_throughputs)) if all_throughputs else 0.0,
        "episodes": episodes,
        "episode_steps": episode_steps,
    }


def main():
    print("======================================================================")
    print("  Multi-Agent RL on grid-4x4 with Heterogeneous Observations")
    print("======================================================================")

    # Step 1: Verify observation space heterogeneity
    print()
    print("[1/6] Checking observation space heterogeneity...")
    pz_env = lightsim.parallel_env("grid-4x4-v0", max_steps=10)
    obs_dims = {}
    for agent in pz_env.possible_agents:
        dim = pz_env.observation_space(agent).shape[0]
        obs_dims[agent] = dim
    dim_counts = {}
    for d in obs_dims.values():
        dim_counts[d] = dim_counts.get(d, 0) + 1
    print("  Observation dimensions: {}".format(dict(sorted(dim_counts.items()))))
    print("  Max obs dim: {} (used for padding)".format(max(obs_dims.values())))
    print("  Total agents: {}".format(len(pz_env.possible_agents)))
    for agent in sorted(obs_dims.keys()):
        print("    {}: obs_dim={}".format(agent, obs_dims[agent]))

    # Step 2: Train DQN on the wrapper
    print()
    print("[2/6] Training DQN with parameter-shared multi-agent wrapper...")
    try:
        from stable_baselines3 import DQN
    except ImportError:
        print("ERROR: stable-baselines3 is required.")
        sys.exit(1)

    train_env = SharedPolicyMultiAgentWrapper(max_steps=720)
    obs, info = train_env.reset(seed=0)
    print("  Wrapper obs shape: {}, dtype: {}".format(obs.shape, obs.dtype))
    print("  Wrapper action space: {}".format(train_env.action_space))

    TOTAL_TIMESTEPS = 100_000
    print("  Training for {:,} timesteps...".format(TOTAL_TIMESTEPS))

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=2_000,
        batch_size=128,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        target_update_interval=500,
        train_freq=4,
        seed=42,
        verbose=0,
    )

    t0 = time.perf_counter()
    checkpoint_interval = TOTAL_TIMESTEPS // 5
    for i in range(5):
        model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False)
        elapsed = time.perf_counter() - t0
        steps_done = (i + 1) * checkpoint_interval
        print("    {:>7,} / {:,} steps  ({:.1f}s elapsed)".format(
            steps_done, TOTAL_TIMESTEPS, elapsed))

    train_time = time.perf_counter() - t0
    print("  Training complete in {:.1f}s".format(train_time))

    # Step 3: Check training obs dim
    print()
    print("[3/6] Verifying trained policy input dimension...")
    test_obs = train_env.observation_space.sample()
    action, _ = model.predict(test_obs, deterministic=True)
    print("  Policy input dim: {}".format(test_obs.shape[0]))
    print("  Policy output action: {} (from Discrete(2))".format(action))
    for native_dim in [10, 12, 14]:
        raw = np.random.rand(native_dim).astype(np.float32)
        padded = pad_obs(raw)
        act, _ = model.predict(padded, deterministic=True)
        print("  Native dim {} -> padded to {}, action={}".format(
            native_dim, len(padded), act))

    # Step 4: Evaluate RL on PettingZoo env
    print()
    print("[4/6] Evaluating RL on grid-4x4 PettingZoo env (5 episodes)...")
    rl_results = evaluate_rl_pettingzoo(model, episodes=5, episode_steps=720)
    print("  RL avg reward/step: {:.2f} +/- {:.2f}".format(
        rl_results["avg_reward_per_step"], rl_results["std_reward_per_step"]))
    if rl_results["avg_throughput"] > 0:
        print("  RL avg throughput: {:.0f}".format(rl_results["avg_throughput"]))

    # Step 5: Baselines
    print()
    print("[5/6] Running baselines (FixedTime and MaxPressure)...")
    print("  Running 720-step episodes for direct comparison with RL.")
    print()
    print("  FixedTime baseline (720-step episodes)...")
    ft_results = evaluate_baseline(FixedTimeController(), episodes=5, episode_steps=720)
    print("    Avg reward/step: {:.2f} +/- {:.2f}".format(
        ft_results["avg_reward_per_step"], ft_results["std_reward_per_step"]))
    print()
    print("  MaxPressure baseline (720-step episodes)...")
    mp_results = evaluate_baseline(
        MaxPressureController(min_green=5.0), episodes=5, episode_steps=720
    )
    print("    Avg reward/step: {:.2f} +/- {:.2f}".format(
        mp_results["avg_reward_per_step"], mp_results["std_reward_per_step"]))

    # Step 6: Summary and save
    print()
    print("[6/6] Summary comparison (reward/step, higher = better)...")
    print("  {:<30} {:>15} {:>10}".format("Method", "Reward/Step", "Std"))
    print("  " + "-" * 55)
    print("  {:<30} {:>15.2f} {:>10.2f}".format(
        "FixedTime", ft_results["avg_reward_per_step"],
        ft_results["std_reward_per_step"]))
    print("  {:<30} {:>15.2f} {:>10.2f}".format(
        "MaxPressure", mp_results["avg_reward_per_step"],
        mp_results["std_reward_per_step"]))
    print("  {:<30} {:>15.2f} {:>10.2f}".format(
        "DQN (param-shared, padded)", rl_results["avg_reward_per_step"],
        rl_results["std_reward_per_step"]))

    methods = {
        "FixedTime": ft_results["avg_reward_per_step"],
        "MaxPressure": mp_results["avg_reward_per_step"],
        "DQN_shared": rl_results["avg_reward_per_step"],
    }
    best = max(methods, key=methods.get)
    print()
    print("  Best method: {} (reward/step = {:.2f})".format(best, methods[best]))
    print()
    print("  Reference (3600-step episodes from prior experiment):")
    print("    FixedTime: -3615.41 reward/step")
    print("    MaxPressure: -3945.88 reward/step")

    results = {
        "scenario": "grid-4x4-v0",
        "num_agents": 16,
        "obs_dimensions": {"corner": 14, "edge": 12, "center": 10},
        "padding_strategy": "zero-pad to max dim (14)",
        "training": {
            "method": "DQN with MlpPolicy",
            "wrapper": "SharedPolicyMultiAgentWrapper",
            "total_timesteps": TOTAL_TIMESTEPS,
            "training_time_seconds": round(train_time, 1),
            "hyperparameters": {
                "learning_rate": 1e-3,
                "buffer_size": 50_000,
                "learning_starts": 2_000,
                "batch_size": 128,
                "gamma": 0.99,
                "exploration_fraction": 0.3,
                "exploration_final_eps": 0.05,
                "target_update_interval": 500,
                "train_freq": 4,
            },
        },
        "evaluation": {
            "episode_steps": 720,
            "episodes": 5,
            "FixedTime": ft_results,
            "MaxPressure": mp_results,
            "DQN_shared_policy": {
                "avg_reward_per_step": rl_results["avg_reward_per_step"],
                "std_reward_per_step": rl_results["std_reward_per_step"],
                "avg_throughput": rl_results["avg_throughput"],
                "episodes": rl_results["episodes"],
                "episode_steps": rl_results["episode_steps"],
                "approach": "independent_learners_parameter_sharing",
                "obs_padding": "zero-pad to 14 dims",
            },
        },
        "reference_baselines_3600steps": {
            "FixedTime": -3615.41,
            "MaxPressure": -3945.88,
        },
    }

    out_path = RESULTS_DIR / "multi_agent_rl.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print()
    print("  Results saved to: {}".format(out_path))
    print()
    print("Done.")


if __name__ == "__main__":
    main()

