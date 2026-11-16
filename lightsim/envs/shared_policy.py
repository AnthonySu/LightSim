"""Shared-policy (parameter-sharing) wrapper for PettingZoo parallel envs.

Turns a PettingZoo ``ParallelEnv`` with N agents into a single-agent
Gymnasium env, so one Stable-Baselines3 policy can control every agent.

Each real environment step is split into N "micro-steps": the wrapper shows
one agent's (zero-padded) observation at a time and takes one action at a
time.  After all N actions are collected, the underlying env steps once.

Reward convention (one round late)
----------------------------------
Each micro-step returns the acting agent's *own* reward from the previous
real env step.  An agent's reward for the current step is not known until
all N actions are in and the env has stepped, so every agent's reward is
delivered on its next turn.  The first round of an episode returns 0.0 for
everyone, and the rewards of the final round are not delivered.  The
network-wide mean reward of each real step is reported in
``info["mean_reward"]`` on the micro-step that completes the round (useful
for evaluation).

Observation padding
-------------------
Observations are zero-padded to ``obs_dim``, which by default is the
largest observation size of any agent in the env.  ``pad_obs`` raises
instead of silently truncating an observation that is too long.

Used by ``scripts/multi_agent_rl.py``, ``scripts/train_pretrained.py`` and
``scripts/rl_crossval_grid.py``.
"""

from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np

def pad_obs(obs, target_dim: int) -> np.ndarray:
    """Pad an observation vector to *target_dim* with trailing zeros.

    Raises
    ------
    ValueError
        If *obs* is longer than *target_dim* (it would otherwise lose data).
    """
    obs = np.asarray(obs, dtype=np.float32).ravel()
    if len(obs) > target_dim:
        raise ValueError(
            f"Observation has {len(obs)} values but target_dim is {target_dim}; "
            "refusing to truncate. Increase obs_dim (see max_obs_dim())."
        )
    padded = np.zeros(target_dim, dtype=np.float32)
    padded[: len(obs)] = obs
    return padded


def max_obs_dim(pz_env) -> int:
    """Largest flattened observation size over all agents of *pz_env*."""
    agents = getattr(pz_env, "possible_agents", None) or pz_env.agents
    return max(
        int(np.prod(pz_env.observation_space(agent).shape)) for agent in agents
    )


class SharedPolicyWrapper(gym.Env):
    """Wrap a PettingZoo parallel env as a single-agent Gym env.

    Parameters
    ----------
    env_factory : callable
        Returns a fresh PettingZoo parallel env.
    obs_dim : int, optional
        Length every observation is padded to.  Defaults to
        ``max_obs_dim(env)``, computed from the env itself.
    n_actions : int
        Size of the shared ``Discrete`` action space.
    obs_transform : callable, optional
        Applied to each raw agent observation before padding
        (e.g. normalisation for SUMO observations).
    recreate_on_reset : bool
        If True, build a new env via *env_factory* on every ``reset``
        (needed for sumo-rl); otherwise build it once and reuse it.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_factory: Callable[[], Any],
        obs_dim: int | None = None,
        n_actions: int = 2,
        obs_transform: Callable[[np.ndarray], np.ndarray] | None = None,
        recreate_on_reset: bool = False,
    ):
        super().__init__()
        self._env_factory = env_factory
        self._recreate_on_reset = recreate_on_reset
        self._obs_transform = obs_transform
        self.pz_env = None if recreate_on_reset else env_factory()
        if obs_dim is None:
            if self.pz_env is not None:
                obs_dim = max_obs_dim(self.pz_env)
            else:
                probe = env_factory()
                try:
                    obs_dim = max_obs_dim(probe)
                finally:
                    try:
                        probe.close()
                    except Exception:
                        pass
        self.obs_dim = int(obs_dim)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(n_actions)
        self._agents: list = []
        self._agent_idx = 0
        self._actions: dict = {}
        self._obs: dict = {}
        self._done = False
        self._last_rewards: dict = {}

    # -- helpers -----------------------------------------------------------

    def _pad(self, obs) -> np.ndarray:
        if self._obs_transform is not None:
            obs = self._obs_transform(obs)
        return pad_obs(obs, self.obs_dim)

    def _zero_obs(self) -> np.ndarray:
        return np.zeros(self.obs_dim, dtype=np.float32)

    def _close_pz(self):
        if self.pz_env is not None:
            try:
                self.pz_env.close()
            except Exception:
                pass

    # -- Gym API -----------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._recreate_on_reset:
            self._close_pz()
            self.pz_env = self._env_factory()
        self._obs, _ = self.pz_env.reset(seed=seed)
        self._agents = list(self.pz_env.agents)
        self._agent_idx = 0
        self._actions = {}
        self._done = False
        self._last_rewards = {}
        if not self._agents:
            self._done = True
            return self._zero_obs(), {}
        return self._pad(self._obs[self._agents[0]]), {}

    def step(self, action):
        if self._done:
            return self._zero_obs(), 0.0, False, True, {}

        agent_name = self._agents[self._agent_idx]
        self._actions[agent_name] = int(action)
        self._agent_idx += 1
        # This agent's own reward from the previous real step (one round late).
        reward = self._last_rewards.get(agent_name, 0.0)

        if self._agent_idx < len(self._agents):
            next_obs = self._pad(self._obs[self._agents[self._agent_idx]])
            return next_obs, reward, False, False, {}

        # All agents acted -- step the underlying env once.
        self._obs, rewards, terms, truncs, infos = self.pz_env.step(self._actions)
        self._last_rewards = {a: float(r) for a, r in rewards.items()}
        info = {
            "round_complete": True,
            "mean_reward": float(np.mean(list(rewards.values()))) if rewards else 0.0,
        }

        if not self.pz_env.agents:
            self._done = True
            return self._zero_obs(), reward, False, True, info

        self._agents = list(self.pz_env.agents)
        self._agent_idx = 0
        self._actions = {}
        return self._pad(self._obs[self._agents[0]]), reward, False, False, info

    def close(self):
        self._close_pz()


def make_lightsim_shared_env(
    scenario: str = "grid-4x4-v0",
    max_steps: int = 720,
    obs_dim: int | None = None,
    **env_kwargs,
) -> SharedPolicyWrapper:
    """Shared-policy wrapper around ``lightsim.parallel_env(scenario)``."""
    import lightsim

    return SharedPolicyWrapper(
        lambda: lightsim.parallel_env(scenario, max_steps=max_steps, **env_kwargs),
        obs_dim=obs_dim,
    )
