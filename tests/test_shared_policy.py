"""Tests for lightsim.envs.shared_policy (issue #1)."""

import numpy as np
import pytest

import lightsim
from lightsim.envs.shared_policy import (
    SharedPolicyWrapper,
    make_lightsim_shared_env,
    max_obs_dim,
    pad_obs,
)

SCENARIO = "grid-4x4-v0"


# -- pad_obs ---------------------------------------------------------------

def test_pad_obs_pads_with_zeros():
    out = pad_obs(np.array([0.1, 0.2]), 4)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0.1, 0.2, 0.0, 0.0], rtol=1e-6)


def test_pad_obs_exact_length_unchanged():
    obs = np.arange(5, dtype=np.float32)
    np.testing.assert_array_equal(pad_obs(obs, 5), obs)


def test_pad_obs_refuses_to_truncate():
    with pytest.raises(ValueError, match="refusing to truncate"):
        pad_obs(np.zeros(15), 14)


# -- observation size --------------------------------------------------------

def test_obs_dim_computed_from_env():
    env = make_lightsim_shared_env(SCENARIO, max_steps=5)
    pz = lightsim.parallel_env(SCENARIO, max_steps=5)
    obs, _ = pz.reset(seed=0)
    expected = max(len(o) for o in obs.values())
    assert env.obs_dim == expected == max_obs_dim(pz)
    assert env.observation_space.shape == (expected,)


def test_too_small_obs_dim_raises_instead_of_truncating():
    env = make_lightsim_shared_env(SCENARIO, max_steps=5, obs_dim=5)
    with pytest.raises(ValueError):
        env.reset(seed=0)


# -- reward attribution --------------------------------------------------------

def test_each_micro_step_gets_own_reward_one_round_late():
    """Micro-step for agent i in round k returns agent i's reward of round k-1."""
    max_steps, n_rounds = 60, 40
    wrapper = SharedPolicyWrapper(
        lambda: lightsim.parallel_env(SCENARIO, max_steps=max_steps))
    ref = lightsim.parallel_env(SCENARIO, max_steps=max_steps)

    wrapper.reset(seed=7)
    ref.reset(seed=7)
    agents = list(ref.agents)
    rng = np.random.default_rng(0)

    prev_ref_rewards = {a: 0.0 for a in agents}
    for _ in range(n_rounds):
        actions = {a: int(rng.integers(2)) for a in agents}
        for i, agent in enumerate(agents):
            _, reward, _, _, info = wrapper.step(actions[agent])
            assert reward == pytest.approx(prev_ref_rewards[agent])
            assert info.get("round_complete", False) == (i == len(agents) - 1)
        _, ref_rewards, _, _, _ = ref.step(actions)
        assert info["mean_reward"] == pytest.approx(
            float(np.mean(list(ref_rewards.values()))))
        prev_ref_rewards = ref_rewards

    # Rewards are genuinely per-agent, not one shared aggregate.
    assert len({round(r, 9) for r in prev_ref_rewards.values()}) > 1


def test_episode_terminates_with_padded_zero_obs():
    max_steps = 3
    env = make_lightsim_shared_env(SCENARIO, max_steps=max_steps)
    env.reset(seed=0)
    n_agents = len(env.pz_env.possible_agents)
    truncated = False
    for _ in range(n_agents * max_steps):
        obs, _, _, truncated, _ = env.step(0)
    assert truncated
    assert obs.shape == (env.obs_dim,) and not obs.any()
