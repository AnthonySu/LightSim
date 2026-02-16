"""Tests for single-agent Gymnasium environment."""

import numpy as np
import pytest

import lightsim


class TestLightSimEnv:
    """Test the single-agent Gymnasium env."""

    def test_make_and_reset(self):
        """Env should be creatable and resettable."""
        env = lightsim.make("single-intersection-v0")
        obs, info = env.reset(seed=42)
        assert obs is not None
        assert isinstance(info, dict)
        assert obs.shape == env.observation_space.shape
        env.close()

    def test_step(self):
        """Env should accept actions and return valid transitions."""
        env = lightsim.make("single-intersection-v0")
        obs, info = env.reset(seed=42)

        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == env.observation_space.shape
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

        env.close()

    def test_truncation(self):
        """Env should truncate after max_steps."""
        env = lightsim.make("single-intersection-v0", max_steps=10)
        env.reset(seed=42)

        truncated = False
        for i in range(20):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if truncated:
                break

        assert truncated, "Environment did not truncate after max_steps"
        env.close()

    def test_observation_in_bounds(self):
        """Observations should be within the observation space bounds."""
        env = lightsim.make("single-intersection-v0")
        obs, _ = env.reset(seed=42)

        for _ in range(20):
            assert env.observation_space.contains(obs), \
                f"Observation out of bounds: {obs}"
            obs, _, _, truncated, _ = env.step(env.action_space.sample())
            if truncated:
                break

        env.close()

    def test_different_obs_builders(self):
        """Should work with all registered observation builders."""
        for obs_name in ["default", "pressure", "full_density"]:
            env = lightsim.make(
                "single-intersection-v0",
                obs_builder=obs_name,
                max_steps=5,
            )
            obs, _ = env.reset(seed=42)
            assert obs is not None
            obs, _, _, _, _ = env.step(env.action_space.sample())
            assert obs is not None
            env.close()

    def test_different_action_handlers(self):
        """Should work with all registered action handlers."""
        for action_name in ["phase_select", "next_or_stay"]:
            env = lightsim.make(
                "single-intersection-v0",
                action_handler=action_name,
                max_steps=5,
            )
            obs, _ = env.reset(seed=42)
            obs, _, _, _, _ = env.step(env.action_space.sample())
            assert obs is not None
            env.close()

    def test_different_reward_fns(self):
        """Should work with all registered reward functions."""
        for reward_name in ["queue", "pressure", "delay", "throughput"]:
            env = lightsim.make(
                "single-intersection-v0",
                reward_fn=reward_name,
                max_steps=5,
            )
            obs, _ = env.reset(seed=42)
            obs, reward, _, _, _ = env.step(env.action_space.sample())
            assert isinstance(reward, float)
            env.close()

    def test_gymnasium_env_checker(self):
        """Run gymnasium's built-in env checker."""
        from gymnasium.utils.env_checker import check_env
        env = lightsim.make("single-intersection-v0", max_steps=50)
        # check_env will raise if anything is wrong
        check_env(env, skip_render_check=True)
        env.close()
