"""Tests for multi-agent PettingZoo environment."""

import numpy as np
import pytest

import lightsim


class TestLightSimParallelEnv:
    """Test the multi-agent PettingZoo ParallelEnv."""

    @pytest.fixture
    def env(self):
        try:
            env = lightsim.parallel_env("grid-4x4-v0", max_steps=20)
        except ImportError:
            pytest.skip("pettingzoo not installed")
        yield env

    def test_create_and_reset(self, env):
        """Env should be creatable and resettable."""
        observations, infos = env.reset(seed=42)
        assert len(observations) > 0
        assert len(infos) > 0
        for agent in env.agents:
            assert agent in observations
            assert observations[agent] is not None

    def test_step(self, env):
        """Env should accept actions and return valid transitions."""
        env.reset(seed=42)

        for _ in range(5):
            if not env.agents:
                break
            actions = {
                agent: env.action_space(agent).sample()
                for agent in env.agents
            }
            observations, rewards, terminations, truncations, infos = env.step(actions)
            for agent in observations:
                assert observations[agent] is not None
                assert isinstance(rewards[agent], float)

    def test_truncation(self, env):
        """All agents should be removed after max_steps."""
        env.reset(seed=42)

        for _ in range(30):
            if not env.agents:
                break
            actions = {
                agent: env.action_space(agent).sample()
                for agent in env.agents
            }
            env.step(actions)

        assert len(env.agents) == 0, "Agents should be empty after truncation"

    def test_pettingzoo_api(self):
        """Run PettingZoo's parallel API test."""
        try:
            import pettingzoo
            from pettingzoo.test import parallel_api_test
        except ImportError:
            pytest.skip("pettingzoo not installed")

        env = lightsim.parallel_env("grid-4x4-v0", max_steps=20)
        parallel_api_test(env, num_cycles=5)
