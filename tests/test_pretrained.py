"""Tests for pretrained model loading."""

from __future__ import annotations

import pytest

from lightsim.pretrained import list_pretrained, load_pretrained, WEIGHTS_DIR


class TestListPretrained:
    def test_returns_list(self):
        result = list_pretrained()
        assert isinstance(result, list)

    def test_contains_expected_models(self):
        result = list_pretrained()
        # These should be in the weights/ directory
        expected = [
            "dqn_single_intersection",
            "ppo_single_intersection",
            "dqn_single_intersection_pressure",
            "ppo_single_intersection_pressure",
        ]
        for name in expected:
            assert name in result, f"Missing pretrained model: {name}"

    def test_all_entries_are_strings(self):
        result = list_pretrained()
        for name in result:
            assert isinstance(name, str)

    def test_weights_dir_exists(self):
        assert WEIGHTS_DIR.exists(), f"Weights directory not found: {WEIGHTS_DIR}"


try:
    import stable_baselines3  # noqa: F401
    _has_sb3 = True
except ImportError:
    _has_sb3 = False


@pytest.mark.skipif(not _has_sb3, reason="stable-baselines3 not installed")
class TestLoadPretrained:
    def test_load_dqn(self):
        import lightsim
        env = lightsim.make("single-intersection-v0", max_steps=720)
        model = load_pretrained("dqn_single_intersection", env=env)
        # Should be able to predict
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert env.action_space.contains(action)
        env.close()

    def test_load_ppo(self):
        import lightsim
        env = lightsim.make("single-intersection-v0", max_steps=720)
        model = load_pretrained("ppo_single_intersection", env=env)
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert env.action_space.contains(action)
        env.close()

    def test_load_invalid_name_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_pretrained("nonexistent_model_12345")

    def test_load_all_available(self):
        """All listed models should be loadable."""
        import lightsim
        env = lightsim.make("single-intersection-v0", max_steps=720)
        available = list_pretrained()
        for name in available:
            if "grid4x4" in name:
                continue  # multi-agent models need different env
            model = load_pretrained(name, env=env)
            obs, _ = env.reset()
            action, _ = model.predict(obs, deterministic=True)
            assert env.action_space.contains(action)
        env.close()
