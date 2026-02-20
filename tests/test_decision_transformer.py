"""Tests for the Decision Transformer module."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lightsim.dt.dataset import (
    Trajectory,
    TrajectoryDataset,
    _compute_rtg,
    _RandomController,
    collect_trajectories,
    save_trajectories,
    load_trajectories,
)
from lightsim.dt.model import DTConfig, DecisionTransformer
from lightsim.dt.train import train_dt, save_dt_model, load_dt_model
from lightsim.dt.controller import DTPolicy, DecisionTransformerController
from lightsim.benchmarks.single_intersection import create_single_intersection
from lightsim.core.engine import SimulationEngine
from lightsim.core.signal import FixedTimeController


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestTrajectory:
    def test_properties(self):
        T, obs_dim = 100, 12
        traj = Trajectory(
            observations=np.random.randn(T, obs_dim).astype(np.float32),
            actions=np.random.randint(0, 4, T).astype(np.int64),
            rewards=np.random.randn(T).astype(np.float32),
            returns_to_go=np.zeros(T, dtype=np.float32),
            timesteps=np.arange(T, dtype=np.int64),
        )
        assert traj.length == T
        assert isinstance(traj.total_return, float)


class TestComputeRTG:
    def test_basic(self):
        rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        rtg = _compute_rtg(rewards)
        np.testing.assert_allclose(rtg, [6.0, 5.0, 3.0])

    def test_negative(self):
        rewards = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
        rtg = _compute_rtg(rewards)
        np.testing.assert_allclose(rtg, [-6.0, -5.0, -3.0])

    def test_single(self):
        rewards = np.array([5.0], dtype=np.float32)
        rtg = _compute_rtg(rewards)
        np.testing.assert_allclose(rtg, [5.0])


class TestTrajectoryDataset:
    def _make_trajectories(self, n=5, T=50, obs_dim=12, act_dim=4):
        trajs = []
        for _ in range(n):
            rewards = np.random.randn(T).astype(np.float32)
            trajs.append(Trajectory(
                observations=np.random.randn(T, obs_dim).astype(np.float32),
                actions=np.random.randint(0, act_dim, T).astype(np.int64),
                rewards=rewards,
                returns_to_go=_compute_rtg(rewards),
                timesteps=np.arange(T, dtype=np.int64),
            ))
        return trajs

    def test_length(self):
        trajs = self._make_trajectories(n=3, T=50)
        ds = TrajectoryDataset(trajs, context_len=20)
        assert len(ds) == 3 * 50  # one entry per timestep per traj

    def test_item_shapes(self):
        trajs = self._make_trajectories(n=2, T=30, obs_dim=8)
        ds = TrajectoryDataset(trajs, context_len=10)
        item = ds[0]
        assert item["observations"].shape == (10, 8)
        assert item["actions"].shape == (10,)
        assert item["returns_to_go"].shape == (10,)
        assert item["timesteps"].shape == (10,)
        assert item["mask"].shape == (10,)

    def test_padding(self):
        """First timestep should be mostly padding."""
        trajs = self._make_trajectories(n=1, T=20)
        ds = TrajectoryDataset(trajs, context_len=10)
        item = ds[0]  # end_pos=0 → seq_len=1, pad_len=9
        assert item["mask"][0].item() == 0.0  # padded
        assert item["mask"][-1].item() == 1.0  # real


class TestCollectTrajectories:
    def test_basic(self):
        trajs = collect_trajectories(
            scenario="single-intersection-v0",
            episodes_per_controller=1,
            max_steps=10,
            seed=42,
            controllers={"FixedTime": FixedTimeController()},
        )
        assert len(trajs) == 1
        assert trajs[0].length == 10
        assert trajs[0].observations.ndim == 2
        assert trajs[0].actions.ndim == 1


class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        T, obs_dim = 50, 12
        trajs = [Trajectory(
            observations=np.random.randn(T, obs_dim).astype(np.float32),
            actions=np.random.randint(0, 4, T).astype(np.int64),
            rewards=np.random.randn(T).astype(np.float32),
            returns_to_go=np.zeros(T, dtype=np.float32),
            timesteps=np.arange(T, dtype=np.int64),
        )]
        path = tmp_path / "trajs.npz"
        save_trajectories(trajs, path)
        loaded = load_trajectories(path)
        assert len(loaded) == 1
        np.testing.assert_array_equal(loaded[0].actions, trajs[0].actions)
        np.testing.assert_allclose(loaded[0].observations, trajs[0].observations)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestDecisionTransformer:
    def _make_config(self, obs_dim=12, act_dim=4):
        return DTConfig(obs_dim=obs_dim, act_dim=act_dim, context_len=10)

    def test_forward_shape(self):
        config = self._make_config()
        model = DecisionTransformer(config)
        B, K = 2, 10
        obs = torch.randn(B, K, 12)
        act = torch.randint(0, 4, (B, K))
        rtg = torch.randn(B, K)
        ts = torch.arange(K).unsqueeze(0).expand(B, -1)
        mask = torch.ones(B, K)

        logits = model(obs, act, rtg, ts, mask)
        assert logits.shape == (B, K, 4)

    def test_forward_no_mask(self):
        config = self._make_config()
        model = DecisionTransformer(config)
        B, K = 1, 5
        obs = torch.randn(B, K, 12)
        act = torch.randint(0, 4, (B, K))
        rtg = torch.randn(B, K)
        ts = torch.arange(K).unsqueeze(0)

        logits = model(obs, act, rtg, ts)
        assert logits.shape == (B, K, 4)

    def test_param_count(self):
        config = DTConfig(obs_dim=12, act_dim=4, hidden_dim=64,
                          n_layers=3, n_heads=4, ffn_dim=256)
        model = DecisionTransformer(config)
        n_params = sum(p.numel() for p in model.parameters())
        # Should be roughly 414K as planned
        assert 300_000 < n_params < 600_000, f"Param count: {n_params}"

    def test_causal_mask(self):
        config = self._make_config()
        model = DecisionTransformer(config)
        mask = model._build_causal_mask(3, torch.device("cpu"))
        # Shape: (9, 9) — 2D additive mask
        assert mask.shape == (9, 9)
        # Position 0 (timestep 0) should not attend to position 3 (timestep 1)
        assert mask[0, 3].item() < -1e8
        # Position 3 (timestep 1) should attend to position 0 (timestep 0)
        assert mask[3, 0].item() == 0.0


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------

class TestTrainDT:
    def _make_trajectories(self, n=5, T=30):
        trajs = []
        for _ in range(n):
            rewards = np.random.randn(T).astype(np.float32)
            trajs.append(Trajectory(
                observations=np.random.randn(T, 12).astype(np.float32),
                actions=np.random.randint(0, 4, T).astype(np.int64),
                rewards=rewards,
                returns_to_go=_compute_rtg(rewards),
                timesteps=np.arange(T, dtype=np.int64),
            ))
        return trajs

    def test_train_runs(self):
        trajs = self._make_trajectories()
        model, losses, rtg_stats = train_dt(
            trajs, epochs=3, batch_size=16, context_len=10, verbose=False,
        )
        assert len(losses) == 3
        assert all(isinstance(l, float) for l in losses)
        assert "mean" in rtg_stats and "std" in rtg_stats

    def test_loss_decreases(self):
        np.random.seed(42)
        trajs = self._make_trajectories(n=10, T=50)
        model, losses, _ = train_dt(
            trajs, epochs=10, batch_size=32, context_len=10,
            verbose=False, lr=1e-3,
        )
        # Loss should generally decrease (compare first vs last)
        assert losses[-1] < losses[0]

    def test_save_load(self, tmp_path):
        trajs = self._make_trajectories()
        model, _, rtg_stats = train_dt(trajs, epochs=2, batch_size=16, context_len=10, verbose=False)

        path = tmp_path / "model.pt"
        save_dt_model(model, path, rtg_stats=rtg_stats)
        loaded, loaded_stats = load_dt_model(path)

        assert loaded_stats["mean"] == rtg_stats["mean"]
        assert loaded_stats["std"] == rtg_stats["std"]

        # Compare predictions
        obs = torch.randn(1, 10, 12)
        act = torch.randint(0, 4, (1, 10))
        rtg = torch.randn(1, 10)
        ts = torch.arange(10).unsqueeze(0)
        mask = torch.ones(1, 10)

        model.eval()
        with torch.no_grad():
            logits1 = model(obs, act, rtg, ts, mask)
            logits2 = loaded(obs, act, rtg, ts, mask)
        torch.testing.assert_close(logits1, logits2)


# ---------------------------------------------------------------------------
# Policy / Controller tests
# ---------------------------------------------------------------------------

class TestDTPolicy:
    def _make_policy(self):
        config = DTConfig(obs_dim=12, act_dim=4, context_len=10)
        model = DecisionTransformer(config)
        return DTPolicy(model, target_return=-100.0)

    def test_predict(self):
        policy = self._make_policy()
        obs = np.random.randn(12).astype(np.float32)
        action = policy.predict(obs)
        assert 0 <= action < 4

    def test_sequential_predict(self):
        policy = self._make_policy()
        for _ in range(15):  # Exceed context_len
            obs = np.random.randn(12).astype(np.float32)
            action = policy.predict(obs)
            policy.update_rtg(-1.0)
            assert 0 <= action < 4

    def test_reset(self):
        policy = self._make_policy()
        for _ in range(5):
            policy.predict(np.random.randn(12).astype(np.float32))
        policy.reset(target_return=-50.0)
        assert policy.target_return == -50.0
        assert len(policy._obs_buffer) == 0


class TestDecisionTransformerController:
    def test_runs_as_controller(self):
        """DT controller should run the simulation without errors."""
        # Single intersection: obs_dim=10, act_dim=2
        config = DTConfig(obs_dim=10, act_dim=2, context_len=10)
        model = DecisionTransformer(config)

        network, demand = create_single_intersection()
        controller = DecisionTransformerController(
            model, target_return=-100.0, sim_steps_per_action=5,
        )

        engine = SimulationEngine(
            network=network, dt=1.0,
            controller=controller,
            demand_profiles=demand,
        )
        controller.set_engine(engine)
        engine.reset(seed=42)

        for _ in range(100):
            engine.step()

        metrics = engine.get_network_metrics()
        assert metrics["total_entered"] > 0
        assert metrics["time"] > 0


# ---------------------------------------------------------------------------
# Smoke test: collect → train → evaluate
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_smoke(self):
        """Collect 2 trajectories, train 3 epochs, predict — should work."""
        trajs = collect_trajectories(
            scenario="single-intersection-v0",
            controllers={"FixedTime": FixedTimeController()},
            episodes_per_controller=2,
            max_steps=20,
            seed=42,
        )
        assert len(trajs) == 2

        model, losses, rtg_stats = train_dt(
            trajs, epochs=3, batch_size=8, context_len=10, verbose=False,
        )
        assert len(losses) == 3

        # Use model as controller
        network, demand = create_single_intersection()
        controller = DecisionTransformerController(
            model, target_return=-50.0, rtg_stats=rtg_stats,
            sim_steps_per_action=5,
        )
        engine = SimulationEngine(
            network=network, dt=1.0,
            controller=controller,
            demand_profiles=demand,
        )
        controller.set_engine(engine)
        engine.reset(seed=0)

        for _ in range(50):
            engine.step()

        metrics = engine.get_network_metrics()
        assert metrics["total_entered"] > 0
