"""Trajectory collection and PyTorch Dataset for Decision Transformer training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import lightsim
from lightsim.core.signal import (
    FixedTimeController,
    MaxPressureController,
    SOTLController,
    SignalController,
)


@dataclass
class Trajectory:
    """A single episode trajectory.

    Attributes
    ----------
    observations : np.ndarray, shape (T, obs_dim)
    actions : np.ndarray, shape (T,), int64
    rewards : np.ndarray, shape (T,)
    returns_to_go : np.ndarray, shape (T,)
    timesteps : np.ndarray, shape (T,), int64
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    returns_to_go: np.ndarray
    timesteps: np.ndarray

    @property
    def length(self) -> int:
        return len(self.actions)

    @property
    def total_return(self) -> float:
        return float(self.rewards.sum())


def _compute_rtg(rewards: np.ndarray) -> np.ndarray:
    """Compute undiscounted return-to-go: RTG[t] = sum(rewards[t:])."""
    rtg = np.zeros_like(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running += rewards[t]
        rtg[t] = running
    return rtg


class _RandomController(SignalController):
    """Random phase selection for data collection diversity."""

    def __init__(self, rng: np.random.Generator | None = None):
        self._rng = rng or np.random.default_rng()

    def get_phase_index(self, node_id, state, net, density) -> int:
        n_phases = net.n_phases_per_node.get(node_id, 1)
        return int(self._rng.integers(0, n_phases))


def collect_trajectories(
    scenario: str = "single-intersection-v0",
    controllers: dict[str, SignalController] | None = None,
    episodes_per_controller: int = 20,
    max_steps: int = 720,
    sim_steps_per_action: int = 5,
    reward_fn: str = "queue",
    seed: int = 42,
    **scenario_kwargs,
) -> list[Trajectory]:
    """Collect trajectories from multiple controllers through LightSimEnv.

    Parameters
    ----------
    scenario : str
        Scenario name for ``lightsim.make()``.
    controllers : dict[str, SignalController] | None
        Named controllers to collect from. Defaults to Random, FixedTime,
        MaxPressure, SOTL.
    episodes_per_controller : int
        Episodes to collect per controller.
    max_steps : int
        Episode length in RL steps.
    sim_steps_per_action : int
        Simulation steps between actions.
    reward_fn : str
        Reward function name.
    seed : int
        Base random seed.
    **scenario_kwargs
        Passed to ``lightsim.make()``.

    Returns
    -------
    list[Trajectory]
    """
    rng = np.random.default_rng(seed)

    if controllers is None:
        controllers = {
            "Random": _RandomController(rng=np.random.default_rng(rng.integers(0, 2**31))),
            "FixedTime": FixedTimeController(),
            "MaxPressure": MaxPressureController(min_green=5.0),
            "SOTL": SOTLController(),
        }

    # Create a temporary env to discover spaces
    env = lightsim.make(
        scenario,
        max_steps=max_steps,
        sim_steps_per_action=sim_steps_per_action,
        reward_fn=reward_fn,
        **scenario_kwargs,
    )
    n_actions = env.action_space.n
    agent_node = env.agent_node
    env.close()

    trajectories: list[Trajectory] = []

    for ctrl_name, controller in controllers.items():
        for ep in range(episodes_per_controller):
            ep_seed = int(rng.integers(0, 2**31))

            # Create env with RL controller (we'll override actions)
            env = lightsim.make(
                scenario,
                max_steps=max_steps,
                sim_steps_per_action=sim_steps_per_action,
                reward_fn=reward_fn,
                **scenario_kwargs,
            )
            obs, info = env.reset(seed=ep_seed)

            observations = [obs]
            actions = []
            rewards = []

            for t in range(max_steps):
                # Query the non-RL controller for what action it would take
                sig_state = env.engine.signal_manager.states.get(
                    agent_node,
                    lightsim.core.signal.SignalState(),
                )
                action = controller.get_phase_index(
                    agent_node,
                    sig_state,
                    env.engine.net,
                    env.engine.state.density,
                )
                action = action % n_actions

                obs, reward, terminated, truncated, info = env.step(action)
                observations.append(obs)
                actions.append(action)
                rewards.append(reward)

                if terminated or truncated:
                    break

            env.close()

            T = len(actions)
            if T == 0:
                continue

            obs_arr = np.array(observations[:T], dtype=np.float32)
            act_arr = np.array(actions, dtype=np.int64)
            rew_arr = np.array(rewards, dtype=np.float32)
            rtg_arr = _compute_rtg(rew_arr)
            ts_arr = np.arange(T, dtype=np.int64)

            trajectories.append(Trajectory(
                observations=obs_arr,
                actions=act_arr,
                rewards=rew_arr,
                returns_to_go=rtg_arr,
                timesteps=ts_arr,
            ))

    return trajectories


def save_trajectories(trajectories: list[Trajectory], path: str | Path) -> None:
    """Save trajectories to a .npz file."""
    path = Path(path)
    data = {}
    data["n_trajectories"] = np.array(len(trajectories))
    for i, traj in enumerate(trajectories):
        data[f"obs_{i}"] = traj.observations
        data[f"act_{i}"] = traj.actions
        data[f"rew_{i}"] = traj.rewards
        data[f"rtg_{i}"] = traj.returns_to_go
        data[f"ts_{i}"] = traj.timesteps
    np.savez_compressed(str(path), **data)


def load_trajectories(path: str | Path) -> list[Trajectory]:
    """Load trajectories from a .npz file."""
    data = np.load(str(path))
    n = int(data["n_trajectories"])
    trajectories = []
    for i in range(n):
        trajectories.append(Trajectory(
            observations=data[f"obs_{i}"],
            actions=data[f"act_{i}"],
            rewards=data[f"rew_{i}"],
            returns_to_go=data[f"rtg_{i}"],
            timesteps=data[f"ts_{i}"],
        ))
    return trajectories


try:
    import torch
    from torch.utils.data import Dataset

    class TrajectoryDataset(Dataset):
        """PyTorch Dataset that samples fixed-length subsequences from trajectories.

        RTG values are normalized to zero mean and unit variance to prevent
        the large magnitude of cumulative rewards from dominating embeddings.

        Parameters
        ----------
        trajectories : list[Trajectory]
            Collected trajectories.
        context_len : int
            Length of subsequences to sample. Shorter sequences are left-padded.
        """

        def __init__(self, trajectories: list[Trajectory], context_len: int = 20):
            self.context_len = context_len
            self.trajectories = trajectories

            # Compute RTG normalization statistics across all trajectories
            all_rtg = np.concatenate([t.returns_to_go for t in trajectories])
            self.rtg_mean = float(all_rtg.mean())
            self.rtg_std = float(all_rtg.std()) + 1e-8

            # Build index: (traj_idx, start_timestep) for all valid windows
            self._indices: list[tuple[int, int]] = []
            for i, traj in enumerate(trajectories):
                # Allow starting from any position (left-pad short sequences)
                for t in range(traj.length):
                    self._indices.append((i, t))

        def normalize_rtg(self, rtg: np.ndarray) -> np.ndarray:
            """Normalize RTG values to zero mean, unit variance."""
            return (rtg - self.rtg_mean) / self.rtg_std

        def __len__(self) -> int:
            return len(self._indices)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            traj_idx, end_pos = self._indices[idx]
            traj = self.trajectories[traj_idx]
            K = self.context_len

            # Extract window ending at end_pos (inclusive)
            start = max(0, end_pos - K + 1)
            seq_len = end_pos - start + 1

            obs = traj.observations[start:end_pos + 1]
            act = traj.actions[start:end_pos + 1]
            rtg = self.normalize_rtg(traj.returns_to_go[start:end_pos + 1])
            ts = traj.timesteps[start:end_pos + 1]

            # Left-pad if shorter than context_len
            pad_len = K - seq_len
            obs_dim = obs.shape[1]

            if pad_len > 0:
                obs = np.concatenate([np.zeros((pad_len, obs_dim), dtype=np.float32), obs])
                act = np.concatenate([np.zeros(pad_len, dtype=np.int64), act])
                rtg = np.concatenate([np.zeros(pad_len, dtype=np.float32), rtg])
                ts = np.concatenate([np.zeros(pad_len, dtype=np.int64), ts])

            # Attention mask: 1 for real tokens, 0 for padding
            mask = np.zeros(K, dtype=np.float32)
            mask[pad_len:] = 1.0

            return {
                "observations": torch.tensor(obs, dtype=torch.float32),
                "actions": torch.tensor(act, dtype=torch.long),
                "returns_to_go": torch.tensor(rtg, dtype=torch.float32),
                "timesteps": torch.tensor(ts, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.float32),
            }

except ImportError:
    pass
