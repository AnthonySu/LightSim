"""Decision Transformer inference: DTPolicy and DecisionTransformerController."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from ..core.engine import SimulationEngine
from ..core.network import CompiledNetwork
from ..core.signal import SignalController, SignalState
from ..core.types import NodeID
from ..envs.observations import DefaultObservation
from ..envs.rewards import QueueReward
from .model import DecisionTransformer


class DTPolicy:
    """Decision Transformer policy for inference.

    Maintains a rolling buffer of (obs, action, RTG) and predicts the next
    action using the trained DT model. Compatible with Gymnasium-style
    predict() interface.

    Parameters
    ----------
    model : DecisionTransformer
        Trained DT model.
    target_return : float
        Target cumulative return for conditioning.
    rtg_stats : dict | None
        RTG normalization stats (``"mean"``, ``"std"``) from training.
        If provided, target_return and RTG buffer values are normalized
        using these stats before being fed to the model.
    context_len : int
        Maximum context window. Defaults to model config.
    device : str
        Torch device.
    """

    def __init__(
        self,
        model: DecisionTransformer,
        target_return: float,
        rtg_stats: dict | None = None,
        context_len: int | None = None,
        device: str = "cpu",
    ):
        self.model = model
        self.model.eval()
        self.device = device
        self.context_len = context_len or model.config.context_len
        self.target_return = target_return
        self.rtg_mean = rtg_stats["mean"] if rtg_stats else 0.0
        self.rtg_std = rtg_stats["std"] if rtg_stats else 1.0

        self._obs_buffer: list[np.ndarray] = []
        self._act_buffer: list[int] = []
        self._rtg_buffer: list[float] = []
        self._timestep = 0
        self._current_rtg = target_return

    def _normalize_rtg(self, rtg: np.ndarray) -> np.ndarray:
        """Normalize RTG to match training distribution."""
        return (rtg - self.rtg_mean) / self.rtg_std

    def reset(self, target_return: float | None = None) -> None:
        """Reset for a new episode."""
        self._obs_buffer = []
        self._act_buffer = []
        self._rtg_buffer = []
        self._timestep = 0
        if target_return is not None:
            self.target_return = target_return
        self._current_rtg = self.target_return

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Predict the next action given current observation.

        Parameters
        ----------
        obs : np.ndarray
            Current observation vector.
        deterministic : bool
            If True, use argmax. If False, sample from softmax.

        Returns
        -------
        action : int
        """
        self._obs_buffer.append(obs.astype(np.float32))
        self._rtg_buffer.append(self._current_rtg)

        K = self.context_len

        # Build input tensors from buffer
        T = len(self._obs_buffer)
        start = max(0, T - K)
        seq_len = T - start

        obs_seq = np.array(self._obs_buffer[start:], dtype=np.float32)
        rtg_seq = self._normalize_rtg(
            np.array(self._rtg_buffer[start:], dtype=np.float32)
        )

        # Actions: use previous actions (shifted by 1). For the first step,
        # use action 0 as placeholder.
        if len(self._act_buffer) == 0:
            act_seq = np.zeros(seq_len, dtype=np.int64)
        else:
            # Actions from start to current (pad first with 0)
            prev_acts = [0] + self._act_buffer[start:]
            act_seq = np.array(prev_acts[:seq_len], dtype=np.int64)

        ts_seq = np.arange(
            max(0, self._timestep - seq_len + 1),
            self._timestep + 1,
            dtype=np.int64,
        )

        # Left-pad to context_len
        pad_len = K - seq_len
        obs_dim = obs_seq.shape[1]

        if pad_len > 0:
            obs_seq = np.concatenate([np.zeros((pad_len, obs_dim), dtype=np.float32), obs_seq])
            act_seq = np.concatenate([np.zeros(pad_len, dtype=np.int64), act_seq])
            rtg_seq = np.concatenate([np.zeros(pad_len, dtype=np.float32), rtg_seq])
            ts_seq = np.concatenate([np.zeros(pad_len, dtype=np.int64), ts_seq])

        # To tensors (batch dim)
        obs_t = torch.tensor(obs_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        act_t = torch.tensor(act_seq, dtype=torch.long).unsqueeze(0).to(self.device)
        rtg_t = torch.tensor(rtg_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        ts_t = torch.tensor(ts_seq, dtype=torch.long).unsqueeze(0).to(self.device)

        mask = torch.zeros(1, K, dtype=torch.float32, device=self.device)
        mask[0, pad_len:] = 1.0

        with torch.no_grad():
            logits = self.model(obs_t, act_t, rtg_t, ts_t, mask)  # (1, K, act_dim)

        # Take the last valid position's logits
        last_logits = logits[0, -1]  # (act_dim,)

        if deterministic:
            action = int(last_logits.argmax().item())
        else:
            probs = torch.softmax(last_logits, dim=-1)
            action = int(torch.multinomial(probs, 1).item())

        self._act_buffer.append(action)
        self._timestep += 1

        return action

    def update_rtg(self, reward: float) -> None:
        """Update return-to-go after receiving a reward."""
        self._current_rtg -= reward


class DecisionTransformerController(SignalController):
    """SignalController that uses a trained Decision Transformer.

    Wraps a DTPolicy and internally computes observations and rewards
    from the engine state, making it usable as a drop-in replacement
    for any other SignalController.

    Parameters
    ----------
    model : DecisionTransformer
        Trained DT model.
    target_return : float
        Target return for conditioning.
    sim_steps_per_action : int
        How often to re-predict (should match training env).
    device : str
        Torch device.
    """

    def __init__(
        self,
        model: DecisionTransformer,
        target_return: float,
        rtg_stats: dict | None = None,
        sim_steps_per_action: int = 5,
        device: str = "cpu",
    ):
        self._policy = DTPolicy(model, target_return, rtg_stats=rtg_stats, device=device)
        self.sim_steps_per_action = sim_steps_per_action
        self._obs_builder = DefaultObservation()
        self._reward_fn = QueueReward()
        self._engine: SimulationEngine | None = None
        self._agent_node: NodeID | None = None
        self._step_counter = 0
        self._cached_action: int = 0
        self._initialized = False

    def set_engine(self, engine: SimulationEngine, agent_node: NodeID | None = None) -> None:
        """Attach to an engine. Must be called before use."""
        self._engine = engine
        if agent_node is not None:
            self._agent_node = agent_node
        else:
            # Find first signalized node
            from ..core.types import NodeType
            for node in engine.net.node_phases:
                self._agent_node = node
                break
        self._policy.reset()
        self._step_counter = 0
        self._initialized = False

    def get_phase_index(
        self,
        node_id: NodeID,
        state: SignalState,
        net: CompiledNetwork,
        density: np.ndarray,
    ) -> int:
        if self._engine is None:
            return state.current_phase_idx

        # Only predict for the agent node
        if node_id != self._agent_node:
            return state.current_phase_idx

        self._step_counter += 1

        # Re-predict every sim_steps_per_action calls
        if not self._initialized or self._step_counter >= self.sim_steps_per_action:
            self._step_counter = 0

            # Compute obs
            obs = self._obs_builder.observe(self._engine, node_id)

            # Compute reward (for RTG update) — skip on first call
            if self._initialized:
                reward = self._reward_fn.compute(self._engine, node_id)
                self._policy.update_rtg(reward)

            # Predict
            self._cached_action = self._policy.predict(obs, deterministic=True)
            self._initialized = True

        return self._cached_action


def _pad_obs(obs: np.ndarray, target_dim: int) -> np.ndarray:
    """Zero-pad obs to *target_dim*."""
    if len(obs) >= target_dim:
        return obs[:target_dim].astype(np.float32)
    padded = np.zeros(target_dim, dtype=np.float32)
    padded[: len(obs)] = obs
    return padded


class MultiAgentDTPolicy:
    """Decentralized multi-agent DT policy with parameter sharing.

    One :class:`DTPolicy` per agent, all sharing the same underlying model.
    Each agent maintains its own independent rolling context buffer.

    Parameters
    ----------
    model : DecisionTransformer
        Trained DT model (shared across all agents).
    agent_names : list[str]
        Agent identifiers (e.g. ``["signal_7", "signal_8", ...]``).
    target_return : float
        Default target return for each agent.
    rtg_stats : dict | None
        RTG normalization stats from training.
    pad_obs_dim : int
        Observation dimension after zero-padding.
    context_len : int | None
        Context window length.  Defaults to model config.
    device : str
        Torch device.
    """

    def __init__(
        self,
        model: DecisionTransformer,
        agent_names: list[str],
        target_return: float,
        rtg_stats: dict | None = None,
        pad_obs_dim: int = 14,
        context_len: int | None = None,
        device: str = "cpu",
    ):
        self.model = model
        self.agent_names = list(agent_names)
        self.target_return = target_return
        self.pad_obs_dim = pad_obs_dim

        # One DTPolicy per agent, all sharing the same model object
        self._policies: dict[str, DTPolicy] = {}
        for name in self.agent_names:
            self._policies[name] = DTPolicy(
                model=model,
                target_return=target_return,
                rtg_stats=rtg_stats,
                context_len=context_len,
                device=device,
            )

    def reset(self, target_return: float | None = None) -> None:
        """Reset all agent policies for a new episode."""
        tr = target_return if target_return is not None else self.target_return
        for policy in self._policies.values():
            policy.reset(target_return=tr)

    def predict(
        self,
        observations: dict[str, np.ndarray],
        deterministic: bool = True,
    ) -> dict[str, int]:
        """Predict actions for all agents.

        Parameters
        ----------
        observations : dict[str, np.ndarray]
            Observation per agent (will be zero-padded to ``pad_obs_dim``).
        deterministic : bool
            If True, use argmax; otherwise sample.

        Returns
        -------
        dict[str, int]
            Action per agent.
        """
        actions = {}
        for agent, obs in observations.items():
            if agent in self._policies:
                padded = _pad_obs(obs, self.pad_obs_dim)
                actions[agent] = self._policies[agent].predict(
                    padded, deterministic=deterministic,
                )
        return actions

    def update_rtg(self, rewards: dict[str, float]) -> None:
        """Update return-to-go for each agent after receiving rewards."""
        for agent, reward in rewards.items():
            if agent in self._policies:
                self._policies[agent].update_rtg(reward)
