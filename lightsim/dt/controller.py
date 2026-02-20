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
    context_len : int
        Maximum context window. Defaults to model config.
    device : str
        Torch device.
    """

    def __init__(
        self,
        model: DecisionTransformer,
        target_return: float,
        context_len: int | None = None,
        device: str = "cpu",
    ):
        self.model = model
        self.model.eval()
        self.device = device
        self.context_len = context_len or model.config.context_len
        self.target_return = target_return

        self._obs_buffer: list[np.ndarray] = []
        self._act_buffer: list[int] = []
        self._rtg_buffer: list[float] = []
        self._timestep = 0
        self._current_rtg = target_return

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
        rtg_seq = np.array(self._rtg_buffer[start:], dtype=np.float32)

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
        sim_steps_per_action: int = 5,
        device: str = "cpu",
    ):
        self._policy = DTPolicy(model, target_return, device=device)
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
