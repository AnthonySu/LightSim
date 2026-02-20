"""Observation builders for LightSim RL environments.

Registry pattern: use ``@register_obs("name")`` to register, then retrieve
with ``get_obs_builder("name")``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np

from ..core.engine import SimulationEngine
from ..core.types import FLOAT, NodeID

_OBS_REGISTRY: dict[str, type[ObservationBuilder]] = {}


def register_obs(name: str):
    """Decorator to register an observation builder."""
    def wrapper(cls):
        _OBS_REGISTRY[name] = cls
        return cls
    return wrapper


def get_obs_builder(name: str, **kwargs) -> ObservationBuilder:
    """Retrieve a registered observation builder by name."""
    if name not in _OBS_REGISTRY:
        raise KeyError(f"Unknown observation builder: {name!r}. "
                       f"Available: {list(_OBS_REGISTRY.keys())}")
    return _OBS_REGISTRY[name](**kwargs)


class ObservationBuilder(ABC):
    """Abstract observation builder."""

    @abstractmethod
    def observation_space(
        self, engine: SimulationEngine, node_id: NodeID,
    ) -> gym.Space:
        """Return the observation space for a single agent (node)."""

    @abstractmethod
    def observe(
        self, engine: SimulationEngine, node_id: NodeID,
    ) -> np.ndarray:
        """Build the observation vector for a single agent (node)."""


@register_obs("default")
class DefaultObservation(ObservationBuilder):
    """Observation: [current_phase_one_hot, incoming_density, incoming_queue].

    For each incoming link to the node: density of last cell + approximate
    queue (binary: above critical density).
    """

    def _get_incoming_links(self, engine: SimulationEngine, node_id: NodeID):
        """Get link IDs of links whose to_node is node_id."""
        return engine.net.node_incoming_links.get(node_id, [])

    def _obs_size(self, engine: SimulationEngine, node_id: NodeID) -> int:
        n_phases = engine.net.n_phases_per_node.get(node_id, 1)
        n_incoming = len(self._get_incoming_links(engine, node_id))
        return n_phases + 2 * n_incoming

    def observation_space(
        self, engine: SimulationEngine, node_id: NodeID,
    ) -> gym.Space:
        size = self._obs_size(engine, node_id)
        return gym.spaces.Box(low=0.0, high=1.0, shape=(size,), dtype=np.float32)

    def observe(
        self, engine: SimulationEngine, node_id: NodeID,
    ) -> np.ndarray:
        net = engine.net
        n_phases = net.n_phases_per_node.get(node_id, 1)
        incoming = self._get_incoming_links(engine, node_id)

        obs = []

        # Phase one-hot
        from ..core.signal import SignalState
        sig_state = engine.signal_manager.states.get(node_id, SignalState())
        phase_oh = np.zeros(n_phases, dtype=np.float32)
        phase_oh[sig_state.current_phase_idx] = 1.0
        obs.append(phase_oh)

        # Incoming link densities (normalized by jam density)
        for lid in incoming:
            last_cell = net.link_last_cell[lid]
            k = engine.state.density[last_cell]
            kj = net.kj[last_cell]
            obs.append(np.array([k / kj if kj > 0 else 0.0], dtype=np.float32))

        # Incoming link queue indicator
        for lid in incoming:
            last_cell = net.link_last_cell[lid]
            k = engine.state.density[last_cell]
            k_crit = net.Q[last_cell] / net.vf[last_cell]
            obs.append(np.array([1.0 if k > k_crit else 0.0], dtype=np.float32))

        return np.concatenate(obs)


@register_obs("pressure")
class PressureObservation(ObservationBuilder):
    """Observation: [current_phase_one_hot, per-movement pressure]."""

    def observation_space(
        self, engine: SimulationEngine, node_id: NodeID,
    ) -> gym.Space:
        n_phases = engine.net.n_phases_per_node.get(node_id, 1)
        n_movs = len(engine.net.node_movements.get(node_id, []))
        size = n_phases + n_movs
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(size,), dtype=np.float32)

    def observe(
        self, engine: SimulationEngine, node_id: NodeID,
    ) -> np.ndarray:
        net = engine.net
        n_phases = net.n_phases_per_node.get(node_id, 1)
        movs = net.node_movements.get(node_id, [])

        obs = []
        from ..core.signal import SignalState
        sig_state = engine.signal_manager.states.get(node_id, SignalState())
        phase_oh = np.zeros(n_phases, dtype=np.float32)
        phase_oh[sig_state.current_phase_idx] = 1.0
        obs.append(phase_oh)

        for mid in movs:
            from_cell = net.mov_from_cell[mid]
            to_cell = net.mov_to_cell[mid]
            pressure = engine.state.density[from_cell] - engine.state.density[to_cell]
            kj_max = max(net.kj[from_cell], net.kj[to_cell])
            norm_pressure = pressure / kj_max if kj_max > 0 else 0.0
            obs.append(np.array([np.clip(norm_pressure, -1, 1)], dtype=np.float32))

        return np.concatenate(obs)


@register_obs("full_density")
class FullDensityObservation(ObservationBuilder):
    """Observation: [current_phase_one_hot, all_cell_densities_normalized]."""

    def observation_space(
        self, engine: SimulationEngine, node_id: NodeID,
    ) -> gym.Space:
        n_phases = engine.net.n_phases_per_node.get(node_id, 1)
        size = n_phases + engine.net.n_cells
        return gym.spaces.Box(low=0.0, high=1.0, shape=(size,), dtype=np.float32)

    def observe(
        self, engine: SimulationEngine, node_id: NodeID,
    ) -> np.ndarray:
        net = engine.net
        n_phases = net.n_phases_per_node.get(node_id, 1)

        obs = []
        from ..core.signal import SignalState
        sig_state = engine.signal_manager.states.get(node_id, SignalState())
        phase_oh = np.zeros(n_phases, dtype=np.float32)
        phase_oh[sig_state.current_phase_idx] = 1.0
        obs.append(phase_oh)

        # All densities normalized
        norm_density = np.where(
            net.kj > 0,
            engine.state.density / net.kj,
            0.0,
        ).astype(np.float32)
        obs.append(norm_density)

        return np.concatenate(obs)
