"""Action handlers for LightSim RL environments.

Registry pattern: use ``@register_action("name")`` to register.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np

from ..core.engine import SimulationEngine
from ..core.signal import RLController
from ..core.types import NodeID

_ACTION_REGISTRY: dict[str, type[ActionHandler]] = {}


def register_action(name: str):
    def wrapper(cls):
        _ACTION_REGISTRY[name] = cls
        return cls
    return wrapper


def get_action_handler(name: str, **kwargs) -> ActionHandler:
    if name not in _ACTION_REGISTRY:
        raise KeyError(f"Unknown action handler: {name!r}. "
                       f"Available: {list(_ACTION_REGISTRY.keys())}")
    return _ACTION_REGISTRY[name](**kwargs)


class ActionHandler(ABC):
    @abstractmethod
    def action_space(
        self, engine: SimulationEngine, node_id: NodeID,
    ) -> gym.Space:
        """Return the action space for a single agent (node)."""

    @abstractmethod
    def apply(
        self,
        action: int | np.integer,
        engine: SimulationEngine,
        node_id: NodeID,
    ) -> None:
        """Apply an RL action to the simulation."""


@register_action("phase_select")
class PhaseSelectAction(ActionHandler):
    """Action = directly select which phase to activate."""

    def action_space(
        self, engine: SimulationEngine, node_id: NodeID,
    ) -> gym.Space:
        n_phases = engine.net.n_phases_per_node.get(node_id, 1)
        return gym.spaces.Discrete(n_phases)

    def apply(
        self,
        action: int | np.integer,
        engine: SimulationEngine,
        node_id: NodeID,
    ) -> None:
        controller = engine.controller
        if isinstance(controller, RLController):
            controller.set_action(node_id, int(action))


@register_action("next_or_stay")
class NextOrStayAction(ActionHandler):
    """Action 0 = keep current phase, action 1 = advance to next phase."""

    def action_space(
        self, engine: SimulationEngine, node_id: NodeID,
    ) -> gym.Space:
        return gym.spaces.Discrete(2)

    def apply(
        self,
        action: int | np.integer,
        engine: SimulationEngine,
        node_id: NodeID,
    ) -> None:
        controller = engine.controller
        if isinstance(controller, RLController):
            if int(action) == 1:
                n_phases = engine.net.n_phases_per_node.get(node_id, 1)
                state = engine.signal_manager.states[node_id]
                next_phase = (state.current_phase_idx + 1) % n_phases
                controller.set_action(node_id, next_phase)
            else:
                state = engine.signal_manager.states[node_id]
                controller.set_action(node_id, state.current_phase_idx)
