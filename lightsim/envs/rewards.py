"""Reward functions for LightSim RL environments.

Registry pattern: use ``@register_reward("name")`` to register.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.engine import SimulationEngine
from ..core.types import NodeID

_REWARD_REGISTRY: dict[str, type[RewardFunction]] = {}


def register_reward(name: str):
    def wrapper(cls):
        _REWARD_REGISTRY[name] = cls
        return cls
    return wrapper


def get_reward_function(name: str, **kwargs) -> RewardFunction:
    if name not in _REWARD_REGISTRY:
        raise KeyError(f"Unknown reward function: {name!r}. "
                       f"Available: {list(_REWARD_REGISTRY.keys())}")
    return _REWARD_REGISTRY[name](**kwargs)


class RewardFunction(ABC):
    @abstractmethod
    def compute(
        self, engine: SimulationEngine, node_id: NodeID,
    ) -> float:
        """Compute the reward for a single agent (node)."""


@register_reward("queue")
class QueueReward(RewardFunction):
    """Reward = negative total queue on incoming links."""

    def compute(
        self, engine: SimulationEngine, node_id: NodeID,
    ) -> float:
        total_queue = 0.0
        for lid in engine.net.node_incoming_links.get(node_id, []):
            total_queue += engine.get_link_queue(lid)
        return -total_queue


@register_reward("pressure")
class PressureReward(RewardFunction):
    """Reward = negative absolute pressure."""

    def compute(
        self, engine: SimulationEngine, node_id: NodeID,
    ) -> float:
        from ..utils.metrics import compute_pressure
        return -abs(compute_pressure(engine, node_id))


@register_reward("delay")
class DelayReward(RewardFunction):
    """Reward = negative total delay on incoming links."""

    def compute(
        self, engine: SimulationEngine, node_id: NodeID,
    ) -> float:
        from ..utils.metrics import compute_link_delay
        total_delay = 0.0
        for lid in engine.net.node_incoming_links.get(node_id, []):
            total_delay += compute_link_delay(engine, lid)
        return -total_delay


@register_reward("throughput")
class ThroughputReward(RewardFunction):
    """Reward = total flow through the intersection's movements."""

    def compute(
        self, engine: SimulationEngine, node_id: NodeID,
    ) -> float:
        net = engine.net
        movs = net.node_movements.get(node_id, [])
        total_flow = 0.0
        for mid in movs:
            from_cell = net.mov_from_cell[mid]
            k = engine.state.density[from_cell]
            q = min(net.vf[from_cell] * k, net.Q[from_cell]) * net.lanes[from_cell]
            total_flow += q
        return total_flow
