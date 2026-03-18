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


@register_reward("waiting_time")
class WaitingTimeReward(RewardFunction):
    """Reward = negative vehicle-seconds waiting on incoming links.

    Estimates waiting time as vehicles_queued × dt for each incoming link.
    More closely approximates true delay than instantaneous queue count.
    """

    def compute(
        self, engine: SimulationEngine, node_id: NodeID,
    ) -> float:
        net = engine.net
        total_wait = 0.0
        for lid in net.node_incoming_links.get(node_id, []):
            # Vehicles above critical density are "waiting"
            for cid in net.link_cells.get(lid, []):
                k = engine.state.density[cid]
                k_crit = net.Q[cid] / net.vf[cid] if net.vf[cid] > 0 else 0
                if k > k_crit:
                    queued_veh = (k - k_crit) * net.length[cid] * net.lanes[cid]
                    total_wait += queued_veh * engine.dt
        return -total_wait


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


@register_reward("normalized_throughput")
class NormalizedThroughputReward(RewardFunction):
    """Reward = throughput / max_possible_throughput, in [0, 1].

    Normalizes throughput by the sum of saturation rates across all
    movements at the node, making it network-agnostic.
    """

    def compute(
        self, engine: SimulationEngine, node_id: NodeID,
    ) -> float:
        net = engine.net
        movs = net.node_movements.get(node_id, [])
        if not movs:
            return 0.0
        total_flow = 0.0
        total_capacity = 0.0
        for mid in movs:
            from_cell = net.mov_from_cell[mid]
            k = engine.state.density[from_cell]
            q = min(net.vf[from_cell] * k, net.Q[from_cell]) * net.lanes[from_cell]
            total_flow += q
            total_capacity += net.mov_sat_rate[mid]
        if total_capacity < 1e-9:
            return 0.0
        return min(total_flow / total_capacity, 1.0)


@register_reward("ev_corridor")
class EVCorridorReward(RewardFunction):
    """Reward for EV corridor optimization.

    Combines EV progress reward with a queue penalty. Requires an
    ``EVTracker`` to be attached via ``set_ev_tracker()``.

    reward = alpha * ev_distance_this_step - beta * queue_at_node + lambda * arrived_bonus

    Parameters
    ----------
    alpha : float
        Weight on EV distance progress (default 1.0).
    beta : float
        Weight on queue penalty (default 0.01).
    arrival_bonus : float
        One-time bonus when EV arrives (default 10.0).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.01,
        arrival_bonus: float = 10.0,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.arrival_bonus = arrival_bonus
        self._ev_tracker = None
        self._prev_distance: float = 0.0
        self._gave_bonus: bool = False

    def set_ev_tracker(self, ev_tracker) -> None:
        """Attach the EVTracker for reward computation."""
        self._ev_tracker = ev_tracker
        self._prev_distance = 0.0
        self._gave_bonus = False

    def compute(
        self, engine: SimulationEngine, node_id: NodeID,
    ) -> float:
        # Queue penalty at this node
        queue = 0.0
        for lid in engine.net.node_incoming_links.get(node_id, []):
            queue += engine.get_link_queue(lid)
        reward = -self.beta * queue

        # EV progress (shared across all agents)
        if self._ev_tracker is not None:
            ev = self._ev_tracker
            current_dist = ev.state.distance_traveled
            delta_dist = current_dist - self._prev_distance
            self._prev_distance = current_dist
            reward += self.alpha * delta_dist

            # Arrival bonus (only once)
            if ev.arrived and not self._gave_bonus:
                reward += self.arrival_bonus
                self._gave_bonus = True

        return reward
