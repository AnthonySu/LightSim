"""Multi-agent PettingZoo ParallelEnv for multiple intersections."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from ..core.demand import DemandProfile
from ..core.engine import SimulationEngine
from ..core.flow_model import FlowModel
from ..core.network import Network
from ..core.signal import RLController
from ..core.types import NodeID, NodeType
from .actions import ActionHandler, get_action_handler
from .observations import ObservationBuilder, get_obs_builder
from .rewards import RewardFunction, get_reward_function

try:
    from pettingzoo import ParallelEnv
    from pettingzoo.utils import wrappers
    HAS_PETTINGZOO = True
except ImportError:
    HAS_PETTINGZOO = False
    # Stub so the class can still be defined
    class ParallelEnv:  # type: ignore[no-redef]
        pass


class LightSimParallelEnv(ParallelEnv):
    """PettingZoo ParallelEnv wrapping a LightSim simulation.

    Each signalised node is an agent.
    """

    metadata = {"render_modes": ["human"], "name": "lightsim_v0"}

    def __init__(
        self,
        network: Network,
        dt: float = 1.0,
        sim_steps_per_action: int = 5,
        max_steps: int = 3600,
        obs_builder: str | ObservationBuilder = "default",
        action_handler: str | ActionHandler = "phase_select",
        reward_fn: str | RewardFunction = "queue",
        demand_profiles: list[DemandProfile] | None = None,
        flow_model: FlowModel | None = None,
        render_mode: str | None = None,
    ) -> None:
        if not HAS_PETTINGZOO:
            raise ImportError("pettingzoo is required for multi-agent env. "
                              "Install with: pip install lightsim[multi]")
        super().__init__()
        self.render_mode = render_mode
        self.network = network
        self.dt = dt
        self.sim_steps_per_action = sim_steps_per_action
        self.max_steps = max_steps
        self.demand_profiles = demand_profiles or []
        self.flow_model = flow_model

        # Find all signalised nodes
        self._agent_nodes: list[NodeID] = []
        for node in network.nodes.values():
            if node.node_type == NodeType.SIGNALIZED and node.phases:
                self._agent_nodes.append(node.node_id)
        self._agent_nodes.sort()

        if not self._agent_nodes:
            raise ValueError("No signalised nodes found in network")

        self.possible_agents = [f"signal_{nid}" for nid in self._agent_nodes]
        self._agent_to_node = {
            f"signal_{nid}": nid for nid in self._agent_nodes
        }

        # RL controller
        self._rl_controller = RLController()

        # Build engine
        self.engine = SimulationEngine(
            network=network,
            dt=dt,
            flow_model=flow_model,
            controller=self._rl_controller,
            demand_profiles=self.demand_profiles,
        )

        # Components
        if isinstance(obs_builder, str):
            self._obs_builder = get_obs_builder(obs_builder)
        else:
            self._obs_builder = obs_builder

        if isinstance(action_handler, str):
            self._action_handler = get_action_handler(action_handler)
        else:
            self._action_handler = action_handler

        if isinstance(reward_fn, str):
            self._reward_fn = get_reward_function(reward_fn)
        else:
            self._reward_fn = reward_fn

        # Spaces
        self.observation_spaces = {
            agent: self._obs_builder.observation_space(
                self.engine, self._agent_to_node[agent]
            )
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: self._action_handler.action_space(
                self.engine, self._agent_to_node[agent]
            )
            for agent in self.possible_agents
        }

        self.agents = list(self.possible_agents)
        self._step_count = 0

    def observation_space(self, agent: str) -> gym.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gym.Space:
        return self.action_spaces[agent]

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        self.engine = SimulationEngine(
            network=self.network,
            dt=self.dt,
            flow_model=self.flow_model,
            controller=self._rl_controller,
            demand_profiles=self.demand_profiles,
        )
        self.engine.reset(seed=seed)
        self.agents = list(self.possible_agents)
        self._step_count = 0

        observations = {}
        infos: dict[str, dict] = {}
        metrics = self.engine.get_network_metrics()
        for agent in self.agents:
            node_id = self._agent_to_node[agent]
            observations[agent] = self._obs_builder.observe(self.engine, node_id)
            infos[agent] = metrics

        return observations, infos

    def step(
        self, actions: dict[str, int | np.integer],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        # Apply actions
        for agent, action in actions.items():
            node_id = self._agent_to_node[agent]
            self._action_handler.apply(action, self.engine, node_id)

        # Run simulation
        for _ in range(self.sim_steps_per_action):
            self.engine.step()

        self._step_count += 1
        truncated_all = self._step_count >= self.max_steps

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos: dict[str, dict] = {}
        metrics = self.engine.get_network_metrics()

        for agent in self.agents:
            node_id = self._agent_to_node[agent]
            observations[agent] = self._obs_builder.observe(self.engine, node_id)
            rewards[agent] = float(self._reward_fn.compute(self.engine, node_id))
            terminations[agent] = False
            truncations[agent] = truncated_all
            infos[agent] = metrics

        if truncated_all:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self) -> None:
        if self.render_mode == "human":
            metrics = self.engine.get_network_metrics()
            print(f"t={metrics['time']:.0f}s  veh={metrics['total_vehicles']:.1f}")
