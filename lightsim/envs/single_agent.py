"""Single-agent Gymnasium environment for one intersection."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from ..core.demand import DemandProfile
from ..core.engine import SimulationEngine
from ..core.flow_model import FlowModel
from ..core.network import Network
from ..core.signal import RLController, SignalController
from ..core.types import NodeID, NodeType
from .actions import ActionHandler, get_action_handler
from .observations import ObservationBuilder, get_obs_builder
from .rewards import RewardFunction, get_reward_function


class LightSimEnv(gym.Env):
    """Gymnasium environment wrapping a LightSim simulation.

    Controls a *single* signalised intersection.

    Parameters
    ----------
    network : Network
        The traffic network.
    agent_node : NodeID
        Which signalised node this agent controls.
    dt : float
        Simulation time step (seconds).
    sim_steps_per_action : int
        How many simulation steps per RL action.
    max_steps : int
        Episode length in RL steps.
    obs_builder : str or ObservationBuilder
        Observation type.
    action_handler : str or ActionHandler
        Action type.
    reward_fn : str or RewardFunction
        Reward type.
    demand_profiles : list[DemandProfile], optional
        Demand for origin links.
    flow_model : FlowModel, optional
        Flow model (default CTM).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        network: Network,
        agent_node: NodeID | None = None,
        dt: float = 1.0,
        sim_steps_per_action: int = 5,
        max_steps: int = 3600,
        obs_builder: str | ObservationBuilder = "default",
        action_handler: str | ActionHandler = "phase_select",
        reward_fn: str | RewardFunction = "queue",
        demand_profiles: list[DemandProfile] | None = None,
        flow_model: FlowModel | None = None,
        render_mode: str | None = None,
        stochastic: bool = False,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode

        # Find the agent node (first signalised node if not specified)
        if agent_node is None:
            for node in network.nodes.values():
                if node.node_type == NodeType.SIGNALIZED and node.phases:
                    agent_node = node.node_id
                    break
        if agent_node is None:
            raise ValueError("No signalised node found in network")

        self.agent_node = agent_node
        self.dt = dt
        self.sim_steps_per_action = sim_steps_per_action
        self.max_steps = max_steps
        self.network = network
        self.demand_profiles = demand_profiles or []
        self.flow_model = flow_model
        self.stochastic = stochastic

        # RL controller
        self._rl_controller = RLController()

        # Build engine
        self.engine = SimulationEngine(
            network=network,
            dt=dt,
            flow_model=flow_model,
            controller=self._rl_controller,
            demand_profiles=self.demand_profiles,
            stochastic=stochastic,
        )

        # Observation / action / reward
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
        self.observation_space = self._obs_builder.observation_space(
            self.engine, self.agent_node
        )
        self.action_space = self._action_handler.action_space(
            self.engine, self.agent_node
        )

        self._step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.engine = SimulationEngine(
            network=self.network,
            dt=self.dt,
            flow_model=self.flow_model,
            controller=self._rl_controller,
            demand_profiles=self.demand_profiles,
            stochastic=self.stochastic,
        )
        self.engine.reset(seed=seed)
        self._step_count = 0
        obs = self._obs_builder.observe(self.engine, self.agent_node)
        info = self.engine.get_network_metrics()
        return obs, info

    def step(
        self, action: int | np.integer,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Apply action
        self._action_handler.apply(action, self.engine, self.agent_node)

        # Run simulation
        for _ in range(self.sim_steps_per_action):
            self.engine.step()

        self._step_count += 1

        obs = self._obs_builder.observe(self.engine, self.agent_node)
        reward = self._reward_fn.compute(self.engine, self.agent_node)
        terminated = False
        truncated = self._step_count >= self.max_steps
        info = self.engine.get_network_metrics()

        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        if self.render_mode == "human":
            metrics = self.engine.get_network_metrics()
            print(f"t={metrics['time']:.0f}s  veh={metrics['total_vehicles']:.1f}  "
                  f"in={metrics['total_entered']:.0f}  out={metrics['total_exited']:.0f}")
