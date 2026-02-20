"""LightSim: Lightweight CTM-based traffic signal simulation for RL research."""

from __future__ import annotations

from typing import Any

from .core.demand import DemandProfile
from .core.engine import SimulationEngine
from .core.flow_model import CTMFlowModel, FlowModel
from .core.network import Network
from .core.signal import (
    EfficientMaxPressureController,
    FixedTimeController,
    GreenWaveController,
    LostTimeAwareMaxPressureController,
    MaxPressureController,
    RLController,
    SOTLController,
    WebsterController,
)
from .core.types import LinkID, NodeID, NodeType, TurnType
from .envs.single_agent import LightSimEnv

__version__ = "0.1.0"


def make(
    scenario: str = "single-intersection-v0",
    *,
    dt: float = 1.0,
    sim_steps_per_action: int = 5,
    max_steps: int = 3600,
    obs_builder: str = "default",
    action_handler: str = "phase_select",
    reward_fn: str = "queue",
    render_mode: str | None = None,
    stochastic: bool = False,
    **scenario_kwargs: Any,
) -> LightSimEnv:
    """Create a single-agent LightSim Gymnasium environment.

    Usage::

        import lightsim
        env = lightsim.make("single-intersection-v0")
        obs, info = env.reset()
        obs, reward, term, trunc, info = env.step(env.action_space.sample())

    Parameters
    ----------
    scenario : str
        Registered scenario name.
    dt, sim_steps_per_action, max_steps : float/int
        Simulation parameters.
    obs_builder, action_handler, reward_fn : str
        Registered component names.
    render_mode : str, optional
        Gymnasium render mode.
    **scenario_kwargs
        Extra kwargs passed to the scenario factory.
    """
    from .benchmarks.scenarios import get_scenario

    factory = get_scenario(scenario)
    network, demand_profiles = factory(**scenario_kwargs)

    return LightSimEnv(
        network=network,
        dt=dt,
        sim_steps_per_action=sim_steps_per_action,
        max_steps=max_steps,
        obs_builder=obs_builder,
        action_handler=action_handler,
        reward_fn=reward_fn,
        demand_profiles=demand_profiles,
        render_mode=render_mode,
        stochastic=stochastic,
    )


def parallel_env(
    scenario: str = "grid-4x4-v0",
    *,
    dt: float = 1.0,
    sim_steps_per_action: int = 5,
    max_steps: int = 3600,
    obs_builder: str = "default",
    action_handler: str = "phase_select",
    reward_fn: str = "queue",
    render_mode: str | None = None,
    stochastic: bool = False,
    **scenario_kwargs: Any,
) -> "LightSimParallelEnv":  # noqa: F821
    """Create a multi-agent LightSim PettingZoo ParallelEnv.

    Requires ``pettingzoo`` to be installed.
    """
    from .benchmarks.scenarios import get_scenario
    from .envs.multi_agent import LightSimParallelEnv

    factory = get_scenario(scenario)
    network, demand_profiles = factory(**scenario_kwargs)

    return LightSimParallelEnv(
        network=network,
        dt=dt,
        sim_steps_per_action=sim_steps_per_action,
        max_steps=max_steps,
        obs_builder=obs_builder,
        action_handler=action_handler,
        reward_fn=reward_fn,
        demand_profiles=demand_profiles,
        render_mode=render_mode,
        stochastic=stochastic,
    )
