"""LightSim: Lightweight CTM-based traffic signal simulation for RL research.

Quick start::

    import lightsim

    # Single-agent Gymnasium environment
    env = lightsim.make("single-intersection-v0")
    obs, info = env.reset()

    # Multi-agent PettingZoo environment
    pz = lightsim.parallel_env("grid-4x4-v0")

    # Load a pretrained checkpoint
    model = lightsim.load_pretrained("ppo_single_intersection", env=env)

Available scenarios: single-intersection-v0, grid-4x4-v0, arterial-5-v0,
and 16 OSM city networks (osm-manhattan-v0, osm-shanghai-v0, etc.).

Controllers: FixedTime, Webster, SOTL, MaxPressure,
LostTimeAwareMaxPressure, EfficientMaxPressure, GreenWave, RL.

Reward functions: queue, pressure, delay, waiting_time, throughput,
normalized_throughput.
"""

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
from .pretrained import list_pretrained, load_pretrained

__version__ = "0.1.0"

# Decision Transformer (requires torch)
try:
    from .dt import (
        DecisionTransformer,
        DecisionTransformerController,
        DTPolicy,
    )
except ImportError:
    pass


def make(
    scenario: str = "single-intersection-v0",
    *,
    dt: float = 1.0,
    sim_steps_per_action: int = 5,
    max_steps: int = 720,
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
        Scenario name. Synthetic: ``single-intersection-v0``, ``grid-4x4-v0``,
        ``arterial-5-v0``. OSM cities: ``osm-manhattan-v0``, ``osm-shanghai-v0``,
        etc. (16 cities total). Use an invalid name to see all options.
    dt : float
        Simulation time step in seconds (default: 1.0).
    sim_steps_per_action : int
        Simulation steps per RL decision step (default: 5).
    max_steps : int
        Episode length in decision steps (default: 720 = 1 hour).
    obs_builder : str
        Observation type: ``"default"``, ``"pressure"``, ``"full_density"``.
    action_handler : str
        Action type: ``"phase_select"``, ``"next_or_stay"``.
    reward_fn : str
        Reward function: ``"queue"``, ``"pressure"``, ``"delay"``,
        ``"waiting_time"``, ``"throughput"``, ``"normalized_throughput"``.
    render_mode : str, optional
        Gymnasium render mode.
    stochastic : bool
        Enable mesoscopic mode with Poisson demand (default: False).
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
    max_steps: int = 720,
    obs_builder: str = "default",
    action_handler: str = "phase_select",
    reward_fn: str = "queue",
    render_mode: str | None = None,
    stochastic: bool = False,
    **scenario_kwargs: Any,
) -> "LightSimParallelEnv":  # noqa: F821
    """Create a multi-agent LightSim PettingZoo ParallelEnv.

    Each signalized intersection becomes an independent agent. Requires
    ``pettingzoo`` (install with ``pip install lightsim[multi]``).

    Usage::

        import lightsim
        env = lightsim.parallel_env("grid-4x4-v0")
        observations, infos = env.reset()
        actions = {a: env.action_space(a).sample() for a in env.agents}
        observations, rewards, terms, truncs, infos = env.step(actions)

    Parameters are the same as :func:`make`. See ``make()`` for details.
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
