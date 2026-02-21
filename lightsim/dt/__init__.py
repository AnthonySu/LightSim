"""Decision Transformer for traffic signal control.

Requires PyTorch: ``pip install lightsim[dt]``
"""

from __future__ import annotations

from .dataset import (
    Trajectory,
    TrajectoryDataset,
    collect_multi_agent_trajectories,
    collect_trajectories,
    load_trajectories,
    pad_obs,
    save_trajectories,
)
from .model import DecisionTransformer
from .train import get_device, load_dt_model, save_dt_model, train_dt
from .controller import DTPolicy, DecisionTransformerController, MultiAgentDTPolicy

__all__ = [
    "Trajectory",
    "TrajectoryDataset",
    "collect_trajectories",
    "collect_multi_agent_trajectories",
    "pad_obs",
    "save_trajectories",
    "load_trajectories",
    "DecisionTransformer",
    "get_device",
    "train_dt",
    "save_dt_model",
    "load_dt_model",
    "DTPolicy",
    "DecisionTransformerController",
    "MultiAgentDTPolicy",
]
