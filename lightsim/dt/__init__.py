"""Decision Transformer for traffic signal control.

Requires PyTorch: ``pip install lightsim[dt]``
"""

from __future__ import annotations

from .dataset import (
    Trajectory,
    TrajectoryDataset,
    collect_trajectories,
    load_trajectories,
    save_trajectories,
)
from .model import DecisionTransformer
from .train import load_dt_model, save_dt_model, train_dt
from .controller import DTPolicy, DecisionTransformerController

__all__ = [
    "Trajectory",
    "TrajectoryDataset",
    "collect_trajectories",
    "save_trajectories",
    "load_trajectories",
    "DecisionTransformer",
    "train_dt",
    "save_dt_model",
    "load_dt_model",
    "DTPolicy",
    "DecisionTransformerController",
]
