"""Type aliases and enumerations for LightSim."""

from __future__ import annotations

from enum import IntEnum, auto
from typing import NewType

import numpy as np

# --- Identifiers (semantic ints) ---
NodeID = NewType("NodeID", int)
LinkID = NewType("LinkID", int)
CellID = NewType("CellID", int)
MovementID = NewType("MovementID", int)
PhaseID = NewType("PhaseID", int)

# --- Enumerations ---

class NodeType(IntEnum):
    ORIGIN = auto()
    DESTINATION = auto()
    SIGNALIZED = auto()
    UNSIGNALIZED = auto()


class TurnType(IntEnum):
    LEFT = auto()
    THROUGH = auto()
    RIGHT = auto()
    UTURN = auto()


# --- NumPy dtypes ---
FLOAT = np.float64
INT = np.int64
