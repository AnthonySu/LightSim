"""Load pretrained model checkpoints shipped with LightSim."""

from pathlib import Path
from typing import Optional

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"


def list_pretrained() -> list[str]:
    """List available pretrained model names."""
    if not WEIGHTS_DIR.exists():
        return []
    return sorted(p.stem for p in WEIGHTS_DIR.glob("*.zip"))


def load_pretrained(name: str, env=None):
    """Load a pretrained Stable-Baselines3 model.

    Args:
        name: Model name (e.g. 'dqn_single_intersection').
              Use list_pretrained() to see available models.
        env: Optional environment to attach to the model.

    Returns:
        Loaded SB3 model ready for evaluation.

    Example:
        >>> import lightsim
        >>> from lightsim.pretrained import load_pretrained
        >>> model = load_pretrained("dqn_single_intersection",
        ...                         env=lightsim.make("single-intersection-v0"))
        >>> obs, _ = model.env.reset()
        >>> action, _ = model.predict(obs, deterministic=True)
    """
    try:
        from stable_baselines3 import DQN, PPO, A2C
    except ImportError:
        raise ImportError(
            "stable-baselines3 is required to load pretrained models. "
            "Install it with: pip install stable-baselines3"
        )

    path = WEIGHTS_DIR / f"{name}.zip"
    if not path.exists():
        available = list_pretrained()
        raise FileNotFoundError(
            f"Pretrained model '{name}' not found at {path}. "
            f"Available models: {available}"
        )

    # Detect algorithm from filename
    name_lower = name.lower()
    if "dqn" in name_lower:
        cls = DQN
    elif "ppo" in name_lower:
        cls = PPO
    elif "a2c" in name_lower:
        cls = A2C
    else:
        # Default to PPO
        cls = PPO

    return cls.load(str(path), env=env)
