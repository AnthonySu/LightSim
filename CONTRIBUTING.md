# Contributing to LightSim

Thank you for considering contributing to LightSim! This document provides guidelines for contributing to the project.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/AnthonySu/lightsim.git
cd lightsim

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linter
ruff check lightsim/ tests/ scripts/ examples/
```

## Project Structure

```
lightsim/
  core/          # CTM engine, network, signal controllers, EV tracking
  envs/          # Gymnasium and PettingZoo RL interfaces
  networks/      # Network generators (grid, arterial, OSM, JSON)
  benchmarks/    # Reproducible experiments and scenarios
  dt/            # Decision Transformer module (requires torch)
  viz/           # Web-based visualization
  utils/         # Metrics, validation, travel time tracking
tests/           # Test suite (270+ tests)
examples/        # Runnable example scripts
scripts/         # Paper reproduction scripts
weights/         # Pretrained model checkpoints
```

## Making Changes

1. **Fork** the repository and create a feature branch from `master`.
2. **Write tests** for any new functionality in `tests/`.
3. **Run the full test suite** before submitting: `pytest tests/ -v`.
4. **Run the linter**: `ruff check lightsim/ tests/`.
5. **Keep commits focused** — one logical change per commit.
6. **Submit a pull request** with a clear description of what and why.

## Code Style

- We follow standard Python conventions (PEP 8) enforced by `ruff`.
- Use type hints for all public function signatures.
- Add docstrings to all public classes and functions (NumPy style).
- Keep dependencies minimal — core LightSim requires only `numpy` and `gymnasium`.

## Adding a Signal Controller

1. Subclass `SignalController` in `lightsim/core/signal.py`.
2. Implement `get_phase_index(node_id, state, net, density) -> int`.
3. Add tests in `tests/test_all_controllers.py`.
4. Export from `lightsim/core/__init__.py`.

## Adding a Network Generator

1. Create a new file in `lightsim/networks/`.
2. Return a `Network` object with nodes, links, movements, and phases.
3. Register as a scenario in `lightsim/benchmarks/scenarios.py`.
4. Add tests in `tests/test_network_generators.py`.

## Adding a Reward Function

1. Add a function in `lightsim/envs/rewards.py` matching the signature `(engine, prev_density) -> float`.
2. Register it in `REWARD_REGISTRY`.
3. Add tests in `tests/test_rewards.py`.

## Reporting Issues

- Use [GitHub Issues](https://github.com/AnthonySu/lightsim/issues).
- Include: Python version, OS, LightSim version, minimal reproduction script.
- For performance issues, include grid size and steps/second measurement.

## Reproducibility

All experiments in the paper can be reproduced using scripts in `scripts/`:

```bash
# Speed benchmarks
python scripts/speed_benchmark.py

# Controller cross-validation (LightSim vs SUMO)
python scripts/cross_validation.py

# RL training (requires stable-baselines3)
python scripts/rl_baselines.py
```

Expected outputs are stored in `results/` as JSON files for comparison.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
