# LightSim

**A Lightweight Cell Transmission Model Simulator for Traffic Signal Control Research**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/numpy-%E2%89%A51.24-orange.svg)](https://numpy.org/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-%E2%89%A50.29-red.svg)](https://gymnasium.farama.org/)

LightSim is a pure-Python traffic simulator built on the Cell Transmission Model (CTM) for reinforcement learning research in traffic signal control. It replaces individual vehicle tracking with macroscopic flow dynamics, achieving thousands of simulation steps per second while preserving the controller performance rankings observed in microscopic simulators such as SUMO.

<p align="center">
  <img src="docs/lightsim_shanghai.gif" alt="Shanghai Lujiazui" width="420">
  &nbsp;
  <img src="docs/lightsim_sf.gif" alt="San Francisco FiDi" width="420">
</p>
<p align="center"><em>Real-world road networks from OpenStreetMap running in real time with MaxPressure signal control.</em></p>

## Key Contributions

- **Fast macroscopic simulation.** A vectorized NumPy CTM engine achieves 800x--21,000x real-time speedup across network sizes from 1 to 64 intersections. RL training is 3--7x faster than SUMO.
- **Standard RL interfaces.** Native [Gymnasium](https://gymnasium.farama.org/) (single-agent) and [PettingZoo](https://pettingzoo.farama.org/) (multi-agent) environments with pluggable observations, actions, and rewards.
- **Cross-simulator fidelity.** Controller rankings agree with SUMO on both rule-based and RL algorithms. Mesoscopic extensions (start-up lost time, stochastic demand) further close the fidelity gap.
- **Built-in benchmarks.** Nine scenarios (three synthetic topologies + six real-world city networks from OpenStreetMap), seven baseline controllers, and reproducible evaluation scripts.
- **OpenStreetMap import.** Import arbitrary road networks with automatic signal detection, movement generation, and demand synthesis via `from_osm_point(lat, lon, dist)`.

## Installation

```bash
pip install -e .                  # core (numpy + gymnasium)
pip install -e ".[viz]"           # + web visualization
pip install -e ".[multi]"         # + PettingZoo multi-agent
pip install -e ".[osm]"           # + OpenStreetMap import
pip install -e ".[all]"           # all optional dependencies
```

## Quick Start

### Single-Agent Environment

```python
import lightsim

env = lightsim.make("single-intersection-v0")
obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

### Multi-Agent Environment

```python
env = lightsim.parallel_env("grid-4x4-v0")
observations, infos = env.reset()

actions = {agent: env.action_space(agent).sample() for agent in env.agents}
observations, rewards, terms, truncs, infos = env.step(actions)
```

### RL Training (Stable-Baselines3)

```python
from stable_baselines3 import PPO
import lightsim

model = PPO("MlpPolicy", lightsim.make("single-intersection-v0"))
model.learn(total_timesteps=100_000)
```

### Classical Controllers

```python
from lightsim import MaxPressureController, SimulationEngine
from lightsim.benchmarks.scenarios import get_scenario

network, demand = get_scenario("grid-4x4-v0")()
engine = SimulationEngine(
    network=network, dt=1.0,
    controller=MaxPressureController(min_green=5.0),
    demand_profiles=demand,
)
engine.reset()
for _ in range(3600):
    engine.step()
print(engine.get_network_metrics())
```

## Built-in Scenarios

| Scenario | Signals | Description |
|---|---|---|
| `single-intersection-v0` | 1 | Four-leg intersection with NS/EW phases |
| `grid-4x4-v0` | 16 | 4x4 signalized grid with boundary demand |
| `arterial-5-v0` | 5 | Linear corridor with coordinated signals |
| `osm-manhattan-v0` | 52 | Midtown Manhattan, New York |
| `osm-shanghai-v0` | 48 | Lujiazui / Pudong, Shanghai |
| `osm-beijing-v0` | 59 | Wangfujing, Beijing |
| `osm-shenzhen-v0` | 40 | Futian CBD, Shenzhen |
| `osm-losangeles-v0` | 36 | Downtown Los Angeles |
| `osm-sanfrancisco-v0` | 65 | Financial District, San Francisco |
| `osm-siouxfalls-v0` | 61 | Downtown Sioux Falls, SD |
| `osm-tokyo-v0` | 130 | Shibuya, Tokyo |
| `osm-chicago-v0` | 58 | The Loop, Chicago |
| `osm-london-v0` | 131 | City of London |
| `osm-paris-v0` | 40 | Champs-Elysees, Paris |
| `osm-singapore-v0` | 40 | Orchard Road, Singapore |
| `osm-seoul-v0` | 55 | Gangnam, Seoul |
| `osm-toronto-v0` | 59 | Downtown, Toronto |
| `osm-mumbai-v0` | 17 | Bandra-Kurla Complex, Mumbai |
| `osm-sydney-v0` | 43 | CBD, Sydney |

## Signal Controllers

| Controller | Strategy |
|---|---|
| `FixedTimeController` | Symmetric fixed-cycle green splits |
| `WebsterController` | Optimal cycle length with demand-proportional splits |
| `SOTLController` | Self-organizing traffic lights with vehicle detection |
| `MaxPressureController` | Throughput-optimal phase selection by queue pressure |
| `LostTimeAwareMaxPressureController` | MaxPressure with switching-cost penalty |
| `EfficientMaxPressureController` | Pressure-proportional green duration extension |
| `GreenWaveController` | Arterial coordination via travel-time offsets |
| `RLController` | Wraps a trained RL checkpoint for evaluation |

## Visualization

A built-in web dashboard streams simulation state via WebSocket to an HTML5 Canvas frontend:

```bash
python -m lightsim.viz --scenario grid-4x4-v0 --controller MaxPressure
python -m lightsim.viz --scenario osm-manhattan-v0
python -m lightsim.viz --checkpoint model.zip --algo PPO
python -m lightsim.viz --replay recording.json
```

## Architecture

```
lightsim/
├── core/              # CTM simulation engine
│   ├── network.py     # Network topology and compilation
│   ├── engine.py      # Simulation step loop
│   ├── signal.py      # Signal controllers
│   ├── flow_model.py  # CTM flow computation
│   ├── demand.py      # Time-varying demand profiles
│   └── types.py       # Type aliases and enums
├── envs/              # RL environments
│   ├── single_agent.py   # Gymnasium interface
│   ├── multi_agent.py    # PettingZoo ParallelEnv
│   ├── observations.py   # Observation builders
│   ├── actions.py        # Action handlers
│   └── rewards.py        # Reward functions
├── networks/          # Network generators
│   ├── grid.py        # NxM grid topology
│   ├── arterial.py    # Linear arterial corridor
│   ├── from_dict.py   # JSON/YAML network loader
│   └── osm.py         # OpenStreetMap import pipeline
├── benchmarks/        # Reproducible experiments
├── viz/               # Web visualization server
└── utils/             # Metrics and validation utilities
```

## Optional Dependencies

| Package | Purpose | Install extra |
|---|---|---|
| `pettingzoo>=1.24` | Multi-agent environments | `[multi]` |
| `fastapi`, `uvicorn`, `websockets` | Web visualization | `[viz]` |
| `osmnx>=1.6` | OpenStreetMap network import | `[osm]` |
| `stable-baselines3` | RL training (DQN, PPO, A2C) | --- |

## Reproducing Paper Results

All experiments reported in the paper can be reproduced:

```bash
pip install lightsim[all]
python -m lightsim.benchmarks.speed_benchmark
python -m lightsim.benchmarks.rl_baselines --train-rl --timesteps 100000
python -m lightsim.benchmarks.sumo_comparison
python scripts/cross_validation_mesoscopic.py
python scripts/rl_mesoscopic_experiment.py
python scripts/rl_cross_validation.py
python scripts/generate_figures.py
```

## Citation

If you use LightSim in your research, please cite:

```bibtex
@article{su2025lightsim,
  author  = {Su, Haoran},
  title   = {LightSim: A Lightweight Cell Transmission Model Simulator for Traffic Signal Control Research},
  year    = {2025},
  url     = {https://github.com/AnthonySu/LightSim},
}
```

## License

[MIT](LICENSE)
