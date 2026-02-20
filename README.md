<div align="center">

# LightSim

### A Lightweight Cell Transmission Model Simulator for Traffic Signal Control Research

**Haoran Su** (NYU) &nbsp;&middot;&nbsp; **Hanxiao Deng** (UC Berkeley)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-97CA00.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/numpy-%E2%89%A51.24-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-%E2%89%A50.29-0081A5.svg)](https://gymnasium.farama.org/)
[![PettingZoo](https://img.shields.io/badge/pettingzoo-multi--agent-8B5CF6.svg)](https://pettingzoo.farama.org/)

[Paper (arXiv)](#) &nbsp;&middot;&nbsp; [Documentation](#quick-start) &nbsp;&middot;&nbsp; [Scenarios](#built-in-scenarios-19) &nbsp;&middot;&nbsp; [Citation](#citation)

</div>

---

LightSim is a pure-Python traffic simulator built on the **Cell Transmission Model** (CTM) for reinforcement learning research in traffic signal control.
It replaces individual vehicle tracking with macroscopic flow dynamics, achieving thousands of simulation steps per second while preserving the controller performance rankings observed in microscopic simulators such as SUMO.

<p align="center">
  <img src="docs/lightsim_shanghai.gif" alt="Shanghai Lujiazui" width="420">
  &nbsp;&nbsp;
  <img src="docs/lightsim_sf.gif" alt="San Francisco FiDi" width="420">
</p>
<p align="center"><sub>Real-world road networks from OpenStreetMap running in real time with MaxPressure signal control.</sub></p>

---

## Why LightSim?

|  | **LightSim** | **SUMO** | **CityFlow** |
|---|:---:|:---:|:---:|
| **Install** | `pip install lightsim` | Platform binaries + env vars | C++ compilation |
| **Model** | Macroscopic (CTM) | Microscopic | Microscopic |
| **Speed** (single intersection) | 21,000 steps/s | 5,000 steps/s | ~8,000 steps/s |
| **RL training speedup** | 3--7x faster than SUMO | baseline | --- |
| **Gymnasium / PettingZoo** | Native | Via wrapper | Via wrapper |
| **Lines to create env** | 3 | ~50 (XML + Python) | ~30 (JSON + Python) |
| **Ranking agreement** | Matches SUMO | --- | --- |
| **Real-world networks** | 16 cities from OSM | Manual XML | Manual JSON |
| **Language** | Pure Python + NumPy | C++ | C++ |

> **Key finding:** Cross-simulator RL experiments (5 algorithms &times; 5 seeds &times; 2 simulators) confirm that LightSim preserves algorithmic rankings while training **3--7x faster** than SUMO.

---

## Installation

```bash
pip install -e .                  # core (numpy + gymnasium)
pip install -e ".[all]"           # all optional dependencies
```

<details>
<summary>Individual extras</summary>

```bash
pip install -e ".[viz]"           # + web visualization (FastAPI + WebSocket)
pip install -e ".[multi]"         # + PettingZoo multi-agent
pip install -e ".[osm]"           # + OpenStreetMap import (osmnx)
```

</details>

---

## Quick Start

**Three lines** to create a Gymnasium environment:

```python
import lightsim
env = lightsim.make("single-intersection-v0")
obs, info = env.reset()
```

**Train an RL agent** with Stable-Baselines3:

```python
from stable_baselines3 import PPO
model = PPO("MlpPolicy", lightsim.make("single-intersection-v0"))
model.learn(total_timesteps=100_000)
```

**Multi-agent** with PettingZoo (16 intersections, independent learners):

```python
env = lightsim.parallel_env("grid-4x4-v0")
observations, infos = env.reset()
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
observations, rewards, terms, truncs, infos = env.step(actions)
```

**Run a classical controller** directly on the engine:

```python
from lightsim import MaxPressureController, SimulationEngine
from lightsim.benchmarks.scenarios import get_scenario

network, demand = get_scenario("grid-4x4-v0")()
engine = SimulationEngine(network=network, dt=1.0,
                          controller=MaxPressureController(min_green=5.0),
                          demand_profiles=demand)
engine.reset()
for _ in range(3600):
    engine.step()
print(engine.get_network_metrics())
```

---

## Built-in Scenarios (19)

### Synthetic Topologies

| Scenario | Signals | Topology |
|---|:---:|---|
| `single-intersection-v0` | 1 | Four-leg intersection with NS/EW phases |
| `grid-4x4-v0` | 16 | 4&times;4 signalized grid with boundary demand |
| `arterial-5-v0` | 5 | Linear corridor with coordinated signals |

### Real-World City Networks (from OpenStreetMap, 500m radius)

| | Scenario | Signals | Location |
|---|---|:---:|---|
| **North America** | `osm-manhattan-v0` | 52 | Midtown Manhattan, New York |
| | `osm-losangeles-v0` | 36 | Downtown, Los Angeles |
| | `osm-sanfrancisco-v0` | 65 | Financial District, San Francisco |
| | `osm-chicago-v0` | 58 | The Loop, Chicago |
| | `osm-toronto-v0` | 59 | Downtown, Toronto |
| | `osm-siouxfalls-v0` | 61 | Downtown, Sioux Falls, SD |
| **East Asia** | `osm-shanghai-v0` | 48 | Lujiazui / Pudong, Shanghai |
| | `osm-beijing-v0` | 59 | Wangfujing, Beijing |
| | `osm-shenzhen-v0` | 40 | Futian CBD, Shenzhen |
| | `osm-tokyo-v0` | 130 | Shibuya, Tokyo |
| | `osm-seoul-v0` | 55 | Gangnam, Seoul |
| **Southeast Asia** | `osm-singapore-v0` | 40 | Orchard Road, Singapore |
| **South Asia** | `osm-mumbai-v0` | 17 | Bandra-Kurla Complex, Mumbai |
| **Europe** | `osm-london-v0` | 131 | City of London |
| | `osm-paris-v0` | 40 | Champs-Elysees, Paris |
| **Oceania** | `osm-sydney-v0` | 43 | CBD, Sydney |

> Networks range from **17 to 131 signalized intersections**. Any additional city can be imported with a single call: `from_osm_point(lat, lon, dist=500)`.

---

## Signal Controllers (8)

| Controller | Strategy | Reference |
|---|---|---|
| `FixedTimeController` | Symmetric fixed-cycle green splits | Webster (1958) |
| `WebsterController` | Optimal cycle length with demand-proportional splits | Webster (1958) |
| `SOTLController` | Self-organizing traffic lights | Gershenson & Heylighen (2005) |
| `MaxPressureController` | Throughput-optimal phase selection by queue pressure | Varaiya (2013) |
| `LostTimeAwareMaxPressureController` | MaxPressure with switching-cost penalty | Ours |
| `EfficientMaxPressureController` | Pressure-proportional green duration extension | Ours |
| `GreenWaveController` | Arterial coordination via travel-time offsets | Classical |
| `RLController` | Wraps a trained RL checkpoint for evaluation | --- |

---

## Visualization

A built-in web dashboard streams simulation state via WebSocket to an HTML5 Canvas frontend:

```bash
python -m lightsim.viz --scenario grid-4x4-v0 --controller MaxPressure
python -m lightsim.viz --scenario osm-manhattan-v0
python -m lightsim.viz --checkpoint model.zip --algo PPO
```

---

## Key Results

Results from the accompanying paper (all experiments on a single laptop, Intel i7, 16 GB RAM):

| Experiment | Finding |
|---|---|
| **Speed** | 800x--21,000x real-time across 1--64 intersections |
| **Fundamental diagram** | Exact match with theoretical CTM (*R*&sup2; = 1.0) |
| **Cross-simulator ranking** | Controller rankings agree with SUMO on both scenarios |
| **RL cross-validation** | 5 RL variants, 5 seeds, 2 simulators: rankings preserved |
| **Training speedup** | 3--7x faster RL training than SUMO |
| **Sim-to-sim transfer** | LightSim-learned timing reduces SUMO delay by 4.9x |
| **Mesoscopic fidelity** | Lost time + stochastic demand close the gap with SUMO |

---

## Architecture

```
lightsim/
├── core/              # CTM simulation engine
│   ├── network.py     # Network topology and compilation
│   ├── engine.py      # Simulation step loop
│   ├── signal.py      # 8 signal controllers
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
├── benchmarks/        # Reproducible experiments & scenarios
├── viz/               # Web visualization server
└── utils/             # Metrics and validation utilities
```

<details>
<summary>Optional dependencies</summary>

| Package | Purpose | Install extra |
|---|---|---|
| `pettingzoo>=1.24` | Multi-agent environments | `[multi]` |
| `fastapi`, `uvicorn`, `websockets` | Web visualization | `[viz]` |
| `osmnx>=1.6` | OpenStreetMap network import | `[osm]` |
| `stable-baselines3` | RL training (DQN, PPO, A2C) | --- |

</details>

---

## Reproducing Paper Results

```bash
pip install lightsim[all]
python -m lightsim.benchmarks.speed_benchmark          # Table 1: speed benchmarks
python -m lightsim.benchmarks.rl_baselines --train-rl  # Table 3: RL baselines
python -m lightsim.benchmarks.sumo_comparison           # Table 2: SUMO comparison
python scripts/rl_cross_validation.py                   # Table/Fig: RL cross-validation
python scripts/cross_validation_mesoscopic.py           # Mesoscopic experiments
python scripts/rl_mesoscopic_experiment.py              # Mesoscopic RL
python scripts/generate_figures.py                      # All paper figures
```

---

## Citation

If you use LightSim in your research, please cite:

```bibtex
@article{su2025lightsim,
  author  = {Su, Haoran and Deng, Hanxiao},
  title   = {LightSim: A Lightweight Cell Transmission Model Simulator
             for Traffic Signal Control Research},
  year    = {2025},
  url     = {https://github.com/AnthonySu/LightSim},
}
```

## License

[MIT](LICENSE)
