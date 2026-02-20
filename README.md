<div align="center">

# LightSim

**Lightweight Cell Transmission Model simulator for traffic signal control research**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/numpy-%E2%89%A51.24-orange.svg)](https://numpy.org/)
[![Gymnasium](https://img.shields.io/badge/gymnasium-%E2%89%A50.29-red.svg)](https://gymnasium.farama.org/)

</div>

<p align="center">
  <img src="docs/lightsim_shanghai.gif" alt="Shanghai Lujiazui — real-time CTM simulation" width="420">
  &nbsp;
  <img src="docs/lightsim_sf.gif" alt="San Francisco Financial District — real-time CTM simulation" width="420">
</p>
<p align="center"><em>Real-world road networks from OpenStreetMap running in real-time with MaxPressure signal control</em></p>

LightSim fills the gap between heavyweight microscopic simulators (SUMO, CityFlow) and the need for fast, flexible RL environments. Built on vectorised NumPy, it provides native Gymnasium and PettingZoo interfaces with pluggable observations, actions, and rewards.

## Features

- **Fast** &mdash; Pure-Python CTM engine with flat NumPy arrays. 10,000+ simulation steps/second on a single core.
- **RL-native** &mdash; Gymnasium single-agent and PettingZoo multi-agent environments out of the box.
- **7 built-in controllers** &mdash; FixedTime, Webster, SOTL, MaxPressure (3 variants), and RLController.
- **Pluggable** &mdash; Registry-based observations, actions, and rewards. Bring your own with decorators.
- **Scenarios** &mdash; Single intersection, 4x4 grid, 5-intersection arterial, and 6 real-world city networks from OpenStreetMap.
- **Visualization** &mdash; Real-time web dashboard with FastAPI + WebSocket + HTML5 Canvas. Supports live simulation, replay, and RL checkpoint playback.
- **OSM Import** &mdash; Import any road network from OpenStreetMap with automatic signal detection and demand generation.

## Requirements

| Requirement | Version |
|---|---|
| Python | >= 3.10 |
| NumPy | >= 1.24 |
| Gymnasium | >= 0.29 |

## Installation

```bash
pip install -e .                  # core only (numpy + gymnasium)
pip install -e ".[viz]"           # + web visualization
pip install -e ".[multi]"         # + PettingZoo multi-agent
pip install -e ".[osm]"           # + OpenStreetMap import
pip install -e ".[all]"           # everything
```

## Quick Start

```python
import lightsim

env = lightsim.make("single-intersection-v0")
obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

### Multi-Agent

```python
env = lightsim.parallel_env("grid-4x4-v0")
observations, infos = env.reset()

actions = {agent: env.action_space(agent).sample() for agent in env.agents}
observations, rewards, terms, truncs, infos = env.step(actions)
```

### Built-in Controllers

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

All controllers: `FixedTimeController`, `WebsterController`, `SOTLController`, `MaxPressureController`, `LostTimeAwareMaxPressureController`, `EfficientMaxPressureController`, `RLController`.

### Visualization

```bash
python -m lightsim.viz                                         # live, FixedTime
python -m lightsim.viz --scenario grid-4x4-v0                  # different scenario
python -m lightsim.viz --controller MaxPressure                # different controller
python -m lightsim.viz --checkpoint model.zip --algo PPO       # replay RL checkpoint
python -m lightsim.viz --replay recording.json                 # replay a recording
python -m lightsim.viz --scenario osm-manhattan-v0             # real-world Manhattan
python -m lightsim.viz --scenario osm-shanghai-v0              # real-world Shanghai
```

### Custom Network

```python
from lightsim.core.network import Network
from lightsim.core.types import LinkID, NodeID, NodeType, TurnType
from lightsim.core.engine import SimulationEngine
from lightsim.core.demand import DemandProfile

net = Network()
net.add_node(NodeID(0), NodeType.SIGNALIZED)
net.add_node(NodeID(1), NodeType.ORIGIN)
net.add_node(NodeID(2), NodeType.DESTINATION)
net.add_link(LinkID(0), NodeID(1), NodeID(0), length=300, lanes=2, n_cells=3)
net.add_link(LinkID(1), NodeID(0), NodeID(2), length=300, lanes=2, n_cells=3)
m = net.add_movement(LinkID(0), LinkID(1), NodeID(0), TurnType.THROUGH)
net.add_phase(NodeID(0), [m.movement_id])

engine = SimulationEngine(
    network=net, dt=1.0,
    demand_profiles=[DemandProfile(LinkID(0), [0.0], [0.3])],
)
engine.reset()
for _ in range(100):
    engine.step()
print(engine.get_network_metrics())
```

### Pluggable Components

Observations, actions, and rewards use a registry pattern. Register custom components with decorators:

```python
from lightsim.envs.observations import register_obs, ObservationBuilder

@register_obs("my_obs")
class MyObservation(ObservationBuilder):
    def observation_space(self, engine, node_id):
        ...
    def observe(self, engine, node_id):
        ...
```

## Scenarios

| Scenario | Intersections | Description |
|---|---|---|
| `single-intersection-v0` | 1 | 4-leg intersection with NS/EW phases |
| `grid-4x4-v0` | 16 | 4x4 grid with boundary demand |
| `arterial-5-v0` | 5 | Linear corridor with side streets |
| `osm-manhattan-v0` | 52 | Midtown Manhattan from OpenStreetMap |
| `osm-shanghai-v0` | 48 | Lujiazui / Pudong, Shanghai |
| `osm-beijing-v0` | 59 | Wangfujing area, Beijing |
| `osm-shenzhen-v0` | 40 | Futian CBD, Shenzhen |
| `osm-losangeles-v0` | 36 | Downtown Los Angeles |
| `osm-sanfrancisco-v0` | 65 | Financial District, San Francisco |

## Project Structure

```
lightsim/
├── core/              # CTM simulation engine
│   ├── network.py     # Network topology
│   ├── engine.py      # Simulation step loop
│   ├── signal.py      # 7 signal controllers
│   ├── flow_model.py  # CTM flow model
│   ├── demand.py      # Time-varying demand profiles
│   └── types.py       # Type aliases and enums
├── envs/              # RL environments
│   ├── single_agent.py   # Gymnasium env
│   ├── multi_agent.py    # PettingZoo ParallelEnv
│   ├── observations.py   # default, pressure, full_density
│   ├── actions.py        # phase_select, next_or_stay
│   └── rewards.py        # queue, pressure, delay, throughput
├── networks/          # Network generators
│   ├── grid.py        # NxM grid
│   ├── arterial.py    # Linear corridor
│   ├── from_dict.py   # JSON/YAML loader
│   └── osm.py         # OpenStreetMap import
├── benchmarks/        # Experiments & baselines
├── viz/               # Web visualization server
└── utils/             # Metrics & validation
```

## Optional Dependencies

| Package | Purpose | Install extra |
|---|---|---|
| `pettingzoo>=1.24` | Multi-agent environments | `[multi]` |
| `fastapi`, `uvicorn`, `websockets` | Web visualization | `[viz]` |
| `osmnx>=1.6` | OpenStreetMap import | `[osm]` |
| `stable-baselines3` | RL training (DQN, PPO, A2C) | &mdash; |

## License

[MIT](LICENSE)
