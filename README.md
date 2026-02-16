# LightSim

Lightweight Cell Transmission Model (CTM) traffic signal simulation for reinforcement learning research.

LightSim fills the gap between heavyweight microscopic simulators (SUMO, CityFlow) and the need for fast, flexible RL environments. Built on vectorised NumPy, it provides native Gymnasium and PettingZoo interfaces with pluggable observations, actions, and rewards.

## Features

- **Fast**: Pure-Python CTM engine with flat NumPy arrays — 10,000+ simulation steps/second on a single core
- **RL-native**: Gymnasium single-agent and PettingZoo multi-agent environments out of the box
- **Pluggable**: Registry-based observations (`default`, `pressure`, `full_density`), actions (`phase_select`, `next_or_stay`), and rewards (`queue`, `pressure`, `delay`, `throughput`)
- **Built-in scenarios**: Single intersection, 4x4 grid, 5-intersection arterial
- **Network generators**: Grid, arterial, JSON/YAML, OpenStreetMap import
- **Baselines**: FixedTime, MaxPressure controllers; DQN/PPO via Stable-Baselines3
- **Visualization**: Real-time web dashboard with FastAPI + WebSocket + HTML5 Canvas

## Installation

```bash
pip install -e .                  # core (numpy + gymnasium)
pip install -e ".[multi]"         # + PettingZoo multi-agent
pip install -e ".[viz]"           # + web visualization
pip install -e ".[osm]"           # + OpenStreetMap import
pip install -e ".[all]"           # everything
```

## Quick Start

```python
import lightsim

env = lightsim.make("single-intersection-v0")
obs, info = env.reset()
obs, reward, term, trunc, info = env.step(env.action_space.sample())
```

### Multi-Agent

```python
env = lightsim.parallel_env("grid-4x4-v0")
observations, infos = env.reset()
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
observations, rewards, terms, truncs, infos = env.step(actions)
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

engine = SimulationEngine(network=net, dt=1.0,
                          demand_profiles=[DemandProfile(LinkID(0), [0.0], [0.3])])
engine.reset()
for _ in range(100):
    engine.step()
print(engine.get_network_metrics())
```

### MaxPressure Controller

```python
from lightsim.core.signal import MaxPressureController

engine = SimulationEngine(
    network=net, dt=1.0,
    controller=MaxPressureController(min_green=5.0),
    demand_profiles=[...],
)
```

### Visualization

```bash
python -m lightsim.viz                              # live simulation
python -m lightsim.viz --scenario grid-4x4-v0       # different scenario
python -m lightsim.viz --replay recording.json      # replay a recording
```

### Benchmarks

```bash
python -m lightsim.benchmarks                       # speed benchmark
python -m lightsim.benchmarks.rl_baselines          # RL baselines
python -m lightsim.benchmarks.rl_baselines --train-rl --timesteps 50000
```

## Available Scenarios

| Scenario | Intersections | Description |
|---|---|---|
| `single-intersection-v0` | 1 | 4-leg intersection with NS/EW phases |
| `grid-4x4-v0` | 16 | 4x4 grid with boundary origins |
| `arterial-5-v0` | 5 | Linear corridor with side streets |

## Architecture

```
lightsim/
├── core/           # CTM simulation engine
│   ├── types.py    # Type aliases and enums
│   ├── network.py  # Network topology + compile()
│   ├── flow_model.py  # FlowModel ABC + CTMFlowModel
│   ├── signal.py   # FixedTime, MaxPressure, RL controllers
│   ├── demand.py   # Time-varying demand profiles
│   └── engine.py   # SimulationEngine step loop
├── envs/           # RL environment wrappers
│   ├── observations.py  # default, pressure, full_density
│   ├── actions.py       # phase_select, next_or_stay
│   ├── rewards.py       # queue, pressure, delay, throughput
│   ├── single_agent.py  # Gymnasium env
│   └── multi_agent.py   # PettingZoo ParallelEnv
├── networks/       # Network generators
│   ├── grid.py     # NxM grid
│   ├── arterial.py # Linear corridor
│   ├── from_dict.py  # JSON/YAML loader
│   └── osm.py     # OpenStreetMap import
├── benchmarks/     # Speed & RL benchmarks
│   ├── speed_benchmark.py
│   ├── rl_baselines.py
│   └── scenarios.py
├── viz/            # Web visualization
│   ├── server.py   # FastAPI + WebSocket
│   ├── recorder.py # State recording
│   └── static/     # HTML5 Canvas frontend
└── utils/
    ├── metrics.py  # Throughput, delay, queue, MFD
    └── validation.py  # Fundamental diagram validation
```

## Core Concepts

**Cell Transmission Model (CTM)**: The network is discretised into cells. Each cell has a density (veh/m/lane). Flow between cells follows the triangular fundamental diagram:

- Sending flow: `S(k) = min(vf * k, Q) * lanes`
- Receiving flow: `R(k) = min(Q, w * (kj - k)) * lanes`

**Pluggable components**: Observations, actions, and rewards use a registry pattern. Register custom components with decorators:

```python
from lightsim.envs.observations import register_obs, ObservationBuilder

@register_obs("my_obs")
class MyObservation(ObservationBuilder):
    def observation_space(self, engine, node_id):
        ...
    def observe(self, engine, node_id):
        ...
```

## Dependencies

| Package | Purpose | Required |
|---|---|---|
| `numpy>=1.24` | Vectorised CTM | Yes |
| `gymnasium>=0.29` | RL interface | Yes |
| `pettingzoo>=1.24` | Multi-agent | Optional |
| `fastapi`, `uvicorn` | Visualization | Optional |
| `osmnx>=1.6` | OSM import | Optional |
| `stable-baselines3` | RL training | Optional |

## License

MIT
