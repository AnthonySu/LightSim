# Changelog

## v0.2.0 (2026-03-18)

### Emergency Vehicle Tracking
- New `EVTracker` class for tracking emergency vehicles along pre-computed routes
- Congestion-dependent EV speed with signal blocking at red lights
- `EVState` dataclass with travel time, distance, stops, and fraction completed
- `distance_to_intersection()` and `distance_to_next_intersection()` methods
- `get_ev_observation()` returns a dict suitable for RL observation builders

### New Signal Controller
- `GreedyEVPreemptionController`: forces green for the EV at every intersection,
  with configurable fallback controller for non-EV intersections

### New Reward Function
- `ev_corridor` reward: combines EV distance progress with queue penalty and arrival bonus

### Performance
- Pre-compute per-node movement index arrays in `SignalManager.__init__`,
  eliminating repeated `np.asarray()` calls in `get_movement_mask()`
- ~19% speedup on 8x8 grids (3,289 → 3,919 steps/s)

### API Additions
- `SignalManager.get_node_phase()`: returns current local phase index at a node
- `SignalManager.is_green_for_movement()`: checks if a specific movement is green

### New Files
- `lightsim/core/ev.py` — EV tracking module
- `examples/ev_corridor.py` — EV corridor optimization example
- `tests/test_ev_tracker.py` — 13 tests for EV tracking
- `CONTRIBUTING.md` — contribution guidelines

### Packaging
- Added `ruff` to dev dependencies
- Added `[tool.ruff]` configuration to `pyproject.toml`

## v0.1.0 (2026-02-25)

Initial public release.

### Core
- Cell Transmission Model (CTM) simulation engine with vectorized NumPy operations
- Triangular fundamental diagram with exact Godunov discretization
- Merge/diverge resolution for multi-lane networks
- Configurable time step, free-flow speed, wave speed, and jam density

### Signal Controllers (7)
- FixedTime, Webster, SOTL, MaxPressure, LostTimeAwareMaxPressure,
  EfficientMaxPressure, GreenWave

### RL Environments
- Gymnasium single-agent interface (`lightsim.make()`)
- PettingZoo multi-agent parallel interface (`lightsim.parallel_env()`)
- Pluggable observations: default, pressure, full_density
- Pluggable actions: phase_select, next_or_stay
- Pluggable rewards: queue, pressure, delay, throughput, waiting_time, normalized_throughput

### Network Generators
- Single intersection, NxM grid, linear arterial
- JSON/dict network loader
- OpenStreetMap import (`from_osm_point(lat, lon, dist)`)

### Built-in Scenarios (19)
- 3 synthetic: single-intersection, grid-4x4, arterial-5
- 16 real-world cities from OpenStreetMap (Manhattan, Shanghai, Beijing,
  Shenzhen, LA, SF, Chicago, Toronto, Sioux Falls, Tokyo, Seoul,
  Singapore, Mumbai, London, Paris, Sydney)

### Mesoscopic Extensions
- Start-up lost time (configurable per-phase)
- Stochastic demand (Poisson arrivals)

### Pretrained Models (6)
- DQN and PPO for single-intersection (queue and pressure rewards)
- DQN and PPO for grid-4x4 multi-agent (shared-parameter)

### Visualization
- Web dashboard with FastAPI + WebSocket + HTML5 Canvas
- Live simulation, replay, and RL checkpoint playback modes

### Decision Transformer
- Offline RL via behavioral cloning on expert demonstrations
- Multi-agent parameter sharing with zero-padded observations

### Utilities
- TravelTimeTracker for link-level travel time measurement
- Network.validate() for topology consistency checks
- JSON scenario I/O for network serialization
