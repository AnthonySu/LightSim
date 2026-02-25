# Changelog

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
