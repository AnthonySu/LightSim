# Examples

Runnable examples demonstrating LightSim's key features. Each script is self-contained.

| Script | Description |
|--------|-------------|
| `quickstart.py` | 3 lines to create a Gymnasium env and run random actions |
| `max_pressure.py` | Compare FixedTime vs MaxPressure on a single intersection |
| `custom_network.py` | Build a T-intersection from scratch with the Network API |
| `json_network.py` | Define a corridor network via JSON/dict specification |
| `multi_agent.py` | Control a 4x4 grid with PettingZoo multi-agent interface |
| `load_pretrained.py` | Load and evaluate a pretrained DQN checkpoint |
| `record_and_replay.py` | Record simulation frames for web visualization replay |

## Running

```bash
python examples/quickstart.py
python examples/max_pressure.py
python examples/custom_network.py
python examples/json_network.py
python examples/multi_agent.py
python examples/load_pretrained.py      # requires stable-baselines3
python examples/record_and_replay.py    # requires lightsim[viz]
```
