# Pretrained Model Weights

Pretrained Stable-Baselines3 checkpoints for reproducing paper results.

## Single-Agent Models (single-intersection-v0)

| File | Algorithm | Reward | Steps | Reward/step | Throughput |
|------|-----------|--------|-------|-------------|------------|
| `dqn_single_intersection.zip` | DQN | queue | 100k | -11.35 | 3,538 |
| `ppo_single_intersection.zip` | PPO | queue | 100k | -6.89 | 3,542 |
| `dqn_single_intersection_pressure.zip` | DQN | pressure | 100k | -0.21 | 3,543 |
| `ppo_single_intersection_pressure.zip` | PPO | pressure | 100k | -0.21 | 3,543 |

**Baselines:** FixedTime = -13.94 reward/step, MaxPressure = -7.90 reward/step.
All RL models outperform baselines. Paper reports mean over 5 seeds; these are single-seed (42).

## Multi-Agent Model (grid-4x4-v0)

| File | Algorithm | Scenario | Steps | Notes |
|------|-----------|----------|-------|-------|
| `dqn_grid4x4_multi.zip` | DQN | grid-4x4-v0 | 50k | Shared-parameter, 16 agents |

## Usage

```python
import lightsim
from lightsim.pretrained import load_pretrained, list_pretrained

# List available models
print(list_pretrained())

# Load and evaluate
env = lightsim.make("single-intersection-v0")
model = load_pretrained("ppo_single_intersection", env=env)

obs, info = env.reset()
total_reward = 0.0
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Reward/step: {total_reward / 720:.2f}")
```

## Reproducing

```bash
python scripts/train_pretrained.py          # single-agent models
python scripts/evaluate_pretrained.py       # evaluate all
```
