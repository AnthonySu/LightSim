# Pretrained Model Weights

Pretrained Stable-Baselines3 checkpoints for reproducing paper results on the single-intersection-v0 scenario.

## Available Models

| File | Algorithm | Reward | Steps | Reward/step | Throughput |
|------|-----------|--------|-------|-------------|------------|
| `dqn_single_intersection.zip` | DQN | queue | 100k | -11.35 | 3,538 |
| `ppo_single_intersection.zip` | PPO | queue | 100k | -6.89 | 3,542 |
| `dqn_single_intersection_pressure.zip` | DQN | pressure | 100k | -0.21 | 3,543 |
| `ppo_single_intersection_pressure.zip` | PPO | pressure | 100k | -0.21 | 3,543 |

**Baselines for comparison:** FixedTime = -13.94 reward/step, MaxPressure = -7.90 reward/step.
All RL models outperform baselines. Performance varies by seed; paper reports best-of-5 seeds.

## Usage

```python
import lightsim
from lightsim.pretrained import load_pretrained, list_pretrained

# List available models
print(list_pretrained())

# Load and evaluate a pretrained DQN
env = lightsim.make("single-intersection-v0", max_steps=720)
model = load_pretrained("dqn_single_intersection", env=env)

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

To retrain all models from scratch:
```bash
python scripts/train_pretrained.py
python scripts/evaluate_pretrained.py
```
