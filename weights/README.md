# Pretrained Model Weights

Pretrained Stable-Baselines3 checkpoints for the LightSim `single-intersection-v0` scenario.
All models were trained for 100,000 timesteps and evaluated over 5 episodes (3,600 steps each).

## Available Models

| File | Algorithm | Scenario | Reward Function | Training Steps | Mean Eval Reward |
|------|-----------|----------|-----------------|----------------|------------------|
| `dqn_single_intersection.zip` | DQN | single-intersection-v0 | queue | 100k | -639,560.54 |
| `ppo_single_intersection.zip` | PPO | single-intersection-v0 | queue | 100k | -25,098.93 |
| `dqn_single_intersection_pressure.zip` | DQN | single-intersection-v0 | pressure | 100k | -777.68 |
| `ppo_single_intersection_pressure.zip` | PPO | single-intersection-v0 | pressure | 100k | -760.28 |

### Notes

- **Queue reward** penalizes total queue length per step, so cumulative rewards are large negative numbers. PPO significantly outperforms DQN on this reward.
- **Pressure reward** penalizes the difference between upstream and downstream occupancy. Both DQN and PPO achieve comparable performance, with PPO slightly ahead.
- All episodes run for the full 3,600 steps (max_steps default).
- Evaluation uses deterministic action selection (`model.predict(obs, deterministic=True)`).

## Usage

```python
from stable_baselines3 import DQN, PPO
import lightsim

# Load a pretrained model
env = lightsim.make("single-intersection-v0")
model = PPO.load("weights/ppo_single_intersection", env=env)

# Evaluate
obs, info = env.reset()
total_reward = 0.0
done = False
truncated = False
while not done and not truncated:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward

print(f"Total reward: {total_reward:.2f}")
```

For pressure reward models:
```python
env = lightsim.make("single-intersection-v0", reward_fn="pressure")
model = PPO.load("weights/ppo_single_intersection_pressure", env=env)
```

## Reproducing

To retrain all models from scratch:
```bash
cd <repo_root>
python train_models.py
```

This will train all 4 models and save evaluation results to `weights/eval_results.json`.
