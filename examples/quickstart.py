"""Quickstart: 3 lines to a working RL environment."""

import lightsim

# Create environment
env = lightsim.make("single-intersection-v0")

# Run one episode with random actions
obs, info = env.reset(seed=42)
total_reward = 0.0
steps = 0

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1
    if terminated or truncated:
        break

print(f"Episode finished after {steps} steps")
print(f"Total reward: {total_reward:.2f}")
print(f"Vehicles in network: {info['total_vehicles']:.1f}")
print(f"Total throughput: {info['total_exited']:.0f} vehicles")
env.close()
