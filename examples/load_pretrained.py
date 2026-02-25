"""Load and evaluate a pretrained RL model.

Demonstrates loading a DQN checkpoint shipped with LightSim
and running it for one episode on the single intersection.

Usage::
    python examples/load_pretrained.py
"""

import lightsim
from lightsim.pretrained import list_pretrained, load_pretrained

# Show available models
print("Available pretrained models:", list_pretrained())

# Load a pretrained DQN
env = lightsim.make("single-intersection-v0", max_steps=720)
model = load_pretrained("dqn_single_intersection", env=env)

# Run one episode
obs, info = env.reset()
total_reward = 0.0
done = False
step = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated
    step += 1

# Print results
metrics = env.unwrapped.engine.get_network_metrics()
print(f"\nEpisode complete ({step} steps)")
print(f"  Total reward: {total_reward:.2f}")
print(f"  Reward/step:  {total_reward / step:.2f}")
print(f"  Throughput:   {metrics.get('total_exited', 0):.0f}")
print(f"  Vehicles:     {metrics.get('total_vehicles', 0):.1f}")
env.close()
