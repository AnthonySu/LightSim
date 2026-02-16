"""Multi-agent example: control a 4x4 grid with PettingZoo."""

import lightsim

env = lightsim.parallel_env("grid-4x4-v0", max_steps=100)

observations, infos = env.reset(seed=42)
print(f"Agents: {env.possible_agents}")
print(f"Observation shape: {next(iter(observations.values())).shape}")

total_rewards = {agent: 0.0 for agent in env.possible_agents}
step = 0

while env.agents:
    actions = {
        agent: env.action_space(agent).sample()
        for agent in env.agents
    }
    observations, rewards, terminations, truncations, infos = env.step(actions)

    for agent, r in rewards.items():
        total_rewards[agent] += r
    step += 1

print(f"\nFinished after {step} steps")
print(f"Sample metrics: {infos[env.possible_agents[0]]}")
print(f"\nTotal rewards per agent:")
for agent in sorted(total_rewards):
    print(f"  {agent}: {total_rewards[agent]:.2f}")
