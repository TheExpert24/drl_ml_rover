from stable_baselines3 import PPO
from environment import RoverEnv
import numpy as np
import matplotlib.pyplot as plt

env = RoverEnv()
model = PPO.load("rover_brain", env=env)

obs, info = env.reset()

positions = [env.rover_position.copy()]
total_reward = 0.0

for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    positions.append(env.rover_position.copy())

    if terminated or truncated:
        break

positions = np.array(positions)

print("Steps:", step + 1)
print("Total reward:", round(total_reward, 2))
print("Reached goal:", terminated)

fig, ax = plt.subplots(figsize=(8, 8))

for obstacle in env.obstacles:
    circle = plt.Circle(
        obstacle,
        3.0,
        fill=True,
        alpha=0.5
    )
    ax.add_patch(circle)

ax.plot(
    positions[:, 0],
    positions[:, 1],
    linewidth=2
)

ax.scatter(
    positions[0, 0],
    positions[0, 1],
    s=100,
    label="Start"
)

ax.scatter(
    env.goal_position[0],
    env.goal_position[1],
    s=150,
    marker="*",
    label="Goal"
)

ax.set_xlim(0, env.world_size)
ax.set_ylim(0, env.world_size)
ax.set_aspect("equal")
ax.set_title("rover brain evaluation")
ax.legend()
ax.grid()

plt.show()

env.close()