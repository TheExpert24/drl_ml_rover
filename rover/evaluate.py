from stable_baselines3 import PPO
from environment import RoverEnv
import numpy as np
import matplotlib.pyplot as plt

env = RoverEnv()
#change the model name to the one you want to evaluate
model = PPO.load("brainzip/rover_brain_5", env=env)

obs, info = env.reset()

start_position = env.rover_position.copy()
positions = [start_position.copy()]

total_reward = 0.0
goal_reached = False
goal_reached_position = None
goal_step = None

left_actions = 0
right_actions = 0
forward_actions = 0
blocked_forward_actions = 0

distance_traveled = 0.0
closest_goal_distance = float("inf")
closest_base_distance = float("inf")

direct_goal_distance = np.linalg.norm(
    env.goal_position - env.base_position
)

direct_home_distance = direct_goal_distance

for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)

    was_returning_home = env.returning_home

    old_position = env.rover_position.copy()

    if action == 0:
        left_actions += 1
    elif action == 1:
        right_actions += 1
    elif action == 2:
        forward_actions += 1

    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    new_position = env.rover_position.copy()
    positions.append(new_position.copy())

    movement_distance = np.linalg.norm(
        new_position - old_position
    )

    distance_traveled += movement_distance

    if action == 2 and movement_distance < 0.001:
        blocked_forward_actions += 1

    goal_distance = np.linalg.norm(
        env.goal_position - new_position
    )

    base_distance = np.linalg.norm(
        env.base_position - new_position
    )

    closest_goal_distance = min(
        closest_goal_distance,
        goal_distance
    )

    closest_base_distance = min(
        closest_base_distance,
        base_distance
    )

    if not was_returning_home and env.returning_home:
        goal_reached = True
        goal_reached_position = new_position.copy()
        goal_step = step + 1

    if terminated or truncated:
        break

positions = np.array(positions)

returned_home = (
    goal_reached
    and np.linalg.norm(
        env.rover_position - env.base_position
    ) < 3.0
)

battery_used = env.max_battery - env.battery

theoretical_minimum_distance = (
    direct_goal_distance * 2.0
)

path_efficiency = (
    theoretical_minimum_distance / distance_traveled
    if distance_traveled > 0
    else 0.0
)

print()
print("rover evaluation")
print()

print("steps:", step + 1)
print("total reward:", round(total_reward, 2))

print()
print("mission")
print("goal reached:", goal_reached)
print("returned to base:", returned_home)

if goal_step is not None:
    print("goal reached at step:", goal_step)

print()
print("battery")
print("battery remaining:", round(env.battery, 2))
print("battery used:", round(battery_used, 2))

print()
print("distance")
print("direct base -> goal:", round(direct_goal_distance, 2))
print("direct goal -> base:", round(direct_home_distance, 2))
print(
    "theoretical minimum mission distance:",
    round(theoretical_minimum_distance, 2)
)
print(
    "actual distance traveled:",
    round(distance_traveled, 2)
)
print(
    "path efficiency:",
    round(path_efficiency * 100, 2),
    "%"
)

print()
print("closest distances")
print(
    "closest to goal:",
    round(closest_goal_distance, 2)
)
print(
    "closest to base:",
    round(closest_base_distance, 2)
)

print()
print("actions")
print("left turns:", left_actions)
print("right turns:", right_actions)
print("forward actions:", forward_actions)
print("blocked forward actions:", blocked_forward_actions)

print()
print("final position")
print(
    "x:",
    round(env.rover_position[0], 2),
    "y:",
    round(env.rover_position[1], 2)
)

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
    linewidth=2,
    label="rover path"
)

ax.scatter(
    start_position[0],
    start_position[1],
    s=120,
    marker="s",
    label="destination"
)

ax.scatter(
    env.goal_position[0],
    env.goal_position[1],
    s=180,
    marker="*",
    label="goal"
)

if goal_reached:
    ax.scatter(
        goal_reached_position[0],
        goal_reached_position[1],
        s=100,
        marker="x",
        label="goal reached"
    )

ax.set_xlim(0, env.world_size)
ax.set_ylim(0, env.world_size)
ax.set_aspect("equal")
ax.set_title("rover brain evaluation")
ax.legend()
ax.grid()

plt.show()

env.close()

