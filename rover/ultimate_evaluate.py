from stable_baselines3 import PPO
from bestneuralnetperversion.environment4_4 import RoverEnv
from mission_planner import MissionPlanner
import numpy as np

#change the model name to the one you want to evaluate
MODEL_PATH = "brainzip/rover_brain_4_4"
NUM_RUNS = 200


model = PPO.load(MODEL_PATH)

results = []

for run in range(NUM_RUNS):
    env = RoverEnv()
    planner = MissionPlanner(env)

    obs, info = env.reset()
    planner.reset()

    total_reward = 0.0
    goal_reached = False
    returned_home = False

    distance_traveled = 0.0
    closest_goal_distance = float("inf")
    closest_base_distance = float("inf")

    blocked_forward_actions = 0
    left_actions = 0
    right_actions = 0
    forward_actions = 0

    planner_override_actions = 0
    obstacle_avoidance_actions = 0
    obstacle_avoidance_events = 0

    previous_avoiding_obstacle = False

    direct_goal_distance = np.linalg.norm(
        env.goal_position - env.base_position
    )

    for step in range(env.max_steps):
        old_position = env.rover_position.copy()
        was_returning_home = env.returning_home

        planner.apply(obs)

        was_avoiding_obstacle = planner.avoiding_obstacle

        action_override = planner.get_action_override(obs)

        if action_override is not None:
            action = action_override
            planner_override_actions += 1

            if planner.avoiding_obstacle:
                obstacle_avoidance_actions += 1

        else:
            action, _ = model.predict(
                obs,
                deterministic=True
            )

        if (
            not previous_avoiding_obstacle
            and planner.avoiding_obstacle
        ):
            obstacle_avoidance_events += 1

        previous_avoiding_obstacle = (
            planner.avoiding_obstacle
        )

        if action == 0:
            left_actions += 1
        elif action == 1:
            right_actions += 1
        elif action == 2:
            forward_actions += 1

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += reward

        new_position = env.rover_position.copy()

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

        if (
            goal_reached
            and env.returning_home
            and base_distance < 3.0
        ):
            returned_home = True

        if terminated or truncated:
            break

    battery_used = env.max_battery - env.battery

    if returned_home:
        theoretical_minimum_distance = (
            direct_goal_distance * 2.0
        )

        path_efficiency = (
            theoretical_minimum_distance
            / distance_traveled
            * 100.0
            if distance_traveled > 0
            else 0.0
        )

    else:
        path_efficiency = None

    if returned_home:
        failure_type = "success"

    elif env.battery <= 0.0:
        if goal_reached:
            failure_type = "battery failure after goal"
        else:
            failure_type = "battery failure"

    elif truncated:
        if blocked_forward_actions > 20:
            failure_type = "timeout / obstacle stuck"
        elif distance_traveled < 20:
            failure_type = "timeout / barely moved"
        else:
            failure_type = "timeout"

    elif goal_reached:
        failure_type = "goal reached / return failed"

    else:
        failure_type = "other failure"

    results.append({
        "run": run + 1,
        "success": returned_home,
        "goal_reached": goal_reached,
        "failure_type": failure_type,
        "steps": step + 1,
        "reward": total_reward,
        "battery": env.battery,
        "battery_used": battery_used,
        "direct_distance": direct_goal_distance,
        "distance_traveled": distance_traveled,
        "efficiency": path_efficiency,
        "closest_goal": closest_goal_distance,
        "closest_base": closest_base_distance,
        "blocked": blocked_forward_actions,
        "left": left_actions,
        "right": right_actions,
        "forward": forward_actions,
        "planner_overrides": planner_override_actions,
        "avoidance_actions": obstacle_avoidance_actions,
        "avoidance_events": obstacle_avoidance_events,
        "replans": planner.replans
    })

    env.close()


successes = [
    r for r in results
    if r["success"]
]

failures = [
    r for r in results
    if not r["success"]
]

successful_efficiencies = [
    r["efficiency"]
    for r in successes
    if r["efficiency"] is not None
]

successful_battery = [
    r["battery"]
    for r in successes
]

successful_steps = [
    r["steps"]
    for r in successes
]

successful_rewards = [
    r["reward"]
    for r in successes
]

failure_types = {}

for result in failures:
    failure_type = result["failure_type"]

    if failure_type not in failure_types:
        failure_types[failure_type] = 0

    failure_types[failure_type] += 1


print()
print("rover benchmark")
print()

print("model:", MODEL_PATH)
print("runs:", NUM_RUNS)
print("controller: v6.1 planner + obstacle avoidance")

print()
print("mission performance")

print(
    "successful missions:",
    len(successes),
    "/",
    NUM_RUNS
)

print(
    "success rate:",
    round(
        len(successes) / NUM_RUNS * 100,
        2
    ),
    "%"
)

print(
    "goal reached:",
    sum(
        r["goal_reached"]
        for r in results
    ),
    "/",
    NUM_RUNS
)

print()
print("successful mission performance")

if successes:
    print(
        "average steps:",
        round(np.mean(successful_steps), 2)
    )

    print(
        "median steps:",
        round(np.median(successful_steps), 2)
    )

    print(
        "average reward:",
        round(np.mean(successful_rewards), 2)
    )

    print(
        "average battery remaining:",
        round(np.mean(successful_battery), 2)
    )

    print(
        "average path efficiency:",
        round(
            np.mean(successful_efficiencies),
            2
        ),
        "%"
    )

    print(
        "median path efficiency:",
        round(
            np.median(successful_efficiencies),
            2
        ),
        "%"
    )

    print(
        "best path efficiency:",
        round(
            np.max(successful_efficiencies),
            2
        ),
        "%"
    )

    print(
        "worst path efficiency:",
        round(
            np.min(successful_efficiencies),
            2
        ),
        "%"
    )

else:
    print("no successful missions")

print()
print("navigation")

print(
    "average distance traveled:",
    round(
        np.mean([
            r["distance_traveled"]
            for r in results
        ]),
        2
    )
)

print(
    "average closest goal distance:",
    round(
        np.mean([
            r["closest_goal"]
            for r in results
        ]),
        2
    )
)

print(
    "average blocked forward actions:",
    round(
        np.mean([
            r["blocked"]
            for r in results
        ]),
        2
    )
)

print(
    "total blocked forward actions:",
    sum(
        r["blocked"]
        for r in results
    )
)

print()
print("planner")

print(
    "average planner override actions:",
    round(
        np.mean([
            r["planner_overrides"]
            for r in results
        ]),
        2
    )
)

print(
    "average obstacle avoidance actions:",
    round(
        np.mean([
            r["avoidance_actions"]
            for r in results
        ]),
        2
    )
)

print(
    "average obstacle avoidance events:",
    round(
        np.mean([
            r["avoidance_events"]
            for r in results
        ]),
        2
    )
)

print(
    "total obstacle avoidance events:",
    sum(
        r["avoidance_events"]
        for r in results
    )
)

print()
print("failure breakdown")

if failure_types:
    for failure_type, count in sorted(
        failure_types.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(
            failure_type + ":",
            count
        )
else:
    print("no failures")

print()
print("all runs")

for result in results:
    efficiency = (
        f"{result['efficiency']:.2f}%"
        if result["efficiency"] is not None
        else "n/a"
    )

    print(
        f"{result['run']:3d} | "
        f"{result['failure_type']:<32} | "
        f"steps {result['steps']:4d} | "
        f"reward {result['reward']:8.2f} | "
        f"battery {result['battery']:6.1f} | "
        f"efficiency {efficiency:>8} | "
        f"blocked {result['blocked']:3d} | "
        f"avoid {result['avoidance_actions']:3d}"
    )

print()
print("benchmark complete")