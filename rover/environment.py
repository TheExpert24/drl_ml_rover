import gymnasium as gym
from gymnasium import spaces
import numpy as np


class RoverEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.world_size = 100.0
        self.max_steps = 1000
        self.num_obstacles = 8
        self.sensor_range = 15.0
        self.max_battery = 150.0

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=np.array([
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                -1.0,
                0.0
            ], dtype=np.float32),
            high=np.array([
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0
            ], dtype=np.float32),
            dtype=np.float32
        )

        self.rover_position = None
        self.base_position = None
        self.goal_position = None
        self.heading = None
        self.obstacles = None
        self.battery = None
        self.steps = 0
        self.returning_home = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.rover_position = self.np_random.uniform(
            25.0,
            75.0,
            size=2
        )

        self.base_position = self.rover_position.copy()

        direction = self.np_random.uniform(
            0.0,
            2.0 * np.pi
        )

        distance = self.np_random.uniform(
            30.0,
            45.0
        )

        self.goal_position = (
            self.rover_position
            + np.array([
                np.cos(direction),
                np.sin(direction)
            ]) * distance
        )

        while (
            self.goal_position[0] < 10.0
            or self.goal_position[0] > 90.0
            or self.goal_position[1] < 10.0
            or self.goal_position[1] > 90.0
        ):
            direction = self.np_random.uniform(
                0.0,
                2.0 * np.pi
            )

            distance = self.np_random.uniform(
                30.0,
                45.0
            )

            self.goal_position = (
                self.rover_position
                + np.array([
                    np.cos(direction),
                    np.sin(direction)
                ]) * distance
            )

        self.heading = self.np_random.uniform(
            -np.pi,
            np.pi
        )

        self.obstacles = []

        for _ in range(self.num_obstacles):
            obstacle = self.np_random.uniform(
                10.0,
                90.0,
                size=2
            )

            if (
                np.linalg.norm(
                    obstacle - self.rover_position
                ) < 10.0
                or
                np.linalg.norm(
                    obstacle - self.goal_position
                ) < 10.0
            ):
                continue

            self.obstacles.append(obstacle)

        self.battery = self.max_battery
        self.steps = 0
        self.returning_home = False

        return self._get_observation(), {}

    def step(self, action):
        self.steps += 1

        if self.returning_home:
            target = self.base_position
        else:
            target = self.goal_position

        old_distance = np.linalg.norm(
            target - self.rover_position
        )

        old_position = self.rover_position.copy()

        if action == 0:
            self.heading -= 0.25
            self.battery -= 0.1

        elif action == 1:
            self.heading += 0.25
            self.battery -= 0.1

        elif action == 2:
            movement = np.array([
                np.cos(self.heading),
                np.sin(self.heading)
            ])

            new_position = self.rover_position + movement

            if not self._check_collision(new_position):
                self.rover_position = new_position

            self.battery -= 1.0

        self.rover_position = np.clip(
            self.rover_position,
            0.0,
            self.world_size
        )

        self.battery = max(
            self.battery,
            0.0
        )

        new_distance = np.linalg.norm(
            target - self.rover_position
        )

        progress = old_distance - new_distance

        reward = progress * 4.0

        reward -= 0.15

        if progress < 0:
            reward += progress * 2.0

        if action == 0 or action == 1:
            reward -= 0.01

        if np.array_equal(
            old_position,
            self.rover_position
        ) and action == 2:
            reward -= 1.0

        collision = self._check_collision(
            self.rover_position
        )

        if collision:
            reward -= 10.0

        terminated = False
        truncated = False

        if (
            not self.returning_home
            and new_distance < 3.0
        ):
            reward += 100.0 + self.battery
            self.returning_home = True

        elif (
            self.returning_home
            and new_distance < 3.0
        ):
            reward += 300.0
            terminated = True

        if (
            self.battery <= 0.0
            and not terminated
        ):
            reward -= 100.0
            terminated = True

        if self.steps >= self.max_steps:
            truncated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            {}
        )

    def _check_collision(self, position):
        for obstacle in self.obstacles:
            if np.linalg.norm(
                position - obstacle
            ) < 3.0:
                return True

        return False

    def _get_sensor_distance(self, angle):
        direction = np.array([
            np.cos(self.heading + angle),
            np.sin(self.heading + angle)
        ])

        distance = 0.0
        step_size = 0.5

        while distance < self.sensor_range:
            distance += step_size

            position = (
                self.rover_position
                + direction * distance
            )

            if (
                position[0] < 0.0
                or position[0] > self.world_size
                or position[1] < 0.0
                or position[1] > self.world_size
            ):
                return distance / self.sensor_range

            for obstacle in self.obstacles:
                if np.linalg.norm(
                    position - obstacle
                ) < 3.0:
                    return distance / self.sensor_range

        return 1.0

    def _get_observation(self):
        if self.returning_home:
            target = self.base_position
        else:
            target = self.goal_position

        direction = target - self.rover_position

        distance = np.linalg.norm(direction)

        if distance > 0:
            direction = direction / distance

        front_sensor = self._get_sensor_distance(0.0)
        left_sensor = self._get_sensor_distance(
            np.pi / 4
        )
        right_sensor = self._get_sensor_distance(
            -np.pi / 4
        )

        base_direction = (
            self.base_position
            - self.rover_position
        )

        base_distance = np.linalg.norm(
            base_direction
        )

        if base_distance > 0:
            base_direction = (
                base_direction
                / base_distance
            )

        mission_phase = (
            1.0 if self.returning_home else 0.0
        )

        return np.array([
            direction[0],
            direction[1],
            np.cos(self.heading),
            np.sin(self.heading),
            min(
                distance / self.world_size,
                1.0
            ),
            front_sensor,
            left_sensor,
            right_sensor,
            self.battery / self.max_battery,
            base_direction[0],
            base_direction[1],
            mission_phase
        ], dtype=np.float32)

    def render(self):
        target = (
            self.base_position
            if self.returning_home
            else self.goal_position
        )

        print(
            f"Rover: {self.rover_position} "
            f"Target: {target} "
            f"Battery: {self.battery:.1f}% "
            f"Returning: {self.returning_home} "
            f"Heading: {self.heading:.2f}"
        )