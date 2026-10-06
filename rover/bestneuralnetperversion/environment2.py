import gymnasium as gym
from gymnasium import spaces
import numpy as np


class RoverEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.world_size = 100.0
        self.max_steps = 1000
        self.num_obstacles = 15
        self.sensor_range = 15.0

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
                1.0
            ], dtype=np.float32),
            dtype=np.float32
        )

        self.rover_position = None
        self.goal_position = None
        self.heading = None
        self.obstacles = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.rover_position = self.np_random.uniform(
            10.0,
            90.0,
            size=2
        )

        self.goal_position = self.np_random.uniform(
            10.0,
            90.0,
            size=2
        )

        while np.linalg.norm(
            self.goal_position - self.rover_position
        ) < 30.0:
            self.goal_position = self.np_random.uniform(
                10.0,
                90.0,
                size=2
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
                np.linalg.norm(obstacle - self.rover_position) < 10.0
                or
                np.linalg.norm(obstacle - self.goal_position) < 10.0
            ):
                continue

            self.obstacles.append(obstacle)

        self.steps = 0

        return self._get_observation(), {}

    def step(self, action):
        self.steps += 1

        old_distance = np.linalg.norm(
            self.goal_position - self.rover_position
        )

        if action == 0:
            self.heading -= 0.25

        elif action == 1:
            self.heading += 0.25

        elif action == 2:
            movement = np.array([
                np.cos(self.heading),
                np.sin(self.heading)
            ])

            new_position = self.rover_position + movement

            collision = self._check_collision(new_position)

            if not collision:
                self.rover_position = new_position

        self.rover_position = np.clip(
            self.rover_position,
            0.0,
            self.world_size
        )

        new_distance = np.linalg.norm(
            self.goal_position - self.rover_position
        )

        reward = old_distance - new_distance

        collision = self._check_collision(self.rover_position)

        if collision:
            reward -= 10.0

        terminated = new_distance < 3.0
        truncated = self.steps >= self.max_steps

        if terminated:
            reward += 100.0

        observation = self._get_observation()

        return observation, reward, terminated, truncated, {}

    def _check_collision(self, position):
        for obstacle in self.obstacles:
            if np.linalg.norm(position - obstacle) < 3.0:
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

            position = self.rover_position + direction * distance

            if (
                position[0] < 0.0
                or position[0] > self.world_size
                or
                position[1] < 0.0
                or position[1] > self.world_size
            ):
                return distance / self.sensor_range

            for obstacle in self.obstacles:
                if np.linalg.norm(position - obstacle) < 3.0:
                    return distance / self.sensor_range

        return 1.0

    def _get_observation(self):
        direction = self.goal_position - self.rover_position
        distance = np.linalg.norm(direction)

        if distance > 0:
            direction = direction / distance

        front_sensor = self._get_sensor_distance(0.0)
        left_sensor = self._get_sensor_distance(np.pi / 4)
        right_sensor = self._get_sensor_distance(-np.pi / 4)

        return np.array([
            direction[0],
            direction[1],
            np.cos(self.heading),
            np.sin(self.heading),
            min(distance / self.world_size, 1.0),
            front_sensor,
            left_sensor,
            right_sensor
        ], dtype=np.float32)

    def render(self):
        print(
            f"Rover: {self.rover_position} "
            f"Goal: {self.goal_position} "
            f"Heading: {self.heading:.2f}"
        )