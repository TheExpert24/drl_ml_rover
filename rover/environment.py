import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RoverEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.world_size = 100.0
        self.max_steps = 1000

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=np.array([
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                0.0
            ], dtype=np.float32),
            high=np.array([
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
            self.rover_position += np.array([
                np.cos(self.heading),
                np.sin(self.heading)
            ]) * 1.0

        self.rover_position = np.clip(
            self.rover_position,
            0.0,
            self.world_size
        )

        new_distance = np.linalg.norm(
            self.goal_position - self.rover_position
        )

        reward = old_distance - new_distance

        terminated = new_distance < 3.0
        truncated = self.steps >= self.max_steps

        if terminated:
            reward += 100.0

        observation = self._get_observation()

        return observation, reward, terminated, truncated, {}

    def _get_observation(self):
        direction = self.goal_position - self.rover_position
        distance = np.linalg.norm(direction)

        if distance > 0:
            direction = direction / distance

        return np.array([
            direction[0],
            direction[1],
            np.cos(self.heading),
            np.sin(self.heading),
            min(distance / self.world_size, 1.0)
        ], dtype=np.float32)

    def render(self):
        print(
            f"Rover: {self.rover_position} "
            f"Goal: {self.goal_position} "
            f"Heading: {self.heading:.2f}"
        )