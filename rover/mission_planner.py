import numpy as np


class MissionPlanner:
    def __init__(self, env):
        self.env = env

        self.phase = "go_to_goal"
        self.replans = 0
        self.goal_reached = False

        self.previous_distance = None
        self.stuck_steps = 0
        self.blocked_steps = 0

        self.recovery_steps = 0
        self.recovery_direction = 1

        self.last_position = None
        self.last_battery = None

        self.avoiding_obstacle = False
        self.avoidance_turns = 0
        self.avoidance_moves = 0
        self.avoidance_direction = 1

    def reset(self):
        self.phase = "go_to_goal"
        self.replans = 0
        self.goal_reached = False

        self.previous_distance = None
        self.stuck_steps = 0
        self.blocked_steps = 0

        self.recovery_steps = 0
        self.recovery_direction = 1

        self.last_position = (
            self.env.rover_position.copy()
        )

        self.last_battery = self.env.battery

        self.avoiding_obstacle = False
        self.avoidance_turns = 0
        self.avoidance_moves = 0
        self.avoidance_direction = 1

    def update(self, obs=None):
        rover = self.env.rover_position

        goal_distance = np.linalg.norm(
            self.env.goal_position - rover
        )

        base_distance = np.linalg.norm(
            self.env.base_position - rover
        )

        if self.phase == "go_to_goal":
            if goal_distance < 3.0:
                self.phase = "return_home"
                self.goal_reached = True
                self.replans += 1

        elif self.phase == "return_home":
            if base_distance < 3.0:
                self.phase = "complete"
                self.replans += 1

        if self.previous_distance is not None:
            if self.phase == "go_to_goal":
                current_distance = goal_distance
            else:
                current_distance = base_distance

            improvement = (
                self.previous_distance -
                current_distance
            )

            if improvement < 0.05:
                self.stuck_steps += 1
            else:
                self.stuck_steps = 0

        if self.phase == "go_to_goal":
            self.previous_distance = goal_distance
        else:
            self.previous_distance = base_distance

        return self.phase

    def is_front_blocked(self, obs):
        if obs is None:
            return False

        front_sensor = float(obs[5])

        return front_sensor < 0.30

    def start_obstacle_avoidance(self, obs):
        if self.avoiding_obstacle:
            return

        if not self.is_front_blocked(obs):
            return

        left_sensor = float(obs[6])
        right_sensor = float(obs[7])

        if left_sensor >= right_sensor:
            self.avoidance_direction = -1
        else:
            self.avoidance_direction = 1

        self.avoiding_obstacle = True

        self.avoidance_turns = 0
        self.avoidance_moves = 0

        self.blocked_steps += 1
        self.replans += 1

    def get_obstacle_action(self, obs):
        if not self.avoiding_obstacle:
            return None

        front_sensor = float(obs[5])
        left_sensor = float(obs[6])
        right_sensor = float(obs[7])

        if (
            front_sensor > 0.60
            and self.avoidance_moves >= 2
        ):
            self.avoiding_obstacle = False
            self.avoidance_turns = 0
            self.avoidance_moves = 0
            return None

        if self.avoidance_turns < 4:
            self.avoidance_turns += 1

            if self.avoidance_direction < 0:
                return 0

            return 1

        if front_sensor < 0.20:
            if left_sensor > right_sensor:
                self.avoidance_direction = -1
            else:
                self.avoidance_direction = 1

            self.avoidance_turns = 0

            if self.avoidance_direction < 0:
                return 0

            return 1

        self.avoidance_moves += 1

        if self.avoidance_moves >= 4:
            self.avoiding_obstacle = False
            self.avoidance_turns = 0
            self.avoidance_moves = 0

        return 2

    def apply(self, obs=None):
        self.update(obs)

        if self.phase == "return_home":
            self.env.returning_home = True

        elif self.phase == "complete":
            self.env.returning_home = True

        self.start_obstacle_avoidance(obs)

        return self.phase

    def get_action_override(self, obs=None):
        action = self.get_obstacle_action(obs)

        if action is not None:
            return action

        return None

    def get_target(self):
        if self.phase in (
            "return_home",
            "complete"
        ):
            return self.env.base_position

        return self.env.goal_position

    def get_status(self):
        rover = self.env.rover_position

        goal_distance = np.linalg.norm(
            self.env.goal_position - rover
        )

        base_distance = np.linalg.norm(
            self.env.base_position - rover
        )

        return {
            "phase": self.phase,
            "goal_distance": goal_distance,
            "base_distance": base_distance,
            "battery": self.env.battery,
            "stuck_steps": self.stuck_steps,
            "blocked_steps": self.blocked_steps,
            "avoiding_obstacle": self.avoiding_obstacle,
            "avoidance_turns": self.avoidance_turns,
            "avoidance_moves": self.avoidance_moves,
            "replans": self.replans
        }