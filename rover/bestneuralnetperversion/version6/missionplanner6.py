import numpy as np


class MissionPlanner:
    FRONT_BLOCKED = 0.30
    FRONT_CLEAR = 0.55
    STUCK_LIMIT = 12
    MAX_AVOIDANCE_ACTIONS = 12
    MAX_RECOVERY_ACTIONS = 8

    def __init__(self, env):
        self.env = env
        self.reset()

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
            if self.env.rover_position is not None
            else None
        )
        self.last_battery = self.env.battery
        self.avoiding_obstacle = False
        self.avoidance_turns = 0
        self.avoidance_moves = 0
        self.avoidance_actions = 0
        self.avoidance_direction = 1
        self.recovering = False
        self.recovery_actions = 0
        self.cooldown_steps = 0

    def update(self, obs=None):
        rover = self.env.rover_position
        goal_distance = np.linalg.norm(
            self.env.goal_position - rover
        )
        base_distance = np.linalg.norm(
            self.env.base_position - rover
        )

        if self.phase == "go_to_goal" and goal_distance < 3.0:
            self.phase = "return_home"
            self.goal_reached = True
            self.replans += 1

        elif self.phase == "return_home" and base_distance < 3.0:
            self.phase = "complete"
            self.replans += 1

        current_distance = (
            goal_distance
            if self.phase == "go_to_goal"
            else base_distance
        )

        if self.previous_distance is not None:
            improvement = self.previous_distance - current_distance

            if improvement < 0.05:
                self.stuck_steps += 1
            else:
                self.stuck_steps = 0

        self.previous_distance = current_distance
        self.last_position = rover.copy()
        self.last_battery = self.env.battery

        if (
            self.stuck_steps >= self.STUCK_LIMIT
            and not self.recovering
        ):
            self.recovering = True
            self.recovery_actions = 0
            self.recovery_steps = 0
            self.avoiding_obstacle = False
            self.avoidance_actions = 0
            self.recovery_direction = self._clearest_turn(obs)
            self.replans += 1

        return self.phase

    def _clearest_turn(self, obs):
        if obs is None:
            return 1

        left_sensor = float(obs[6])
        right_sensor = float(obs[7])

        return 1 if left_sensor >= right_sensor else -1

    def is_front_blocked(self, obs):
        return (
            obs is not None
            and float(obs[5]) < self.FRONT_BLOCKED
        )

    def start_obstacle_avoidance(self, obs):
        if (
            self.recovering
            or self.avoiding_obstacle
            or self.cooldown_steps > 0
        ):
            return

        if not self.is_front_blocked(obs):
            return

        self.avoidance_direction = self._clearest_turn(obs)
        self.avoiding_obstacle = True
        self.avoidance_turns = 0
        self.avoidance_moves = 0
        self.avoidance_actions = 0
        self.blocked_steps += 1
        self.replans += 1

    def _finish_avoidance(self):
        self.avoiding_obstacle = False
        self.avoidance_turns = 0
        self.avoidance_moves = 0
        self.avoidance_actions = 0
        self.cooldown_steps = 3

    def get_obstacle_action(self, obs):
        if not self.avoiding_obstacle or obs is None:
            return None

        front_sensor = float(obs[5])
        left_sensor = float(obs[6])
        right_sensor = float(obs[7])

        if (
            front_sensor >= self.FRONT_CLEAR
            and self.avoidance_moves >= 1
        ):
            self._finish_avoidance()
            return None

        if self.avoidance_actions >= self.MAX_AVOIDANCE_ACTIONS:
            self._finish_avoidance()
            return None

        self.avoidance_actions += 1

        if front_sensor < self.FRONT_BLOCKED:
            self.avoidance_direction = (
                1 if left_sensor >= right_sensor else -1
            )
            self.avoidance_turns = 0
            return (
                1 if self.avoidance_direction > 0 else 0
            )

        if self.avoidance_turns < 2:
            self.avoidance_turns += 1
            return (
                1 if self.avoidance_direction > 0 else 0
            )

        self.avoidance_moves += 1
        return 2

    def _get_recovery_action(self, obs):
        if obs is None:
            return None

        if self.recovery_actions >= self.MAX_RECOVERY_ACTIONS:
            self.recovering = False
            self.stuck_steps = 0
            self.previous_distance = None
            self.cooldown_steps = 3
            return None

        self.recovery_actions += 1
        self.recovery_steps = self.recovery_actions

        front_sensor = float(obs[5])
        left_sensor = float(obs[6])
        right_sensor = float(obs[7])

        if front_sensor < self.FRONT_BLOCKED:
            self.recovery_direction = (
                1 if left_sensor >= right_sensor else -1
            )
            return (
                1 if self.recovery_direction > 0 else 0
            )

        if self.recovery_actions <= 3:
            return (
                1 if self.recovery_direction > 0 else 0
            )

        return 2

    def apply(self, obs=None):
        self.update(obs)

        if self.phase in ("return_home", "complete"):
            self.env.returning_home = True

        if self.cooldown_steps > 0:
            self.cooldown_steps -= 1

        self.start_obstacle_avoidance(obs)
        return self.phase

    def get_action_override(self, obs=None):
        if self.recovering:
            return self._get_recovery_action(obs)

        return self.get_obstacle_action(obs)

    def get_target(self):
        if self.phase in ("return_home", "complete"):
            return self.env.base_position

        return self.env.goal_position

    def get_status(self):
        rover = self.env.rover_position

        return {
            "phase": self.phase,
            "goal_distance": float(
                np.linalg.norm(self.env.goal_position - rover)
            ),
            "base_distance": float(
                np.linalg.norm(self.env.base_position - rover)
            ),
            "battery": self.env.battery,
            "stuck_steps": self.stuck_steps,
            "blocked_steps": self.blocked_steps,
            "avoiding_obstacle": self.avoiding_obstacle,
            "avoidance_turns": self.avoidance_turns,
            "avoidance_moves": self.avoidance_moves,
            "avoidance_actions": self.avoidance_actions,
            "recovering": self.recovering,
            "recovery_actions": self.recovery_actions,
            "cooldown_steps": self.cooldown_steps,
            "replans": self.replans,
        }