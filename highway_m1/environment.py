"""Group 25 milestone 1: a nine-command highway decision environment.

The simulator, bicycle dynamics, collision detection, IDM/MOBIL traffic and
renderer are supplied by ``highway-env``. This module implements the project's
decision interface, bounded ego controller, observation, structural action mask
and reward. It does not train an agent or claim that a policy is safe.

Units are metres, seconds, radians and m/s unless stated otherwise. Lane 0 is
the leftmost lane, lane 2 the rightmost. Actions are lane-major, with speed
commands Slower/Hold/Faster nested within Left/Keep/Right. The default policy
interval is 0.2 s, containing three 1/15 s simulation updates.
"""

from __future__ import annotations

from functools import partial
from numbers import Real
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from highway_env import utils
from highway_env.envs.common.action import ActionType
from .rendering import enable_rgb_rendering
from highway_env.envs.common.observation import ObservationType
from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle


from .schema import (
    SPEED_TARGETS, LATERAL_COMMANDS, SPEED_COMMANDS, ACTION_NAMES, OBSERVATION_NAMES,
)

class BoundedEgoVehicle(MDPVehicle):
    """Library lane/speed tracking, with project-specific acceleration limits.

    ``MDPVehicle.act('FASTER')`` rounds *measured* speed to choose its new
    target. We deliberately do not use that command: a second speed command
    must increment the pending target even before the first target is reached.
    """

    MIN_SPEED = 0.0
    MAX_SPEED = 35.0
    MIN_ACCELERATION = -5.0
    MAX_ACCELERATION = 3.0

    def speed_control(self, target_speed: float) -> float:
        return float(np.clip(
            super().speed_control(target_speed),
            self.MIN_ACCELERATION, self.MAX_ACCELERATION,
        ))

    def clip_actions(self) -> None:
        # The simulator applies its collision braking here, after speed_control.
        # Clamp again so that crash recovery cannot bypass the declared bounds.
        super().clip_actions()
        self.action["acceleration"] = float(np.clip(
            self.action["acceleration"],
            self.MIN_ACCELERATION, self.MAX_ACCELERATION,
        ))


class JointDecisionAction(ActionType):
    """Execute a valid joint command once per policy decision."""

    def space(self) -> spaces.Discrete:
        return spaces.Discrete(9)

    @property
    def vehicle_class(self):
        return partial(BoundedEgoVehicle, target_speeds=np.array(SPEED_TARGETS))

    def act(self, action: int | np.ndarray) -> None:
        lateral, longitudinal = divmod(int(action), 3)
        vehicle = self.controlled_vehicle
        if lateral != 1:
            origin, destination, target = vehicle.target_lane_index
            vehicle.target_lane_index = (
                origin, destination, int(target + lateral - 1)
            )
        if longitudinal != 1:
            vehicle.speed_index = int(np.clip(
                vehicle.speed_index + longitudinal - 1, 0, len(SPEED_TARGETS) - 1
            ))
            vehicle.target_speed = float(SPEED_TARGETS[vehicle.speed_index])
        # Keep means preserve the target lane, including a lane change in flight.
        # Calling the base tracker avoids the library's measured-speed indexing.
        ControlledVehicle.act(vehicle)

    def get_available_actions(self) -> list[int]:
        return np.flatnonzero(self.env.action_mask()).astype(int).tolist()


class DecisionObservation(ObservationType):
    def space(self) -> spaces.Box:
        # Presence, gaps and lane encodings are [0,1]; signed velocities and
        # offsets are [-1,1]. A common Box bound keeps the interface compact.
        return spaces.Box(-1.0, 1.0, shape=(34,), dtype=np.float32)

    def observe(self) -> np.ndarray:
        return self.env._observe()


class HighwayDecisionEnv(HighwayEnv):
    """Gymnasium environment for the proposal's first code/report milestone.

    ``config`` accepts the simulator's usual rendering settings, plus duration,
    vehicles_count (zero is useful for controlled tests), policy_frequency,
    simulation_frequency, traffic_density_range, traffic_speed_range and
    initial_speed_index. Three lanes and one ego vehicle are fixed by the
    34-feature observation specification. A mask-invalid action in 0..8 maps
    atomically to action 4 (Keep/Hold); out-of-space actions raise ValueError.

    State injection for debugging can use ``env.vehicle`` and
    ``env.road.vehicles``. Traffic sensing reads only current positions, lengths,
    velocities and physical lane membership, never another driver's targets.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "lanes_count": 3,
            "vehicles_count": 20,
            "controlled_vehicles": 1,
            "initial_lane_id": 1,
            "initial_speed_index": 3,
            "duration": 40.0,
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "traffic_density_range": [0.8, 1.2],
            "traffic_speed_range": [18.0, 30.0],
            "sensing_radius": 100.0,
            "settled_lateral_tolerance": 0.2,
            "settled_heading_tolerance": 0.03,
            "headway_seconds": 1.5,
            "offroad_terminal": True,
            "screen_width": 1000,
            "screen_height": 300,
            "scaling": 6.0,
            "centering_position": [0.35, 0.5],
            "real_time_rendering": False,
            "normalize_reward": False,
        })
        return config

    def __init__(self, config: dict | None = None, render_mode: str | None = None):
        self._previous_action = np.zeros(9, dtype=np.float32)
        self._requested_action: int | None = None
        self._executed_action: int | None = None
        self._invalid_action = False
        self._lane_change_initiated = False
        self._speed_target_changed = False
        self._reward_components = self._zero_rewards()
        self._delta_x = 0.0
        self._start_x = 0.0
        self._decision_count = 0
        self._episode_done = False
        super().__init__(config=config, render_mode=render_mode)

    def configure(self, config: dict | None) -> None:
        super().configure(config)
        if self.config["lanes_count"] != 3 or self.config["controlled_vehicles"] != 1:
            raise ValueError("The 34-feature interface requires three lanes and one ego.")
        if self.config.get("manual_control", False):
            raise ValueError("manual_control must be False: actions use the nine-command interface.")
        # Reject invalid numeric settings before they can produce NaN observations,
        # an endless episode or silently truncated counts/indices.
        numeric_keys = (
            "simulation_frequency", "policy_frequency", "duration", "vehicles_count",
            "initial_speed_index", "sensing_radius", "headway_seconds",
            "settled_lateral_tolerance", "settled_heading_tolerance",
        )
        for name in numeric_keys:
            value = self.config[name]
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite real number.")
        for name in ("simulation_frequency", "policy_frequency", "vehicles_count", "initial_speed_index"):
            if not float(self.config[name]).is_integer():
                raise ValueError(f"{name} must be an integer.")
            self.config[name] = int(self.config[name])
        sim = self.config["simulation_frequency"]
        policy = self.config["policy_frequency"]
        if sim <= 0 or policy <= 0 or sim < policy or sim % policy != 0:
            raise ValueError("simulation_frequency must be a positive multiple of policy_frequency.")
        if self.config["duration"] <= 0 or self.config["vehicles_count"] < 0:
            raise ValueError("duration must be positive; vehicles_count cannot be negative.")
        if not 0 <= int(self.config["initial_speed_index"]) < len(SPEED_TARGETS):
            raise ValueError("initial_speed_index must be in 0..4.")
        for name in ("sensing_radius", "headway_seconds"):
            if self.config[name] <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in ("settled_lateral_tolerance", "settled_heading_tolerance"):
            if self.config[name] < 0:
                raise ValueError(f"{name} cannot be negative.")
        for name in ("traffic_density_range", "traffic_speed_range"):
            bounds = self.config[name]
            if (
                not isinstance(bounds, (list, tuple, np.ndarray))
                or np.ndim(bounds) != 1 or len(bounds) != 2
                or any(isinstance(value, bool) or not isinstance(value, Real)
                       or not np.isfinite(value) for value in bounds)
                or bounds[0] <= 0 or bounds[1] < bounds[0]
            ):
                raise ValueError(f"{name} must contain ordered positive bounds.")

    def define_spaces(self) -> None:
        # Explicit types avoid modifying highway-env's global action factories.
        self.action_type = JointDecisionAction(self)
        self.observation_type = DecisionObservation(self)
        self.action_space = self.action_type.space()
        self.observation_space = self.observation_type.space()

    def _reset(self) -> None:
        # All randomness, including built-in vehicle generation and IDM behavior,
        # flows through Gymnasium's seeded generator. Density is drawn per episode.
        self.config["vehicles_density"] = float(self.np_random.uniform(
            *self.config["traffic_density_range"]
        ))
        super()._reset()
        initial_index = int(self.config["initial_speed_index"])
        self.vehicle.speed_index = initial_index
        self.vehicle.speed = float(SPEED_TARGETS[initial_index])
        self.vehicle.target_speed = float(SPEED_TARGETS[initial_index])
        for other in self.road.vehicles:
            if other is not self.vehicle:
                speed = float(self.np_random.uniform(*self.config["traffic_speed_range"]))
                other.speed = speed
                other.target_speed = speed
        self._start_x = float(self.vehicle.position[0])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._previous_action = np.zeros(9, dtype=np.float32)
        self._requested_action = self._executed_action = None
        self._invalid_action = False
        self._lane_change_initiated = self._speed_target_changed = False
        self._reward_components = self._zero_rewards()
        self._delta_x = 0.0
        self._decision_count = 0
        self._episode_done = False
        # AbstractEnv reset creates spaces twice, around road/vehicle creation.
        # Our bookkeeping is ready before it requests an observation/info.
        return super().reset(seed=seed, options=options)

    @property
    def decision_dt(self) -> float:
        return 1.0 / float(self.config["policy_frequency"])

    @staticmethod
    def _zero_rewards() -> dict[str, float]:
        return dict.fromkeys(
            ("progress", "collision", "offroad", "headway", "lane_change", "speed_change"),
            0.0,
        )

    def manoeuvre_in_progress(self) -> bool:
        vehicle = self.vehicle
        if vehicle is None:
            return False
        if vehicle.lane_index != vehicle.target_lane_index:
            return True
        lane = self.road.network.get_lane(vehicle.target_lane_index)
        longitudinal, lateral = lane.local_coordinates(vehicle.position)
        heading_error = utils.wrap_to_pi(vehicle.heading - lane.heading_at(longitudinal))
        return bool(
            abs(lateral) > self.config["settled_lateral_tolerance"]
            or abs(heading_error) > self.config["settled_heading_tolerance"]
        )

    def action_mask(self) -> np.ndarray:
        """Structural validity only; proximity to traffic never hides a command."""
        mask = np.ones(9, dtype=bool)
        vehicle = self.vehicle
        if vehicle is None:
            return mask
        target_lane = int(vehicle.target_lane_index[2])
        changing = self.manoeuvre_in_progress()
        for action in range(9):
            lateral, longitudinal = divmod(action, 3)
            lane_valid = lateral == 1 or (
                not changing and 0 <= target_lane + lateral - 1 < 3
            )
            speed_valid = 0 <= vehicle.speed_index + longitudinal - 1 < len(SPEED_TARGETS)
            mask[action] = lane_valid and speed_valid
        return mask

    def _sense(self) -> list[dict[str, Any]]:
        """Nearest physical front/rear neighbour per absolute lane within 100m.

        Sensing uses a Euclidean centre-to-centre disc. The reported longitudinal
        gap is bumper-to-bumper, clipped to zero for longitudinal overlap. An
        absent neighbour has presence=False, gap=radius and relative_vx=0.
        """
        radius = float(self.config["sensing_radius"])
        neighbours = [
            {
                "lane": lane,
                "front": {"presence": False, "gap": radius, "relative_vx": 0.0},
                "rear": {"presence": False, "gap": radius, "relative_vx": 0.0},
            }
            for lane in range(3)
        ]
        nearest_dx = np.full((3, 2), np.inf)
        ego = self.vehicle
        ego_vx = float(ego.speed * np.cos(ego.heading))
        for other in self.road.vehicles:
            if other is ego:
                continue
            displacement = np.asarray(other.position) - np.asarray(ego.position)
            if float(np.linalg.norm(displacement)) > radius:
                continue
            # Physical nearest-lane membership is measurable. Driver targets and
            # MOBIL/IDM internal intentions are deliberately not inspected.
            lane_index = int(other.lane_index[2])
            if not 0 <= lane_index < 3:
                continue
            dx = float(displacement[0])
            side = 0 if dx >= 0 else 1
            if abs(dx) >= nearest_dx[lane_index, side]:
                continue
            nearest_dx[lane_index, side] = abs(dx)
            key = "front" if side == 0 else "rear"
            gap = max(0.0, abs(dx) - (ego.LENGTH + other.LENGTH) / 2)
            neighbours[lane_index][key] = {
                "presence": True,
                "gap": float(gap),
                "relative_vx": float(other.speed * np.cos(other.heading) - ego_vx),
            }
        return neighbours

    def _observe(self) -> np.ndarray:
        if self.vehicle is None:
            return np.zeros(34, dtype=np.float32)
        values: list[float] = []
        radius = float(self.config["sensing_radius"])
        for lane in self._sense():
            for side in ("front", "rear"):
                neighbour = lane[side]
                values.extend((
                    float(neighbour["presence"]),
                    float(np.clip(neighbour["gap"] / radius, 0.0, 1.0)),
                    float(np.clip(neighbour["relative_vx"] / BoundedEgoVehicle.MAX_SPEED, -1.0, 1.0)),
                ))
        vehicle = self.vehicle
        lane = self.road.network.get_lane(vehicle.lane_index)
        longitudinal, lateral = lane.local_coordinates(vehicle.position)
        half_width = lane.width_at(longitudinal) / 2.0
        values.extend((
            float(np.clip(vehicle.speed * np.cos(vehicle.heading) / 35.0, -1.0, 1.0)),
            float(np.clip(vehicle.speed * np.sin(vehicle.heading) / 35.0, -1.0, 1.0)),
            float(np.clip(lateral / half_width, -1.0, 1.0)),
            float(vehicle.lane_index[2]) / 2.0,
            float(vehicle.target_lane_index[2]) / 2.0,
            float(vehicle.target_speed) / SPEED_TARGETS[-1],
            float(self.manoeuvre_in_progress()),
        ))
        values.extend(self._previous_action.tolist())
        return np.asarray(values, dtype=np.float32)

    def step(self, action: int):
        if self._episode_done or self.done:
            raise gym.error.ResetNeeded("Call reset() before stepping a finished episode.")
        if not self.action_space.contains(action):
            raise ValueError(f"Expected an integer action in 0..8, received {action!r}.")
        self._requested_action = int(action)
        self._invalid_action = not bool(self.action_mask()[int(action)])
        self._executed_action = 4 if self._invalid_action else int(action)
        lateral, longitudinal = divmod(self._executed_action, 3)
        self._lane_change_initiated = lateral != 1
        self._speed_target_changed = longitudinal != 1
        old_x = float(self.vehicle.position[0])
        self._previous_action = np.eye(9, dtype=np.float32)[self._executed_action]
        self._decision_count += 1
        self.time += self.decision_dt
        self._simulate(self._executed_action)
        self._delta_x = float(self.vehicle.position[0] - old_x)
        reward = self._reward(self._executed_action)
        terminated = bool(self._is_terminated())
        truncated = bool(not terminated and self._is_truncated())
        self._episode_done = terminated or truncated
        self.done = self._episode_done
        observation = self._observe()
        info = self._info(observation, self._executed_action)
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), terminated, truncated, info

    def _reward(self, action: int) -> float:
        ego = self.vehicle
        front = self._sense()[int(ego.lane_index[2])]["front"]
        vx = max(0.0, float(ego.speed * np.cos(ego.heading)))
        desired_gap = float(self.config["headway_seconds"]) * vx
        deficit = (
            max(0.0, 1.0 - front["gap"] / desired_gap)
            if front["presence"] and desired_gap > 1e-6 else 0.0
        )
        self._reward_components = {
            "progress": float(np.clip(self._delta_x / (SPEED_TARGETS[-1] * self.decision_dt), 0.0, 1.0)),
            "collision": -20.0 * float(ego.crashed),
            "offroad": -20.0 * float(not ego.on_road),
            "headway": -0.5 * deficit,
            "lane_change": -0.03 * float(self._lane_change_initiated),
            "speed_change": -0.01 * float(self._speed_target_changed),
        }
        return float(sum(self._reward_components.values()))

    def _is_terminated(self) -> bool:
        return bool(self.vehicle.crashed or not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        # Integer decision count avoids 0.2 s floating-point accumulation causing
        # a nominal 40 s episode to take a 201st action.
        return bool(self._decision_count * self.decision_dt >= float(self.config["duration"]) - 1e-9)

    def _rewards(self, action: int | None = None) -> dict[str, float]:
        return self._reward_components.copy()

    def _info(self, obs: np.ndarray, action: int | None = None) -> dict[str, Any]:
        vehicle = self.vehicle
        return {
            "action_mask": self.action_mask(),
            "elapsed_time": float(self._decision_count * self.decision_dt),
            "simulation_time_s": float(self.steps / self.config["simulation_frequency"]),
            "duration": float(self.config["duration"]),
            "ego_speed": float(vehicle.speed),
            "speed": float(vehicle.speed),
            "target_speed": float(vehicle.target_speed),
            "target_speed_index": int(vehicle.speed_index),
            "current_lane": int(vehicle.lane_index[2]),
            "target_lane": int(vehicle.target_lane_index[2]),
            "manoeuvre_in_progress": self.manoeuvre_in_progress(),
            "requested_action": self._requested_action,
            "executed_action": self._executed_action,
            "invalid_action": self._invalid_action,
            "lane_change_initiated": self._lane_change_initiated,
            "speed_target_changed": self._speed_target_changed,
            "progress_m": float(vehicle.position[0] - self._start_x),
            "delta_x": self._delta_x,
            "reward_components": self._reward_components.copy(),
            "crashed": bool(vehicle.crashed),
            "on_road": bool(vehicle.on_road),
            "step_count": self._decision_count,
            "simulation_steps": int(self.steps),
            "nearest_vehicles": self._sense(),
            "control": {
                "acceleration": float(vehicle.action["acceleration"]),
                "steering": float(vehicle.action["steering"]),
            },
            "traffic_density": float(self.config["vehicles_density"]),
        }

    def render(self) -> np.ndarray | None:
        # highway-env 1.12.1 disables EnvViewer.display() whenever SDL uses its
        # dummy driver, including when drawing only to an offscreen Surface.
        # Re-enable surface drawing for RGB exports; this opens no OS window.
        enable_rgb_rendering(self)
        return super().render()
