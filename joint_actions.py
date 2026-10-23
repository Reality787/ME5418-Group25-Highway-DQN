"""可复用的九动作目标控制器与阶段 1B-1 调试包装器。

这里只设置目标，真实车辆运动仍由 HighwayEnv 的控制器和动力学产生。
JointDebugEnv 保留原生观测/奖励，不是最终的 HighwayLaneEnv。
"""

from __future__ import annotations

from copy import deepcopy

import gymnasium as gym
import numpy as np
import pygame
from highway_env import utils
from highway_env.envs.common.graphics import EnvViewer
from highway_env.vehicle.controller import MDPVehicle

import config


def _validate_speed_index(value: int | None) -> int | None:
    """None 表示按初始化规则选档；0 是合法且明确提供的档位。"""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("initial_target_speed_index 必须为整数或 None")
    if not 0 <= int(value) < len(config.TARGET_SPEEDS_MPS):
        raise ValueError("initial_target_speed_index 超出五档速度索引范围")
    return int(value)


class JointActionController:
    """绑定一辆 MDPVehicle，提供统一的联合目标更新与结构掩码。

    不调用 env.step，不设置车辆的实际位置、实际速度或航向。
    后续最终项目环境可直接复用本类，不依赖演示脚本。
    """

    def __init__(
        self, vehicle: MDPVehicle, initial_target_speed_index: int | None = None
    ) -> None:
        if not isinstance(vehicle, MDPVehicle):
            raise TypeError("联合控制器需要 HighwayEnv MDPVehicle")
        index = _validate_speed_index(initial_target_speed_index)
        speeds = np.asarray(config.TARGET_SPEEDS_MPS, dtype=float)
        if index is None:
            if config.JOINT_INITIAL_TARGET_SPEED_RULE != "nearest_actual_speed_ties_lower":
                raise ValueError("未知的初始目标速度选档规则")
            # 只在初始化时使用实际车速；容纳机器浮点误差后，等距取较低档。
            distances = np.abs(speeds - vehicle.speed)
            roundoff = 4 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(speeds))))
            nearest = np.flatnonzero(distances <= distances.min() + roundoff)
            index = int(nearest[0])
        self.vehicle = vehicle
        vehicle.target_speeds = speeds.copy()
        vehicle.target_lane_index = vehicle.lane_index
        # 显式赋值，绕过库构造函数 target_speed or speed 对数值 0 的误处理。
        vehicle.speed_index = index
        vehicle.target_speed = float(speeds[index])

    def lane_alignment(self) -> tuple[float, float]:
        """读取目标车道中心误差和相对航向误差，不改变状态。"""
        vehicle = self.vehicle
        lane = vehicle.road.network.get_lane(vehicle.target_lane_index)
        longitudinal, lateral = lane.local_coordinates(vehicle.position)
        heading = utils.wrap_to_pi(vehicle.heading - lane.heading_at(longitudinal))
        return float(lateral), float(heading)

    def is_lane_change_in_progress(self) -> bool:
        """不能只靠最近车道编号判定完成；还必须接近中心并对齐航向。"""
        lateral, heading = self.lane_alignment()
        return bool(
            self.vehicle.lane_index != self.vehicle.target_lane_index
            or abs(lateral) > config.LANE_CHANGE_LATERAL_TOLERANCE_M
            or abs(heading) > config.LANE_CHANGE_HEADING_TOLERANCE_RAD
        )

    def get_action_mask(self) -> np.ndarray:
        """只检查道路边缘、尚未完成的换道、目标速度索引边界。

        返回 shape=(9,) 的 bool 数组；不读取周围车辆、间距、THW 或 TTC。
        Keep 类动作在换道期间仍可用，动作 4 始终满足这三个结构条件。
        """
        vehicle = self.vehicle
        start, end, target_lane = vehicle.target_lane_index
        lane_count = len(vehicle.road.network.graph[start][end])
        changing = self.is_lane_change_in_progress()
        mask = np.empty(config.ACTION_COUNT, dtype=bool)
        for action in range(config.ACTION_COUNT):
            lateral_delta = action // 3 - 1
            speed_delta = action % 3 - 1
            lateral_valid = (
                lateral_delta == 0
                or (
                    not changing
                    and 0 <= target_lane + lateral_delta < lane_count
                )
            )
            speed_valid = (
                0 <= vehicle.speed_index + speed_delta < len(config.TARGET_SPEEDS_MPS)
            )
            mask[action] = lateral_valid and speed_valid
        return mask

    def validate_action(self, action: int) -> int:
        """先严格检查全部分量；失败时不修改任何目标或实际状态。"""
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise TypeError("联合动作必须是整数编号；不接受 bool、浮点数或数组")
        action = int(action)
        if not 0 <= action < config.ACTION_COUNT:
            raise ValueError("联合动作编号必须在 0 到 8 之间")
        if not self.get_action_mask()[action]:
            raise ValueError(f"联合动作 {action} ({config.ACTION_NAMES[action]}) 被结构掩码禁止")
        return action

    def apply(self, action: int) -> int:
        """同时设置两个目标一次；不调用原生 FASTER/SLOWER，也不推进物理过程。"""
        action = self.validate_action(action)
        vehicle = self.vehicle
        start, end, lane_id = vehicle.target_lane_index
        next_lane = (start, end, lane_id + action // 3 - 1)
        next_index = int(vehicle.speed_index + action % 3 - 1)
        next_speed = float(config.TARGET_SPEEDS_MPS[next_index])
        # Keep/hold 的增量为零，保留已有目标，绝不依据实际车速重算。
        vehicle.target_lane_index, vehicle.speed_index, vehicle.target_speed = (
            next_lane, next_index, next_speed
        )
        return action

    def get_debug_state(self, simulation_time: float, physics_steps: int) -> dict:
        """返回独立快照，便于比较更新前、目标写入后与物理推进后的状态。"""
        vehicle = self.vehicle
        lateral, heading = self.lane_alignment()
        return {
            "simulation_time": float(simulation_time),
            "physics_steps": int(physics_steps),
            "current_lane": int(vehicle.lane_index[2]),
            "target_lane": int(vehicle.target_lane_index[2]),
            "actual_speed": float(vehicle.speed),
            "target_speed": float(vehicle.target_speed),
            "target_speed_index": int(vehicle.speed_index),
            "lane_change_in_progress": self.is_lane_change_in_progress(),
            "lateral_error": lateral,
            "heading_error": heading,
            "action_mask": self.get_action_mask(),
        }

    def get_low_level_parameters(self) -> dict:
        """记录实际复用的控制参数；不把库默认值称为 proposal 的执行器限制。"""
        vehicle = self.vehicle
        names = (
            "TAU_ACC", "TAU_HEADING", "TAU_LATERAL", "TAU_PURSUIT",
            "KP_A", "KP_HEADING", "KP_LATERAL", "MAX_STEERING_ANGLE",
            "MIN_SPEED", "MAX_SPEED", "LENGTH", "WIDTH",
        )
        values = {name: float(getattr(vehicle, name)) for name in names}
        values["fixed_acceleration_limit"] = None
        values["limit_note"] = (
            "沿用 HighwayEnv 1.12.1：无常规固定加速度限幅；"
            "实际车速越过 MIN/MAX_SPEED 后约束加速度，碰撞后另行制动。"
            "最终项目执行器限值待确认。"
        )
        return values


class JointDebugViewer(EnvViewer):
    """保留窗口事件，但不把方向键转成原生五动作，防止绕过结构掩码。"""

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
                raise KeyboardInterrupt("joint-debug 窗口已关闭")
            self.sim_surface.handle_event(event)
            # 不调用原生 EventHandler；它会直接改变车道/速度目标。


class JointDebugEnv(gym.Wrapper):
    """真实九动作包装器，内层仍为原生五动作环境。

    内层只接收原生 IDLE；九动作编号从不直接传给原生 DiscreteMetaAction。
    原生观测、原生奖励与 200m 感知暂时保留，不宣称已完成最终环境。
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(config.ACTION_COUNT)
        self.controller: JointActionController | None = None
        self._needs_reset = True
        self._ensure_debug_viewer()

    def _ensure_debug_viewer(self) -> None:
        """仅替换本实例的 human viewer，不改变 stock 或第三方全局行为。"""
        raw = self.unwrapped
        if raw.render_mode == "human" and not isinstance(raw.viewer, JointDebugViewer):
            if raw.viewer is not None:
                raw.viewer.close()
            raw.viewer = JointDebugViewer(raw)

    def _require_controller(self) -> JointActionController:
        if self.controller is None:
            raise gym.error.ResetNeeded("请先调用 joint-debug reset()")
        return self.controller

    def reset(self, *, seed=None, options=None):
        """初始化五档目标与掩码；options 可明确提供 initial_target_speed_index。"""
        native_options = deepcopy(options) if options is not None else {}
        index = _validate_speed_index(native_options.pop("initial_target_speed_index", None))
        native_config = native_options.setdefault("config", {})
        if "action" in native_config:
            raise ValueError("joint-debug 不接受覆盖原生动作适配配置")
        if native_config.get("manual_control", self.unwrapped.config["manual_control"]):
            raise ValueError("joint-debug 不支持 manual_control 覆盖联合目标")
        native_config["action"] = {
            "type": "DiscreteMetaAction",
            "target_speeds": list(config.TARGET_SPEEDS_MPS),
        }
        # reset 重新创建车辆，旧控制器和旧换道状态不再沿用。
        self.controller = None
        self._needs_reset = True
        self._ensure_debug_viewer()
        observation, info = self.env.reset(seed=seed, options=native_options)
        self.controller = JointActionController(self.unwrapped.vehicle, index)
        self.action_space = gym.spaces.Discrete(config.ACTION_COUNT)
        self.observation_space = self.env.observation_space
        self._needs_reset = False
        return observation, self._with_debug_info(info, None, None, None, None)

    def get_action_mask(self) -> np.ndarray:
        """演示、测试和 info 共用同一个结构掩码实现。"""
        return self._require_controller().get_action_mask()

    def get_debug_state(self) -> dict:
        return self._require_controller().get_debug_state(
            self.unwrapped.time, self.unwrapped.steps
        )

    def _with_debug_info(self, native_info, action, native_action, before, applied):
        after = self.get_debug_state()
        info = dict(native_info)
        # reset 时没有真正执行动作，不能把底层随机填入的 action 当作历史动作。
        info["action"] = action
        info["native_action"] = native_action
        info["action_mask"] = after["action_mask"].copy()
        info["joint_debug"] = {
            "action_id": action,
            "action_name": None if action is None else config.ACTION_NAMES[action],
            "native_action_id": native_action,
            "before": before,
            "targets_applied": applied,
            "after": after,
        }
        return info

    def step(self, action):
        """全动作通过校验后才写目标，并且只调用一次内层 step(IDLE)。"""
        controller = self._require_controller()
        if self._needs_reset:
            raise gym.error.ResetNeeded("回合已经结束，请 reset 后再执行动作")
        action = controller.validate_action(action)
        # 在任何目标写入之前获取底层接口；IDLE 保持已设置的速度和车道目标。
        native_idle = self.unwrapped.action_type.actions_indexes["IDLE"]
        before = self.get_debug_state()
        controller.apply(action)
        applied = self.get_debug_state()
        observation, reward, terminated, truncated, info = self.env.step(native_idle)
        self._needs_reset = bool(terminated or truncated)
        info = self._with_debug_info(info, action, int(native_idle), before, applied)
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        try:
            self.env.close()
        finally:
            self.controller = None
            self._needs_reset = True
