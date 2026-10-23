"""正式 34 维 / 九动作 Gymnasium 环境，组合现有仿真、控制和观测模块。"""

from __future__ import annotations

from copy import deepcopy

import gymnasium as gym
import numpy as np

import config
from environment import make_observation_debug_env
from reward import compute_reward


class HighwayLaneEnv(gym.Env):
    """项目奖励替换原生奖励；无需 DQN、模型文件或 PyTorch。

    invalid_action_mode='hold' 仅兼容结构禁止的有效编号，整项执行动作4并明确记录；
    'raise' 在任何状态修改前拒绝。类型错误和范围错误在两种模式都拒绝。
    unwrapped 保持本正式环境，Gym checker 不会绕过到原生五动作环境。
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": config.POLICY_FREQUENCY_HZ}

    def __init__(self, render_mode: str | None = None, invalid_action_mode: str = "hold") -> None:
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError("render_mode 必须为 None、human 或 rgb_array")
        if invalid_action_mode not in ("hold", "raise"):
            raise ValueError("invalid_action_mode 必须为 hold 或 raise")
        self.render_mode = render_mode
        self.invalid_action_mode = invalid_action_mode
        self.simulator = make_observation_debug_env(render_mode=render_mode)
        if render_mode == "rgb_array":
            # 真正离屏渲染，避免图像采样收到原生键盘驾驶事件。
            self.simulator.unwrapped.configure({"offscreen_rendering": True})
        self.action_space = self.simulator.action_space
        self.observation_space = self.simulator.observation_space
        self._needs_reset = True
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.invalid_action_count = 0

    @property
    def controller(self):
        return self.simulator.controller

    def get_action_mask(self) -> np.ndarray:
        return self.simulator.get_action_mask().copy()

    def get_debug_state(self) -> dict:
        return self.simulator.get_debug_state()

    def get_observation(self):
        """复用无副作用观测读取，主要供诊断和可控场景测试使用。"""
        return self.simulator.get_observation()

    def _validate_reset_options(self, options):
        options = deepcopy(options) if options is not None else {}
        overrides = options.get("config", {})
        fixed = {
            "lanes_count": config.LANE_COUNT,
            "controlled_vehicles": config.CONTROLLED_VEHICLE_COUNT,
            "duration": config.EPISODE_DURATION_SECONDS,
            "policy_frequency": config.POLICY_FREQUENCY_HZ,
            "simulation_frequency": config.SIMULATION_FREQUENCY_HZ,
            "offroad_terminal": True,
            "manual_control": False,
        }
        for key, value in fixed.items():
            if key in overrides and overrides[key] != value:
                raise ValueError(f"正式环境固定 {key}={value}，不支持此覆盖")
        # 动作适配、观测提取和被控车辆类型不能由 reset 悄悄替换。
        if "action" in overrides or "observation" in overrides:
            raise ValueError("正式环境不支持覆盖 action/observation 接口")
        if self.render_mode == "rgb_array" and overrides.get("offscreen_rendering", True) is not True:
            raise ValueError("正式 rgb_array 模式必须离屏渲染")
        return options

    def reset(self, *, seed=None, options=None):
        options = self._validate_reset_options(options)
        super().reset(seed=seed)
        self._needs_reset = True
        observation, native_info = self.simulator.reset(seed=seed, options=options)
        raw = self.simulator.unwrapped
        # Gym 所见的 RNG 就是此次真实场景初始化使用的实例 RNG，不重置全局随机数。
        self._np_random = raw.np_random
        self._np_random_seed = raw.np_random_seed
        self.action_space = self.simulator.action_space
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.invalid_action_count = 0
        self._needs_reset = False
        info = self._project_info(native_info, None, None, False)
        info["reward_terms"] = dict.fromkeys(
            ("progress", "headway", "failure", "lane_change", "target_speed"), 0.0
        )
        info["reward_diagnostics"] = {}
        info["native_reward_diagnostic"] = None
        return observation.copy(), info

    def _project_info(self, native_info, requested, executed, invalid):
        # 深复制历史诊断，后续 step 不会回写此前返回的数据。
        info = deepcopy(native_info)
        native_terms = info.pop("rewards", None)
        if native_terms is not None:
            info["native_reward_terms_diagnostic"] = native_terms
        raw = self.simulator.unwrapped
        collision = bool(raw.vehicle.crashed)
        off_road = not bool(raw.vehicle.on_road)
        info.update({
            "requested_action": requested, "executed_action": executed,
            "invalid_action": bool(invalid),
            "action_mask": self.get_action_mask(),
            "simulation_time": float(raw.time), "physics_steps": int(raw.steps),
            "collision": collision, "off_road": off_road,
            "failure_reasons": [name for name, active in
                                (("collision", collision), ("off_road", off_road)) if active],
            "episode_reward": float(self.episode_reward),
            "episode_steps": self.episode_steps,
            "invalid_action_count": self.invalid_action_count,
        })
        return info

    def step(self, action):
        if self._needs_reset:
            raise gym.error.ResetNeeded("请先 reset；结束或关闭后不能继续 step")
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise TypeError("项目动作必须是整数编号，不接受 bool、浮点数或数组")
        requested = int(action)
        if not 0 <= requested < config.ACTION_COUNT:
            raise ValueError("项目动作编号必须在0到8之间")
        invalid = not bool(self.get_action_mask()[requested])
        if invalid and self.invalid_action_mode == "raise":
            # 共享控制器提供一致错误；拒绝前不改计数、历史、目标或物理状态。
            self.controller.validate_action(requested)
        executed = 4 if invalid else requested
        raw = self.simulator.unwrapped
        before_position = raw.vehicle.position.copy()
        before_steps = raw.steps
        road_direction = self.simulator.builder._lanes()[0].direction.copy()
        observation, native_reward, _, _, native_info = self.simulator.step(executed)
        # 从实际物理更新数获得经过的时间；不再 step 来测量位移或时距。
        dt = (raw.steps - before_steps) / raw.config["simulation_frequency"]
        displacement = float((raw.vehicle.position - before_position) @ road_direction)
        physical = native_info["observation_debug"]["raw_values"]
        current_lane, target_lane = int(physical[21]), int(physical[22])
        checked_lanes = [current_lane]
        if bool(physical[24]) and target_lane != current_lane:
            checked_lanes.append(target_lane)
        gaps = [float(physical[lane * 6 + 1]) for lane in checked_lanes
                if bool(physical[lane * 6])]
        debug = native_info["joint_debug"]
        started = bool(debug["before"]["target_lane"] != debug["targets_applied"]["target_lane"])
        changed = bool(debug["before"]["target_speed_index"] != debug["targets_applied"]["target_speed_index"])
        collision = bool(raw.vehicle.crashed)
        off_road = not bool(raw.vehicle.on_road)
        terminated = bool(collision or off_road)
        # 失败优先；40秒同刻失败不能被标记为成功时间完成。
        truncated = bool(raw.time >= raw.config["duration"] and not terminated)
        self._needs_reset = terminated or truncated
        reward, terms, diagnostics = compute_reward(
            dt=dt, longitudinal_displacement=displacement,
            ego_longitudinal_speed=float(physical[18]), front_bumper_gaps=gaps,
            failure=terminated, lane_change_started=started, target_speed_changed=changed,
        )
        self.episode_reward += reward
        self.episode_steps += 1
        self.invalid_action_count += int(invalid)
        info = self._project_info(native_info, requested, executed, invalid)
        info["reward_terms"] = terms
        info["reward_diagnostics"] = {
            **diagnostics, "dt": float(dt), "longitudinal_displacement": displacement,
            "ego_longitudinal_speed": float(physical[18]), "front_bumper_gaps": gaps,
            "checked_lanes": checked_lanes,
            "lane_change_started": started, "target_speed_changed": changed,
        }
        info["native_reward_diagnostic"] = float(native_reward)
        return observation.copy(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if self._needs_reset and self.simulator.builder is None:
            raise gym.error.ResetNeeded("请先 reset 再渲染")
        return self.simulator.render()

    def close(self) -> None:
        try:
            self.simulator.close()
        finally:
            self._needs_reset = True
