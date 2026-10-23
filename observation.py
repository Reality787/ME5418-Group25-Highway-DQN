"""阶段 1B-2：从真实状态读取 34 维观测；不包含奖励或驾驶策略。

固定三条平行直车道，按实际编号 0/1/2 排列，工程约定见 config 与规格文档。
读取不推进仿真、不修改控制目标、不消费随机数。诊断与策略输入分开返回。
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from highway_env.road.lane import StraightLane

import config
from joint_actions import JointActionController, JointDebugEnv


def make_observation_space() -> gym.spaces.Box:
    """逐特征上下界与归一化配置一致；不是原生 (5, 5) 观测的变形。"""
    return gym.spaces.Box(
        low=np.asarray(config.OBSERVATION_LOW, dtype=np.float32),
        high=np.asarray(config.OBSERVATION_HIGH, dtype=np.float32),
        dtype=np.float32,
    )


def _finite(values, description: str) -> np.ndarray:
    """错误物理值明确失败；不能用 clip 或 nan_to_num 掩盖 NaN/Inf。"""
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"观测中的 {description} 包含非有限值")
    return values


class ObservationBuilder:
    """复用现有控制器的换道判定，从道路/车辆状态独立构造观测。

    build(previous_action) 不维护历史，历史由交互包装器在成功 step 后维护。
    使用库 vehicle.velocity 的航向速度，不将其称为含转向侧偏的位移积分速度。
    """

    def __init__(self, controller: JointActionController) -> None:
        self.controller = controller
        self.observation_space = make_observation_space()
        self.feature_names = config.OBSERVATION_FEATURE_NAMES

    def _lanes(self):
        ego = self.controller.vehicle
        start, end, _ = ego.lane_index
        lanes = ego.road.network.graph[start][end]
        if len(lanes) != config.LANE_COUNT or any(
            not isinstance(lane, StraightLane) for lane in lanes
        ):
            raise ValueError("34 维观测目前仅支持三条同向平行直车道")
        if any(not np.allclose(lane.direction, lanes[0].direction) for lane in lanes):
            raise ValueError("34 维观测要求三条车道方向相同")
        return lanes

    @staticmethod
    def _lane_id(position: np.ndarray, lanes) -> int:
        # 只看当前位置到中心线的横向距离；平局取低编号，不读取隐藏目标或缓存车道。
        return min(
            range(len(lanes)),
            key=lambda index: (abs(lanes[index].local_coordinates(position)[1]), index),
        )

    @staticmethod
    def _half_extent(vehicle, road_heading: float) -> float:
        """矩形车身沿道路方向的投影半长；不是精确二维碰撞距离。"""
        length, width, heading = _finite(
            [vehicle.LENGTH, vehicle.WIDTH, vehicle.heading], "车身尺寸/航向"
        )
        if length <= 0 or width <= 0:
            raise ValueError("车辆长度和宽度必须为正")
        angle = heading - road_heading
        return float((length * abs(np.cos(angle)) + width * abs(np.sin(angle))) / 2)

    def build(self, previous_action: int | None = None) -> tuple[np.ndarray, dict]:
        """返回 (float32[34], diagnostics)，同一状态重复读取完全一致。

        先按中心欧氏距离 <=100m 过滤，再在每条车道选最近前/后车。
        ds=0 归前车；同纵向距离按 road.vehicles 顺序选首个，排除自车。
        """
        if previous_action is not None:
            if isinstance(previous_action, (bool, np.bool_)) or not isinstance(
                previous_action, (int, np.integer)
            ):
                raise TypeError("上一联合动作必须是整数或 None")
            if not 0 <= previous_action < config.ACTION_COUNT:
                raise ValueError("上一联合动作超出 0～8")

        ego = self.controller.vehicle
        lanes = self._lanes()
        reference = lanes[0]
        position = _finite(ego.position, "自车位置")
        _finite([ego.speed, ego.heading], "自车速度/航向")
        velocity = _finite(ego.velocity, "自车速度")
        direction = reference.direction
        lateral_direction = reference.direction_lateral
        ego_s = reference.local_coordinates(position)[0]
        road_heading = reference.heading_at(ego_s)
        ego_v_long = float(velocity @ direction)
        ego_v_lateral = float(velocity @ lateral_direction)
        ego_half = self._half_extent(ego, road_heading)
        current_lane = self._lane_id(position, lanes)
        target_lane = ego.target_lane_index[2]
        if not 0 <= target_lane < config.LANE_COUNT:
            raise ValueError("目标车道编号超出当前三车道范围")
        lateral_offset = lanes[current_lane].local_coordinates(position)[1]

        # 缺失编码是确定的输入约定；100m 在 raw_values 中是缺失间距的哨兵值。
        raw = np.zeros(config.OBSERVATION_DIM, dtype=float)
        raw[:config.TRAFFIC_FEATURE_DIM] = np.tile(
            [0.0, config.OBS_GAP_SCALE_M, 0.0], 6
        )
        nearest = {}
        for order, other in enumerate(ego.road.vehicles):
            if other is ego:
                continue
            other_position = _finite(other.position, "其他车辆位置")
            distance = float(np.linalg.norm(other_position - position))
            if distance > config.PERCEPTION_RANGE_METERS:
                continue
            lane_id = self._lane_id(other_position, lanes)
            ds = float(reference.local_coordinates(other_position)[0] - ego_s)
            side = 0 if ds >= 0 else 1
            key = (lane_id, side)
            rank = (abs(ds), order)
            if key not in nearest or rank < nearest[key][0]:
                nearest[key] = (rank, other)

        for (lane_id, side), (rank, other) in nearest.items():
            _finite([other.speed, other.heading], "可见车辆速度/航向")
            other_velocity = _finite(other.velocity, "可见车辆速度")
            gap = max(0.0, rank[0] - ego_half - self._half_extent(other, road_heading))
            relative_speed = float(other_velocity @ direction) - ego_v_long
            offset = lane_id * 6 + side * 3
            raw[offset:offset + 3] = (1.0, gap, relative_speed)

        raw[18:25] = (
            ego_v_long, ego_v_lateral, lateral_offset, current_lane, target_lane,
            ego.target_speed, float(self.controller.is_lane_change_in_progress()),
        )
        if previous_action is not None:
            raw[25 + int(previous_action)] = 1.0
        _finite(raw, "原始特征")
        scaled = raw / np.asarray(config.OBSERVATION_SCALES, dtype=float)
        low = np.asarray(config.OBSERVATION_LOW, dtype=float)
        high = np.asarray(config.OBSERVATION_HIGH, dtype=float)
        bounded = np.clip(scaled, low, high)
        clipped = []
        for index in np.flatnonzero((scaled < low) | (scaled > high)):
            clipped.append({
                "index": int(index), "name": self.feature_names[index],
                "raw_value": float(raw[index]),
                "unclipped_normalized": float(scaled[index]),
                "normalized_value": float(bounded[index]),
                "low": float(low[index]), "high": float(high[index]),
            })
        observation = bounded.astype(np.float32)
        if not self.observation_space.contains(observation):
            raise ValueError("构造的观测与声明的 Box 不一致")
        return observation, {
            "feature_names": self.feature_names,
            "raw_values": raw.copy(),
            "normalized_values": observation.copy(),
            "clipped_features": clipped,
        }


class ObservationDebugEnv(gym.Wrapper):
    """在 JointDebugEnv 外替换观测；动作、掩码、真实 step 与原生奖励均复用。

    本类不是最终 HighwayLaneEnv。诊断仅放 info，不拼进 34 维策略输入。
    """

    def __init__(self, env: JointDebugEnv) -> None:
        super().__init__(env)
        self.observation_space = make_observation_space()
        self.builder: ObservationBuilder | None = None
        self.previous_action: int | None = None

    @property
    def controller(self) -> JointActionController:
        return self.env._require_controller()

    def get_action_mask(self) -> np.ndarray:
        return self.env.get_action_mask()

    def get_debug_state(self) -> dict:
        return self.env.get_debug_state()

    def get_observation(self) -> tuple[np.ndarray, dict]:
        """只读当前状态，供测试与可读诊断使用；不额外调用 step。"""
        if self.builder is None:
            raise gym.error.ResetNeeded("请先调用 observation-debug reset()")
        return self.builder.build(self.previous_action)

    def _observation_and_info(self, info):
        observation, diagnostics = self.get_observation()
        info = dict(info)
        info["observation_debug"] = diagnostics
        return observation, info

    def reset(self, *, seed=None, options=None):
        _, info = self.env.reset(seed=seed, options=options)
        self.previous_action = None
        self.builder = ObservationBuilder(self.controller)
        self.action_space = self.env.action_space
        return self._observation_and_info(info)

    def step(self, action):
        # 内层严格校验；拒绝动作时本层历史保持原值。
        _, reward, terminated, truncated, info = self.env.step(action)
        self.previous_action = int(info["action"])
        observation, info = self._observation_and_info(info)
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        try:
            self.env.close()
        finally:
            self.builder = None
            self.previous_action = None
