"""统一物理指标：失败回合照常保留，距离分母固定为 27.80×40=1112 米。

仅读取真实环境状态，不推进仿真、不修改控制目标，也不根据奖励反推位移。
所有方法后续都应使用同一指标；选模分数只是工程规则，不构成安全保证。
"""

from __future__ import annotations

import math
from statistics import fmean

import config


class EpisodeMetrics:
    """在正式环境 reset 之后创建，每次真实 step 后恰好 update 一次。"""

    def __init__(self, env) -> None:
        raw = env.simulator.unwrapped
        self._environment = env
        self._previous_position = raw.vehicle.position.copy()
        self._road_direction = env.simulator.builder._lanes()[0].direction.copy()
        self._start_time = float(raw.time)
        self._previous_time = self._start_time
        self._start_frames = int(raw.steps)
        self._previous_frames = self._start_frames
        self.distance = 0.0
        self.reward = 0.0
        self.steps = 0
        self.stopped_time = 0.0
        self.lane_change_count = 0
        self.speed_change_count = 0
        self.collision = False
        self.off_road = False
        self.terminated = False
        self.truncated = False
        self._finished = False

    def update(self, env, reward: float, terminated: bool, truncated: bool, info: dict) -> None:
        """停车比例按步后纵速 <1m/s 的实际物理时长统计；仅作诊断。"""
        if env is not self._environment:
            raise ValueError("不能把其他环境的 step 记入当前回合")
        if self._finished:
            raise ValueError("回合已结束，不能继续更新指标")
        if not math.isfinite(float(reward)):
            raise ValueError("指标不能接受非有限奖励")
        raw = env.simulator.unwrapped
        frame_delta = int(raw.steps) - self._previous_frames
        dt = frame_delta / float(raw.config["simulation_frequency"])
        if dt <= 0:
            raise ValueError("每个真实 step 后只能记录一次指标，物理时间必须前进")
        distance_delta = float((raw.vehicle.position - self._previous_position) @ self._road_direction)
        speed = float(raw.vehicle.velocity @ self._road_direction)
        if not all(math.isfinite(value) for value in (dt, distance_delta, speed)):
            raise ValueError("物理指标包含非有限值")
        # 从联合控制器实际已应用的目标变化统计事件，而不是从动作标签猜测。
        debug = info["joint_debug"]
        before, after = debug["before"], debug["targets_applied"]
        self.lane_change_count += int(before["target_lane"] != after["target_lane"])
        self.speed_change_count += int(before["target_speed_index"] != after["target_speed_index"])
        self.distance += distance_delta
        self.reward += float(reward)
        self.steps += 1
        if speed < config.TRAINING_CONFIG["stopped_speed_threshold_mps"]:
            self.stopped_time += dt
        self.collision = self.collision or bool(info["collision"])
        self.off_road = self.off_road or bool(info["off_road"])
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        self._finished = self.terminated or self.truncated
        self._previous_position = raw.vehicle.position.copy()
        self._previous_frames = int(raw.steps)
        self._previous_time = float(raw.time)

    def finish(self) -> dict:
        """允许读取未完成回合，但它绝不计为 completion；不裁剪进度。"""
        elapsed = (self._previous_frames - self._start_frames) / float(
            self._environment.simulator.unwrapped.config["simulation_frequency"]
        )
        failure = self.collision or self.off_road
        completion = bool(
            not failure and not self.terminated and self.truncated
            and self._previous_time >= config.EPISODE_DURATION_SECONDS - 1e-9
        )
        return {
            "collision": bool(self.collision), "off_road": bool(self.off_road),
            "failure": bool(failure), "completion": completion,
            "distance": float(self.distance),
            "normalized_progress": float(self.distance / config.TRAINING_CONFIG["progress_denominator_m"]),
            "reward": float(self.reward), "steps": int(self.steps),
            "physics_frames": self._previous_frames - self._start_frames,
            "elapsed_seconds": float(elapsed),
            "stopped_seconds": float(self.stopped_time),
            "stopped_fraction": float(self.stopped_time / elapsed) if elapsed else 0.0,
            "lane_change_count": self.lane_change_count,
            "speed_change_count": self.speed_change_count,
            "terminated": self.terminated, "truncated": self.truncated,
            "ended": self._finished,
        }


def aggregate_metrics(records: list[dict]) -> dict:
    """所有回合等权聚合，包括失败回合；禁止只统计成功样本。"""
    if not records:
        raise ValueError("至少需要一个回合才能汇总指标")
    result = {"episodes": len(records)}
    for key in ("collision", "off_road", "failure", "completion"):
        result[f"{key}_rate"] = fmean(float(record[key]) for record in records)
    for key in ("distance", "normalized_progress", "reward", "stopped_fraction",
                "elapsed_seconds", "lane_change_count", "speed_change_count"):
        result[f"mean_{key}"] = fmean(float(record[key]) for record in records)
    score = result["mean_normalized_progress"] - config.TRAINING_CONFIG["validation_failure_weight"] * result["failure_rate"]
    result["validation_score"] = score
    result["score"] = score
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("回合汇总包含非有限值")
    return result


def is_better(candidate: dict, best: dict | None) -> bool:
    """先比 score，精确同分比碰撞率，再比更早的环境步；不临时调权。"""
    if best is None:
        return True
    candidate_score = candidate.get("validation_score", candidate.get("score"))
    best_score = best.get("validation_score", best.get("score"))
    if candidate_score != best_score:
        return candidate_score > best_score
    if candidate["collision_rate"] != best["collision_rate"]:
        return candidate["collision_rate"] < best["collision_rate"]
    return candidate["env_steps"] < best["env_steps"]
