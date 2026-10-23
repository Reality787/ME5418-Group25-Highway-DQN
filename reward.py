"""阶段 1B-3 项目奖励纯函数。

权重和阈值是用户本轮指定的初始工程配置，尚未通过训练验证。
本模块仅接收环境已提取的物理量，不访问仿真、动作控制器或随机数。
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Sequence

import config


def _finite_number(value: object, name: str) -> float:
    """拒绝布尔、非数值及 NaN/Inf，避免无声地产生无效奖励。"""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} 必须是有限实数")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} 必须是有限实数") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} 必须是有限实数")
    return result


def compute_reward(
    *,
    dt: float,
    longitudinal_displacement: float,
    ego_longitudinal_speed: float,
    front_bumper_gaps: Sequence[float],
    failure: bool,
    lane_change_started: bool,
    target_speed_changed: bool,
) -> tuple[float, dict[str, float], dict[str, float | None]]:
    """按真实推进时间计算奖励，返回总值、五项加权贡献和诊断。

    环境负责使用共用的 100 米感知物理量提取逻辑选择相关前车：
    正常行驶选当前车道；换道期间选当前车道和目标车道。
    本函数在传入间距中选最短 THW，不从裁剪后的观测反推间距。
    failure 已合并碰撞和离路；事件标记必须描述本步实际执行的目标变化。
    该车道选择与 THW 计算是初始工程近似，不表示驾驶安全保证。
    """
    dt = _finite_number(dt, "dt")
    if dt <= 0:
        raise ValueError("dt 必须大于 0，不能使用替代分母掩盖未推进仿真")
    displacement = _finite_number(longitudinal_displacement, "longitudinal_displacement")
    ego_speed = _finite_number(ego_longitudinal_speed, "ego_longitudinal_speed")
    for name, value in (
        ("failure", failure),
        ("lane_change_started", lane_change_started),
        ("target_speed_changed", target_speed_changed),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} 必须是 bool")

    # 复制为本地标量，不修改调用方的间距列表；负间距必须由物理量提取层截为0。
    gaps = tuple(_finite_number(gap, f"front_bumper_gaps[{index}]")
                 for index, gap in enumerate(front_bumper_gaps))
    if any(gap < 0 for gap in gaps):
        raise ValueError("front_bumper_gaps 中的保险杠间距不能为负数")

    progress_fraction = min(max(displacement / (config.DESIRED_SPEED_MPS * dt), 0.0), 1.0)
    min_thw = min(gaps) / max(ego_speed, config.REWARD_MIN_SPEED_MPS) if gaps else None
    if min_thw is not None and not math.isfinite(min_thw):
        raise ValueError("THW 计算溢出，不能输出非有限诊断")
    headway_penalty = (
        min(max(1.0 - min_thw / config.REWARD_THW_THRESHOLD_S, 0.0), 1.0)
        if min_thw is not None else 0.0
    )
    terms = {
        "progress": dt * config.REWARD_PROGRESS_WEIGHT * progress_fraction,
        "headway": -dt * config.REWARD_HEADWAY_WEIGHT * headway_penalty,
        "failure": -config.REWARD_FAILURE_COST * int(failure),
        "lane_change": -config.REWARD_LANE_CHANGE_COST * int(lane_change_started),
        "target_speed": -config.REWARD_TARGET_SPEED_COST * int(target_speed_changed),
    }
    reward = sum(terms.values())
    if not math.isfinite(reward) or not all(math.isfinite(value) for value in terms.values()):
        raise ValueError("奖励计算溢出，不能输出非有限奖励")
    diagnostics = {
        "progress_fraction": progress_fraction,
        "headway_penalty": headway_penalty,
        "min_thw": min_thw,
    }
    return reward, terms, diagnostics
