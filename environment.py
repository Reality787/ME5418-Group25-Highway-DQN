"""阶段 1A 原生入口、阶段 1B-1/1B-2 调试入口与正式项目环境入口。

make_stock_env 使用真实 HighwayEnv，保留原生观测、五动作和奖励。
make_joint_debug_env 提供真实九动作控制，但仍保留原生观测与奖励。
make_observation_debug_env 提供真实 34 维观测和 100m 感知，仍用原生调试奖励。
HighwayLaneEnv 延迟导出正式 Gymnasium Env；make_env 创建同一正式环境。
导入本模块不会创建仿真器或窗口，第三方依赖仅在调用工厂函数时导入。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import config

if TYPE_CHECKING:
    import gymnasium as gym


def make_stock_env(render_mode: str | None = None) -> gym.Env:
    """创建真实 highway-v0，仅用于原生仿真检查，不是最终九动作环境。

    场景参数读取 config.py；不改写原生观测、动作空间、速度档位或奖励。
    human 模式只额外启用实时播放节奏，不改变仿真频率或物理过程。
    调用者负责 reset、step 和 finally 中的 close。
    """
    import gymnasium as gym
    import highway_env  # 导入时注册 highway-v0；不能省略这一步。

    stock_config = {
        "lanes_count": config.LANE_COUNT,
        "vehicles_count": config.INITIAL_SURROUNDING_VEHICLE_COUNT,
        "controlled_vehicles": config.CONTROLLED_VEHICLE_COUNT,
        "duration": config.EPISODE_DURATION_SECONDS,
        "policy_frequency": config.POLICY_FREQUENCY_HZ,
        "simulation_frequency": config.SIMULATION_FREQUENCY_HZ,
        "offroad_terminal": "off_road" in config.TERMINATION_EVENTS,
    }
    if render_mode == "human":
        stock_config["real_time_rendering"] = True
    return gym.make("highway-v0", config=stock_config, render_mode=render_mode)



def make_joint_debug_env(render_mode: str | None = None) -> gym.Env:
    """阶段 1B-1 九动作调试入口；仍使用原生观测、奖励和感知。

    JointActionController 可供最终环境复用；本入口仍保留原生调试奖励；正式入口为 make_env。
    延迟导入确保仅查看 --help 或导入本模块时不强制加载仿真依赖。
    """
    from joint_actions import JointDebugEnv

    env = make_stock_env(render_mode=render_mode)
    try:
        return JointDebugEnv(env)
    except BaseException:
        env.close()
        raise


def make_observation_debug_env(render_mode: str | None = None) -> gym.Env:
    """阶段 1B-2：34维/100m观测包装现有九动作；保留原生调试奖励。"""
    from observation import ObservationDebugEnv

    env = make_joint_debug_env(render_mode=render_mode)
    try:
        return ObservationDebugEnv(env)
    except BaseException:
        env.close()
        raise


def make_env(render_mode: str | None = None, invalid_action_mode: str = "hold") -> gym.Env:
    """创建正式34维/九动作/项目奖励环境，不需要学习模块或模型文件。"""
    from project_environment import HighwayLaneEnv

    return HighwayLaneEnv(render_mode=render_mode, invalid_action_mode=invalid_action_mode)


def __getattr__(name: str):
    """延迟导出真实 Gym Env 类，保留仅导入入口/查看帮助无需仿真依赖的行为。"""
    if name == "HighwayLaneEnv":
        from project_environment import HighwayLaneEnv

        globals()[name] = HighwayLaneEnv
        return HighwayLaneEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
