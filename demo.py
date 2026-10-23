"""原生/联合动作/观测调试与正式项目演示；导入本模块不会启动回合。"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import math
import sys
from typing import Sequence

import config
from environment import make_env, make_joint_debug_env, make_observation_debug_env, make_stock_env


def positive_integer(value: str) -> int:
    """为回合数提供明确的命令行错误信息。"""
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("episodes 必须是正整数") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("episodes 必须是正整数")
    return number


def print_runtime_details(env, mode: str = "stock") -> None:
    """输出实际安装版本及配置，不把项目九动作配置用于原生环境。"""
    if mode == "stock":
        print("当前模式：原生仿真检查，未接入项目九动作。")
    elif mode == "joint-debug":
        print("当前模式：joint-debug 九动作控制检查，不是最终 HighwayLaneEnv。")
        print("本模式仍用原生观测和原生调试奖励；34维/100米观测请使用 observation-debug，项目奖励请使用 project 模式。")
        print("joint-debug 禁用方向键驾驶，防止绕过联合动作；可用 Ctrl+C 或关闭窗口停止。")
    else:
        print("当前模式：observation-debug，真实34维观测/100米感知/九动作；不是最终 HighwayLaneEnv。")
        print("本模式奖励仍是原生调试奖励；项目奖励请使用 project，学习模块尚未实现。")
        print("感知按中心欧氏距离<=100m过滤；观测速度使用库的航向速度，未计转向侧偏。")
        print("方向键驾驶已禁用，可用 Ctrl+C 或关闭窗口停止。")
    print(f"解释器路径：{sys.executable}")
    print(f"Python 版本：{sys.version.split()[0]}")
    for package in ("gymnasium", "highway-env", "numpy", "pygame-ce", "matplotlib"):
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            # 元数据缺失不能当作已验证；pygame 也可能由其他发行包提供。
            version = "未找到发行包元数据，版本未验证"
        print(f"依赖：{package}={version}")

    keys = (
        "lanes_count", "vehicles_count", "controlled_vehicles", "duration",
        "policy_frequency", "simulation_frequency", "offroad_terminal",
        "real_time_rendering",
    )
    effective = {key: env.unwrapped.config[key] for key in keys}
    print("实际生效的场景参数：" + json.dumps(effective, ensure_ascii=False))
    label = "项目观测" if mode == "observation-debug" else "原生"
    print(f"{label} observation_space：{env.observation_space}")
    if mode == "stock":
        print(f"原生 action_space：{env.action_space}")
        print(f"原生动作映射：{env.unwrapped.action_type.actions}")
    else:
        print(f"对外联合 action_space：{env.action_space}")
        print(f"联合动作映射：{config.ACTION_NAMES}")
        print(f"内层原生 action_space：{env.unwrapped.action_space}，仅用原生 IDLE 推进")
    print("以下是随机动作的仿真记录，不是训练结果或模型性能。", flush=True)


def run_stock_demo(render_mode: str | None, episodes: int, seed: int) -> None:
    """运行有限回合；正常结束、异常或 Ctrl+C 均通过 finally 关闭环境。"""
    env = None
    try:
        env = make_stock_env(render_mode=render_mode)
        print_runtime_details(env)
        effective = env.unwrapped.config
        # 多留一步容纳时间累加的浮点误差；耗尽时报告错误，不伪造截断。
        max_decisions = math.ceil(
            effective["duration"] * effective["policy_frequency"]
        ) + 1
        for episode_index in range(episodes):
            episode_seed = seed + episode_index
            observation, info = env.reset(seed=episode_seed)
            # reset 会重建 action_space，因此必须在 reset 之后播种。
            env.action_space.seed(episode_seed)
            print(
                f"回合 {episode_index + 1}/{episodes} 开始："
                f"seed={episode_seed}，action_seed={episode_seed}",
                flush=True,
            )
            total_reward = 0.0
            for decision_count in range(1, max_decisions + 1):
                # 仅从原生空间采样；不使用 config.ACTION_NAMES 的九动作编号。
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                # human 的 reset/step 自动渲染，无需重复调用 render。
                if terminated or truncated:
                    print(
                        f"回合 {episode_index + 1}/{episodes} 结束："
                        f"seed={episode_seed}，决策次数={decision_count}，"
                        f"累计原生奖励={total_reward:.6f}，"
                        f"terminated={bool(terminated)}，truncated={bool(truncated)}",
                        flush=True,
                    )
                    break
            else:
                raise RuntimeError(
                    f"回合 seed={episode_seed} 在 {max_decisions} 次决策后仍未结束，"
                    "请检查真实环境的时间上限配置。"
                )
    finally:
        if env is not None:
            env.close()



def format_joint_state(state: dict) -> str:
    """只格式化控制器快照；掩码逻辑始终来自 get_action_mask。"""
    mask = "".join(str(int(value)) for value in state["action_mask"])
    return (
        f"t={state['simulation_time']:.3f}s physics_steps={state['physics_steps']} "
        f"lane={state['current_lane']} target_lane={state['target_lane']} "
        f"speed={state['actual_speed']:.6f} target_speed={state['target_speed']:.2f} "
        f"target_index={state['target_speed_index']} "
        f"changing={state['lane_change_in_progress']} "
        f"lateral_error={state['lateral_error']:.4f} "
        f"heading_error={state['heading_error']:.4f} mask={mask}"
    )


def print_observation_details(info: dict) -> None:
    """可选的人类可读表，缺失车辆的 raw gap=100 是哨兵值。"""
    debug = info["observation_debug"]
    print("index | feature_name | raw_value (unit) | normalized_value")
    for index, name in enumerate(debug["feature_names"]):
        print(
            f"{index:2d} | {name} | {debug['raw_values'][index]:.6f} "
            f"({config.OBSERVATION_UNITS[index]}) | {debug['normalized_values'][index]:.6f}"
        )
    print("缺失交通槽 raw gap=100 是编码哨兵，须与 presence 一起读取。", flush=True)


def report_observation_clipping(info: dict) -> None:
    """即使不打印完整表，裁剪仍明确报告，不能静默掩盖超界值。"""
    clipped = info["observation_debug"]["clipped_features"]
    if clipped:
        print("观测裁剪诊断：" + json.dumps(clipped, ensure_ascii=False), flush=True)


def run_joint_debug_demo(
    render_mode: str | None, episodes: int, seed: int,
    *, mode: str = "joint-debug", observation_steps: int = 2,
) -> None:
    """仅从结构掩码允许的动作中可复现地采样，输出三阶段真实控制快照。"""
    import numpy as np

    env = None
    try:
        factory = make_observation_debug_env if mode == "observation-debug" else make_joint_debug_env
        env = factory(render_mode=render_mode)
        print_runtime_details(env, mode=mode)
        effective = env.unwrapped.config
        max_decisions = math.ceil(
            effective["duration"] * effective["policy_frequency"]
        ) + 1
        for episode_index in range(episodes):
            episode_seed = seed + episode_index
            observation, info = env.reset(seed=episode_seed)
            env.action_space.seed(episode_seed)
            random = np.random.default_rng(episode_seed)
            if episode_index == 0:
                print(
                    "复用的底层控制参数（不是 proposal 限值）："
                    + json.dumps(env.controller.get_low_level_parameters(), ensure_ascii=False)
                )
            debug = info["joint_debug"]
            print(
                f"[reset] episode={episode_index + 1} seed={episode_seed} "
                f"sampler_seed={episode_seed} action=None name=None native_action=None\n"
                f"  after: {format_joint_state(debug['after'])}", flush=True,
            )
            if mode == "observation-debug":
                print_observation_details(info)
                report_observation_clipping(info)
            total_reward = 0.0
            for decision_count in range(1, max_decisions + 1):
                valid_actions = np.flatnonzero(env.get_action_mask())
                action = int(random.choice(valid_actions))
                observation, reward, terminated, truncated, info = env.step(action)
                debug = info["joint_debug"]
                if mode == "joint-debug" or decision_count <= observation_steps:
                    print(
                        f"[step] action={debug['action_id']} name={debug['action_name']} "
                        f"native_action={debug['native_action_id']}(IDLE)\n"
                        f"  before:  {format_joint_state(debug['before'])}\n"
                        f"  targets: {format_joint_state(debug['targets_applied'])}\n"
                        f"  after:   {format_joint_state(debug['after'])}\n"
                        f"  native_debug_reward={float(reward):.6f} "
                        f"terminated={bool(terminated)} truncated={bool(truncated)}",
                        flush=True,
                    )
                if mode == "observation-debug":
                    if decision_count <= observation_steps:
                        print_observation_details(info)
                    report_observation_clipping(info)
                total_reward += float(reward)
                if terminated or truncated:
                    print(
                        f"{mode} 回合 {episode_index + 1}/{episodes} 结束："
                        f"seed={episode_seed}，决策次数={decision_count}，"
                        f"累计原生调试奖励={total_reward:.6f}，"
                        f"terminated={bool(terminated)}，truncated={bool(truncated)}",
                        flush=True,
                    )
                    break
            else:
                raise RuntimeError(f"{mode} 超过有限决策上限，真实环境没有返回结束标志")
    except KeyboardInterrupt:
        joint_env = env.env if env is not None and mode == "observation-debug" else env
        if joint_env is not None and joint_env.controller is not None:
            print(
                f"[interrupted] mode={mode} "
                + format_joint_state(env.get_debug_state()), flush=True,
            )
        raise
    finally:
        if env is not None:
            env.close()


def print_project_runtime_details(env) -> None:
    """正式环境使用自己的34维/九动作空间，仿真配置从内部仿真器读取。"""
    print("当前模式：project，HighwayLaneEnv 的真实34维观测、100米感知、九动作和项目奖励。")
    print("非法动作模式：hold；本演示仅采样合法掩码动作，正常不应触发回退。")
    print(f"解释器路径：{sys.executable}")
    print(f"Python 版本：{sys.version.split()[0]}")
    for package in ("gymnasium", "highway-env", "numpy", "pygame-ce", "matplotlib"):
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            version = "未找到发行包元数据，版本未验证"
        print(f"依赖：{package}={version}")
    raw = env.simulator.unwrapped
    keys = ("lanes_count", "vehicles_count", "controlled_vehicles", "duration",
            "policy_frequency", "simulation_frequency", "offroad_terminal")
    print("实际生效的场景参数：" + json.dumps(
        {key: raw.config[key] for key in keys}, ensure_ascii=False))
    print(f"项目 observation_space：{env.observation_space}")
    print(f"项目 action_space：{env.action_space}")
    print(f"项目感知范围：{config.PERCEPTION_RANGE_METERS}m（中心欧氏距离）")
    reward_settings = {
        key: getattr(config, key) for key in (
            "REWARD_PROGRESS_WEIGHT", "REWARD_HEADWAY_WEIGHT", "REWARD_FAILURE_COST",
            "REWARD_LANE_CHANGE_COST", "REWARD_TARGET_SPEED_COST",
            "REWARD_THW_THRESHOLD_S", "REWARD_MIN_SPEED_MPS",
        )
    }
    print("奖励初始工程配置（非proposal原文数值、非最优参数）：" + json.dumps(reward_settings, ensure_ascii=False))
    print("r = dt*(1.0*p - 2.0*h) - 10.0*failure - 0.05*lane_change_started - 0.02*target_speed_changed")
    print("以下为随机合法动作的环境接口检查，不是训练结果或模型性能。", flush=True)


def run_project_demo(
    render_mode: str | None, episodes: int, seed: int, *, reward_steps: int = 2,
) -> None:
    """正式环境独立演示；项目奖励不与原生诊断奖励相加。"""
    import numpy as np

    env = None
    last_info = None
    try:
        env = make_env(render_mode=render_mode, invalid_action_mode="hold")
        print_project_runtime_details(env)
        effective = env.simulator.unwrapped.config
        max_decisions = math.ceil(effective["duration"] * effective["policy_frequency"]) + 1
        for episode_index in range(episodes):
            episode_seed = seed + episode_index
            observation, info = env.reset(seed=episode_seed)
            last_info = info
            env.action_space.seed(episode_seed)
            sampler = np.random.default_rng(episode_seed)
            print(
                f"[project reset] episode={episode_index + 1}/{episodes} "
                f"seed={episode_seed} sampler_seed={episode_seed} "
                f"observation_shape={observation.shape} dtype={observation.dtype} "
                f"action_mask={info['action_mask'].astype(int).tolist()}", flush=True,
            )
            report_observation_clipping(info)
            total_reward = 0.0
            for decision_count in range(1, max_decisions + 1):
                action = int(sampler.choice(np.flatnonzero(env.get_action_mask())))
                observation, reward, terminated, truncated, info = env.step(action)
                last_info = info
                total_reward += float(reward)
                if decision_count <= reward_steps:
                    print(
                        f"[project step] step={decision_count} t={info['simulation_time']:.3f}s "
                        f"requested={info['requested_action']} executed={info['executed_action']} "
                        f"name={config.ACTION_NAMES[info['executed_action']]} "
                        f"invalid_action={info['invalid_action']} project_reward={reward:.6f} "
                        f"reward_terms={json.dumps(info['reward_terms'], ensure_ascii=False)} "
                        f"native_reward_diagnostic={info['native_reward_diagnostic']:.6f}",
                        flush=True,
                    )
                report_observation_clipping(info)
                if terminated or truncated:
                    print(
                        f"project 回合 {episode_index + 1}/{episodes} 结束："
                        f"seed={episode_seed}，决策次数={decision_count}，"
                        f"仿真时长={info['simulation_time']:.3f}s，累计项目奖励={total_reward:.6f}，"
                        f"collision={info['collision']}，off_road={info['off_road']}，"
                        f"terminated={bool(terminated)}，truncated={bool(truncated)}，"
                        f"非法动作回退次数={info['invalid_action_count']}", flush=True,
                    )
                    break
            else:
                raise RuntimeError("project 超过有限决策上限，正式环境没有返回结束标志")
    except KeyboardInterrupt:
        # 窗口可在某个物理子步关闭；只报告最后一个完整返回的决策，不虚报整回合完成。
        if last_info is not None:
            print(
                f"[project interrupted] 最后完整决策：steps={last_info['episode_steps']} "
                f"t={last_info['simulation_time']:.3f}s "
                f"episode_reward={last_info['episode_reward']:.6f}；本回合未完整完成。",
                flush=True,
            )
        raise
    finally:
        if env is not None:
            env.close()


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令行并运行指定仿真模式，不创建学习网络或训练循环。"""
    parser = argparse.ArgumentParser(
        description="ME5418：stock、joint-debug、observation-debug 与 project 环境演示。",
        epilog="project 使用34维/100米/九动作/项目奖励；其他模式保留原生调试奖励，所有模式均不训练。",
    )
    parser.add_argument(
        "--mode", choices=("stock", "joint-debug", "observation-debug", "project"), required=True,
        help="stock 原生五动作；joint-debug 九动作；observation-debug 加入34维/100米；project 再加入项目奖励",
    )
    parser.add_argument(
        "--render", choices=("none", "human"), default="none",
        help="none 无窗口；human 显示车辆运动（默认 none）",
    )
    parser.add_argument(
        "--episodes", type=positive_integer, default=1,
        help="回合数，必须为正整数（默认 1）",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="首回合整数种子，随后每回合加 1（默认 0）",
    )
    parser.add_argument(
        "--observation-steps", type=int, default=2,
        help="observation-debug 每回合打印初始表及前N步的完整表，N>=0（默认2）；其余步只报告裁剪和结束",
    )
    parser.add_argument(
        "--reward-steps", type=int, default=2,
        help="project 每回合仅打印前N步奖励分量，N>=0（默认2）；结束时始终打印汇总",
    )
    args = parser.parse_args(argv)
    if args.reward_steps < 0:
        parser.error("reward-steps 必须为非负整数")
    if args.observation_steps < 0:
        parser.error("observation-steps 必须为非负整数")
    if args.seed < 0:
        parser.error("Gymnasium 要求 seed 为非负整数")

    try:
        options = dict(
            render_mode=None if args.render == "none" else "human",
            episodes=args.episodes, seed=args.seed,
        )
        if args.mode == "stock":
            run_stock_demo(**options)
        elif args.mode == "project":
            run_project_demo(**options, reward_steps=args.reward_steps)
        else:
            run_joint_debug_demo(
                **options, mode=args.mode, observation_steps=args.observation_steps,
            )
    except KeyboardInterrupt:
        print("演示已停止（Ctrl+C 或窗口关闭），环境已关闭。", file=sys.stderr)
        return 130
    except ImportError as error:
        print(
            f"依赖不可用，真实仿真未验证：{error}\n"
            "请使用 me5418-highway 项目解释器；如需安装，请在该环境执行 "
            "python -m pip install -r requirements.txt。",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
