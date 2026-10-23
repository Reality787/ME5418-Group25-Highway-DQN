"""复用 M1 的 overlay/演示策略；默认连接冻结正式环境，不是驾驶模型评测。"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import config


def project_demo_info(info: dict) -> dict:
    """只适配诊断字段名，复用控制器已有状态；不修改观测、奖励或物理状态。"""
    result = deepcopy(info)
    state = info["joint_debug"]["after"]
    result.update({
        "elapsed_time": info["simulation_time"],
        "simulation_time_s": info["physics_steps"] / config.SIMULATION_FREQUENCY_HZ,
        "ego_speed": state["actual_speed"],
        "target_speed": state["target_speed"],
        "target_speed_index": state["target_speed_index"],
        "current_lane": state["current_lane"], "target_lane": state["target_lane"],
        "manoeuvre_in_progress": state["lane_change_in_progress"],
        "crashed": info["collision"], "on_road": not info["off_road"],
    })
    return result



def render_project_frame(env):
    """复用 M1 离屏处理，仅改展示尺寸；不改变训练环境配置或车辆状态。"""
    from highway_m1.rendering import enable_rgb_rendering
    raw = env.simulator.unwrapped
    if raw.viewer is None:
        raw.configure({"screen_width": 1000, "screen_height": 300})
    enable_rgb_rendering(raw)
    return env.render()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=("project", "m1"), default="project")
    parser.add_argument("--policy", choices=("scripted", "random", "hold"), default="random")
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True, help="新的 PNG 文件路径，不覆盖旧图")
    args = parser.parse_args(argv)
    if args.seed < 0 or not 1 <= args.steps <= 200:
        parser.error("seed 必须非负，steps 必须在 1..200")
    if args.output.suffix.lower() != ".png" or args.output.exists():
        parser.error("output 必须为尚不存在的 .png 路径")
    # 只导入已检查的展示函数，不自动运行队友的录制/训练脚本。
    from teammates.m1.demo import DemonstrationPolicy, overlay
    from PIL import Image
    if args.environment == "project":
        from environment import make_env
        env = make_env(render_mode="rgb_array", invalid_action_mode="raise")
        adapt = project_demo_info
    else:
        from highway_m1 import HighwayDecisionEnv
        env = HighwayDecisionEnv(config={"offscreen_rendering": True}, render_mode="rgb_array")
        adapt = lambda info: info
    policy = DemonstrationPolicy(args.policy, args.seed)
    total_reward = 0.0
    terminated = truncated = False
    try:
        _, info = env.reset(seed=args.seed)
        for count in range(1, args.steps + 1):
            action = policy(adapt(info))
            _, reward, terminated, truncated, info = env.step(action)
            if info["executed_action"] != action or info["invalid_action"]:
                raise RuntimeError("所选动作未按原编号执行")
            total_reward += reward
            if terminated or truncated:
                break
        scene = render_project_frame(env) if args.environment == "project" else env.render()
        frame = overlay(scene, adapt(info), reward, args.policy, args.seed,
                        config.TARGET_SPEEDS_MPS)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(args.output)
        print(json.dumps({"environment": args.environment, "steps": count,
                          "simulation_time_s": adapt(info)["simulation_time_s"],
                          "reward": total_reward, "terminated": terminated,
                          "truncated": truncated, "output": str(args.output),
                          "note": "M1 策略/可视化复用演示；两种环境奖励和动力学不等价，非模型性能"},
                         ensure_ascii=False))
    finally:
        env.close()


if __name__ == "__main__":
    main()
