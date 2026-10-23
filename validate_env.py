"""检查正式 34维/九动作项目环境；独立于 DQN，不自动安装依赖。

运行每项检查并如实汇总；某项失败不会被计作通过，最终退出码为 1。
Gymnasium checker 直接检查 HighwayLaneEnv，不跳过到内层原生环境。
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import sys
import traceback
from typing import Sequence
import warnings


def positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("episodes 必须为正整数")
    return number


def check_observation(env, observation, info) -> None:
    """验证真实输出的类型、空间和当前状态掩码；不构造替代输出。"""
    import numpy as np

    assert isinstance(observation, np.ndarray), "观测必须为 numpy.ndarray"
    assert observation.shape == (34,) and observation.dtype == np.float32
    assert np.all(np.isfinite(observation)), "观测存在 NaN/Inf"
    assert env.observation_space.contains(observation), "观测超出声明空间"
    assert isinstance(info, dict), "info 必须为字典"
    mask = info["action_mask"]
    assert isinstance(mask, np.ndarray) and mask.shape == (9,) and mask.dtype == bool
    assert bool(mask[4]), "Keep + hold 必须满足结构约束"
    np.testing.assert_array_equal(mask, env.get_action_mask())


def check_reward(reward, info) -> None:
    assert isinstance(reward, (int, float)) and not isinstance(reward, bool)
    assert math.isfinite(reward), "奖励必须有限"
    terms = info["reward_terms"]
    assert set(terms) == {"progress", "headway", "failure", "lane_change", "target_speed"}
    assert all(math.isfinite(value) for value in terms.values())
    assert math.isclose(math.fsum(terms.values()), reward, rel_tol=1e-10, abs_tol=1e-10)


def check_gym_contract() -> None:
    """不修改第三方 checker，不筛选种子，不屏蔽 checker 的异常。"""
    from gymnasium.utils.env_checker import check_env
    from environment import HighwayLaneEnv

    env = HighwayLaneEnv(render_mode=None, invalid_action_mode="hold")
    try:
        assert env.unwrapped is env, "检查目标必须是正式项目环境自身"
        print(f"checker 目标：{type(env).__module__}.{type(env).__name__}，invalid_action_mode=hold")
        print(f"observation_space={env.observation_space}; action_space={env.action_space}")
        check_env(env, skip_render_check=False, skip_close_check=False)
    finally:
        env.close()
        env.close()


def check_masked_episodes(episodes: int, seed: int) -> None:
    """多回合真实随机交互；碰撞是正常回合结束，不能因此筛掉某个种子。"""
    import numpy as np
    import config
    from environment import make_env

    env = make_env(render_mode=None, invalid_action_mode="hold")
    try:
        assert env.action_space.n == config.ACTION_COUNT == 9
        for index in range(episodes):
            episode_seed = seed + index
            observation, info = env.reset(seed=episode_seed)
            check_observation(env, observation, info)
            assert np.count_nonzero(observation[25:]) == 0
            assert info["episode_steps"] == 0 and info["episode_reward"] == 0
            assert info["invalid_action_count"] == 0
            sampler = np.random.default_rng(episode_seed)
            total = 0.0
            previous_time = info["simulation_time"]
            limit = math.ceil(config.EPISODE_DURATION_SECONDS * config.POLICY_FREQUENCY_HZ) + 1
            for decision in range(1, limit + 1):
                action = int(sampler.choice(np.flatnonzero(env.get_action_mask())))
                old_observation, old_mask = observation, info["action_mask"]
                saved_observation, saved_mask = observation.copy(), old_mask.copy()
                result = env.step(action)
                assert isinstance(result, tuple) and len(result) == 5
                observation, reward, terminated, truncated, info = result
                check_observation(env, observation, info)
                check_reward(reward, info)
                np.testing.assert_array_equal(old_observation, saved_observation)
                np.testing.assert_array_equal(old_mask, saved_mask)
                expected_history = np.zeros(9, dtype=np.float32)
                expected_history[action] = 1
                np.testing.assert_array_equal(observation[25:], expected_history)
                assert isinstance(terminated, (bool, np.bool_))
                assert isinstance(truncated, (bool, np.bool_))
                assert info["requested_action"] == action == info["executed_action"]
                assert info["invalid_action"] is False and info["invalid_action_count"] == 0
                assert math.isclose(info["simulation_time"] - previous_time,
                                    1 / config.POLICY_FREQUENCY_HZ, abs_tol=1e-9)
                previous_time = info["simulation_time"]
                total += reward
                assert info["episode_steps"] == decision
                assert math.isclose(info["episode_reward"], total, abs_tol=1e-9)
                if terminated or truncated:
                    assert bool(terminated) == bool(info["collision"] or info["off_road"])
                    assert not (terminated and truncated), "失败优先，不能同时报告成功时间截断"
                    print(
                        f"masked episode seed={episode_seed} steps={decision} "
                        f"time={info['simulation_time']:.3f}s project_reward={total:.6f} "
                        f"collision={info['collision']} off_road={info['off_road']} "
                        f"terminated={terminated} truncated={truncated} fallbacks=0",
                        flush=True,
                    )
                    break
            else:
                raise AssertionError("达到有限决策上限仍未返回回合结束")
    finally:
        env.close()


def check_strict_mask_rejection(seed: int) -> None:
    """raise 模式拒绝结构非法动作后，检查物理状态、控制目标和历史没有变化。"""
    import numpy as np
    from environment import make_env

    env = make_env(render_mode=None, invalid_action_mode="raise")
    try:
        observation, info = env.reset(seed=seed, options={"initial_target_speed_index": 0})
        check_observation(env, observation, info)
        masked = np.flatnonzero(~env.get_action_mask())
        assert masked.size > 0
        rejected_action = int(masked[0])
        raw = env.simulator.unwrapped
        vehicle = env.controller.vehicle
        before = deepcopy((vehicle.position, vehicle.speed, vehicle.heading,
                           vehicle.target_lane_index, vehicle.target_speed,
                           vehicle.speed_index, raw.time, raw.steps,
                           env.simulator.previous_action, raw.np_random.bit_generator.state,
                           env.episode_steps, env.episode_reward, env.invalid_action_count))
        try:
            env.step(rejected_action)
        except ValueError:
            pass  # 此处明确预期 ValueError；其他异常会作为失败向外报告。
        else:
            raise AssertionError("raise 模式接受了被结构掩码禁止的动作")
        after = (vehicle.position, vehicle.speed, vehicle.heading,
                 vehicle.target_lane_index, vehicle.target_speed,
                 vehicle.speed_index, raw.time, raw.steps,
                 env.simulator.previous_action, raw.np_random.bit_generator.state,
                 env.episode_steps, env.episode_reward, env.invalid_action_count)
        np.testing.assert_array_equal(before[0], after[0])
        assert before[1:] == after[1:], "拒绝动作改变了状态"
        current_observation, _ = env.simulator.get_observation()
        np.testing.assert_array_equal(current_observation, observation)
        valid_action = int(np.flatnonzero(env.get_action_mask())[0])
        result = env.step(valid_action)
        check_observation(env, result[0], result[4])
        check_reward(result[1], result[4])
        print(f"raise 模式：拒绝结构非法 action={rejected_action} 且状态未改变；合法 action={valid_action} 真实推进通过")
    finally:
        env.close()


def check_rgb_array(seed: int) -> None:
    """独立验证实际 RGB 图像；它不能替代 human 桌面窗口验证。"""
    import numpy as np
    from environment import make_env

    env = make_env(render_mode="rgb_array", invalid_action_mode="hold")
    try:
        env.reset(seed=seed)
        env.step(4)
        frame = env.render()
        assert isinstance(frame, np.ndarray) and frame.dtype == np.uint8
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert frame.shape[0] > 0 and frame.shape[1] > 0
        print(f"rgb_array：真实图像 shape={frame.shape}, dtype={frame.dtype}")
    finally:
        env.close()
        env.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ME5418 正式环境验收：Gymnasium checker 与真实掩码交互，不启动训练。")
    parser.add_argument("--episodes", type=positive_integer, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("seed 必须是非负整数")
    print(f"验收解释器：{sys.executable}", flush=True)
    print("目标：HighwayLaneEnv，真实34维/九动作/项目奖励；checker采用默认hold模式。", flush=True)
    checks = (
        ("Gymnasium check_env（正式 hold 环境）", check_gym_contract),
        ("多回合掩码交互、空间、历史、奖励分量", lambda: check_masked_episodes(args.episodes, args.seed)),
        ("raise 模式及结构掩码拒绝", lambda: check_strict_mask_rejection(args.seed)),
        ("rgb_array 与重复 close", lambda: check_rgb_array(args.seed)),
    )
    passed, failed, reported_warnings = [], [], []
    for name, check in checks:
        print(f"[检查] {name}", flush=True)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            try:
                check()
            except Exception as error:
                detail = f"{type(error).__name__}: {error}"
                if isinstance(error, ImportError):
                    detail = "依赖缺失，未验证；" + detail
                failed.append({"check": name, "error": detail})
                print(f"[失败] {name}: {detail}", file=sys.stderr, flush=True)
                traceback.print_exc()
            else:
                passed.append(name)
                print(f"[通过] {name}", flush=True)
        for item in captured:
            warning = f"{item.category.__name__}: {item.message}"
            reported_warnings.append({"check": name, "warning": warning})
            print(f"[警告] {warning}", flush=True)
    not_executed = ["human 桌面窗口：本脚本不自动打开；请单独运行 demo.py --mode project --render human"]
    print("验收汇总：" + json.dumps({"passed": passed, "failed": failed,
        "not_executed": not_executed, "warnings": reported_warnings}, ensure_ascii=False), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
