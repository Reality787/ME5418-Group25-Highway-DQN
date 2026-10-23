"""无视频依赖的 M1 有界演示；复用原始环境、演示策略和状态叠图。"""
import argparse
import json
import math
from pathlib import Path


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('必须是正整数')
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=25)
    parser.add_argument('--episodes', type=positive_int, default=1)
    parser.add_argument('--policy', choices=('hold', 'random', 'scripted'), default='scripted')
    parser.add_argument('--render', choices=('none', 'human', 'rgb_array'), default='none')
    parser.add_argument('--output', type=Path, help='可选：单回合 rgb_array 的新 PNG 文件')
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error('seed 必须非负')
    if args.output and (args.render != 'rgb_array' or args.episodes != 1
                        or args.output.suffix.lower() != '.png' or args.output.exists()):
        parser.error('output 仅支持单回合 rgb_array，且必须是尚不存在的 .png')
    from environment import make_env, SPEED_TARGETS
    from demo import DemonstrationPolicy, overlay
    from PIL import Image
    env = make_env(render_mode=None if args.render == 'none' else args.render)
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            observation, info = env.reset(seed=seed)
            policy = DemonstrationPolicy(args.policy, seed)
            total = 0.0
            # M1 的时限是有限的；保护入口不因接口异常形成无限循环。
            limit = math.ceil(env.config['duration'] / env.decision_dt) + 1
            for step in range(1, limit + 1):
                action = policy(info)
                observation, reward, terminated, truncated, info = env.step(action)
                if info['invalid_action'] or info['executed_action'] != action:
                    raise RuntimeError('合法演示策略触发了动作回退，请检查接口')
                total += reward
                if terminated or truncated:
                    break
            else:
                raise RuntimeError('超过 M1 配置时限仍未结束')
            if args.output:
                frame = overlay(env.render(), info, reward, args.policy, seed, SPEED_TARGETS)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(frame).save(args.output)
            print(json.dumps({'environment': 'M1 HighwayDecisionEnv', 'seed': seed,
                              'steps': step, 'simulation_time_s': info['simulation_time_s'],
                              'observation_shape': list(observation.shape), 'reward': total,
                              'terminated': terminated, 'truncated': truncated,
                              'note': '未训练策略的功能演示，不是模型评测'}, ensure_ascii=False))
    except KeyboardInterrupt:
        print('用户中断：不能记作完整回合验证。')
        return 130
    finally:
        env.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
