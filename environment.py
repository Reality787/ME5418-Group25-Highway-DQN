"""正式环境入口：直接创建队友 M1，不覆盖其观测、奖励、控制或默认配置。"""
from highway_m1 import ACTION_NAMES, OBSERVATION_NAMES, SPEED_TARGETS, HighwayDecisionEnv

__all__ = ['HighwayDecisionEnv', 'make_env', 'ACTION_NAMES', 'OBSERVATION_NAMES', 'SPEED_TARGETS']


def make_env(config=None, render_mode=None):
    """原样转交 M1 构造参数；reset/step/action_mask 使用 M1 原接口。"""
    return HighwayDecisionEnv(config=config, render_mode=render_mode)
