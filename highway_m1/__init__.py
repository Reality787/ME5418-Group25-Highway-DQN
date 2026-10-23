"""M1 公共接口：保留队友名称，环境类延迟导入，不影响纯网络使用。"""
from .schema import ACTION_NAMES, OBSERVATION_NAMES, SPEED_TARGETS

__all__ = ["HighwayDecisionEnv", "ACTION_NAMES", "OBSERVATION_NAMES", "SPEED_TARGETS"]


def __getattr__(name):
    if name == "HighwayDecisionEnv":
        from .environment import HighwayDecisionEnv
        return HighwayDecisionEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
