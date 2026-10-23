"""M1 原有离屏渲染修正抽成小函数；供原环境和正式环境的展示入口共用。"""
from highway_env.envs.common.graphics import EnvViewer


def enable_rgb_rendering(env):
    """保留队友做法：dummy 驱动下启用 Surface 绘制，不创建操作系统窗口。"""
    if env.render_mode == "rgb_array":
        if env.viewer is None:
            env.viewer = EnvViewer(env)
        if env.viewer.offscreen:
            env.viewer.enabled = True
