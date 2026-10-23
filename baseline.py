"""Gap-based controller 基线接口骨架，不实现驾驶策略或规则阈值。"""

from __future__ import annotations

from typing import Sequence


class GapBasedController:
    """未来与 DQN 共享观测、九动作映射、执行器限制和初始场景。

    交通间隙可供基线策略决策，但不能据此修改共享的结构动作掩码。
    规则、阈值及相同初始场景的复现方式仍为 TODO。
    """

    def select_action(
        self,
        observation: Sequence[float],
        action_mask: Sequence[bool],
    ) -> int:
        """未来依据共享观测和结构掩码选取有效动作，返回编号 0 至 8。"""
        raise NotImplementedError("当前仅完成项目骨架，功能待实现：基线动作选择")
