"""阶段 2：只计算九个原始 Q 值的 PyTorch 全连接网络。

此模块不创建环境或模型实例，不执行优化、探索、动作掩码或驾驶训练。
环境模块不导入本模块，因此环境演示仍可独立于 PyTorch 运行。
"""

from __future__ import annotations

import torch
from torch import nn

import config


class DQNNetwork(nn.Module):
    """34 → 128 → 128 → 9；隐藏层 ReLU，输出层没有激活函数。

    输入必须是 torch.Tensor，形状为 [34] 或 [B,34]（B>=1）。
    模型默认位于 CPU，参数为 float32；输入须与参数的 dtype/device 一致。
    调用方可显式 model.to(...)，并在网络外转换输入；forward 不隐式转换。
    环境观测可用 torch.from_numpy(observation) 转为 float32 CPU 张量。
    输出按固定动作编号 0～8 排列，不应用结构掩码，保留完整计算图。
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_dim = config.NETWORK_INPUT_DIM
        self.output_dim = config.NETWORK_OUTPUT_DIM
        first_hidden, second_hidden = config.HIDDEN_LAYER_SIZES
        # 显式使用 float32 CPU；不受调用方全局默认 dtype/device 的意外影响。
        options = {"bias": True, "dtype": torch.float32, "device": "cpu"}
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, first_hidden, **options),
            nn.ReLU(),
            nn.Linear(first_hidden, second_hidden, **options),
            nn.ReLU(),
            nn.Linear(second_hidden, self.output_dim, **options),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """返回 [9] 或 [B,9] 的原始 Q 值；不修改输入或断开梯度。"""
        if not isinstance(observation, torch.Tensor):
            raise TypeError("observation 必须是 torch.Tensor；请在网络外显式转换 NumPy 数组")
        if observation.ndim not in (1, 2) or observation.shape[-1] != self.input_dim:
            raise ValueError(
                f"observation 形状必须是 [{self.input_dim}] 或 [B,{self.input_dim}]，"
                f"实际为 {tuple(observation.shape)}"
            )
        if observation.ndim == 2 and observation.shape[0] == 0:
            raise ValueError("observation 批量大小 B 必须大于 0")
        parameter = self.layers[0].weight
        if observation.dtype != parameter.dtype:
            raise TypeError(
                f"observation dtype 必须与模型一致：{parameter.dtype}；实际为 {observation.dtype}"
            )
        if observation.device != parameter.device:
            raise ValueError(
                f"observation device 必须与模型一致：{parameter.device}；实际为 {observation.device}"
            )
        # Linear/ReLU 均非原地操作；不转 NumPy、不 detach、不移动设备。
        return self.layers(observation)
