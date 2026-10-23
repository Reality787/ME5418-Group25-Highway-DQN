"""自行实现的均匀经验回放：环形存储、独立随机数流、终止/截断分开保存。"""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral, Real

import numpy as np

import config


class ReplayBuffer:
    """保存真实 step 的最终观测，不在此重置环境或计算学习目标。"""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        if isinstance(capacity, (bool, np.bool_)) or not isinstance(capacity, Integral) or capacity <= 0:
            raise ValueError("capacity 必须是正整数")
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self._size = 0
        self._position = 0
        self._data = {
            "obs": np.empty((self.capacity, config.OBSERVATION_DIM), dtype=np.float32),
            "action": np.empty(self.capacity, dtype=np.int64),
            "reward": np.empty(self.capacity, dtype=np.float32),
            "next_obs": np.empty((self.capacity, config.OBSERVATION_DIM), dtype=np.float32),
            "terminated": np.empty(self.capacity, dtype=np.bool_),
            "truncated": np.empty(self.capacity, dtype=np.bool_),
            "next_action_mask": np.empty((self.capacity, config.ACTION_COUNT), dtype=np.bool_),
        }

    def add(self, obs, action, reward, next_obs, terminated, truncated, next_action_mask) -> None:
        """验证整条经验后写入独立副本；拒绝输入不会改变缓存或游标。"""
        values = {}
        for key, observation in (("obs", obs), ("next_obs", next_obs)):
            array = np.asarray(observation, dtype=np.float32)
            if array.shape != (config.OBSERVATION_DIM,) or not np.isfinite(array).all():
                raise ValueError(f"{key} 必须是有限的 ({config.OBSERVATION_DIM},) 观测")
            values[key] = array
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, Integral):
            raise TypeError("action 必须是整数")
        if not 0 <= action < config.ACTION_COUNT:
            raise ValueError("action 超出九动作范围")
        if isinstance(reward, (bool, np.bool_)) or not isinstance(reward, Real) or not np.isfinite(reward):
            raise ValueError("reward 必须是有限实数")
        if abs(reward) > np.finfo(np.float32).max:
            raise ValueError("reward 超出 float32 范围")
        for key, value in (("terminated", terminated), ("truncated", truncated)):
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{key} 必须是布尔值")
            values[key] = bool(value)
        mask = np.asarray(next_action_mask)
        if mask.shape != (config.ACTION_COUNT,) or mask.dtype != np.bool_:
            raise ValueError("next_action_mask 必须是 shape=(9,) 的布尔数组")
        if not terminated and not mask.any():
            raise ValueError("需要 bootstrap 的下一状态不能具有空动作掩码")
        values.update(action=int(action), reward=float(reward), next_action_mask=mask)
        for key, value in values.items():
            self._data[key][self._position] = value
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """均匀不放回采样；返回独立数组，学习代码无法改写历史经验。"""
        if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, Integral) or batch_size <= 0:
            raise ValueError("batch_size 必须是正整数")
        if batch_size > self._size:
            raise ValueError(f"经验不足：需要 {batch_size} 条，当前仅 {self._size} 条")
        indices = self.rng.choice(self._size, size=int(batch_size), replace=False)
        return {key: value[indices].copy() for key, value in self._data.items()}

    def __len__(self) -> int:
        return self._size

    def state_dict(self) -> dict:
        """仅序列化已写入部分；Tensor/基础类型兼容 weights_only=True。"""
        import torch

        return {
            "version": 1, "capacity": self.capacity, "size": self._size,
            "position": self._position, "rng_state": deepcopy(self.rng.bit_generator.state),
            "data": {key: torch.from_numpy(value[:self._size].copy()) for key, value in self._data.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        """恢复环形位置和下次采样随机状态；容量/形状不兼容时拒绝。"""
        import torch

        if state.get("version") != 1 or state.get("capacity") != self.capacity:
            raise ValueError("ReplayBuffer 检查点版本或容量不兼容")
        size, position = state["size"], state["position"]
        if not isinstance(size, int) or not 0 <= size <= self.capacity:
            raise ValueError("ReplayBuffer size 非法")
        if not isinstance(position, int) or not 0 <= position < self.capacity:
            raise ValueError("ReplayBuffer position 非法")
        if size < self.capacity and position != size:
            raise ValueError("ReplayBuffer 尚未填满时游标应等于 size")
        if set(state["data"]) != set(self._data):
            raise ValueError("ReplayBuffer 字段不匹配")
        restored = {}
        for key, template in self._data.items():
            tensor = state["data"][key]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("ReplayBuffer 数据应为 Tensor")
            array = tensor.detach().cpu().numpy()
            if array.shape != (size,) + template.shape[1:] or array.dtype != template.dtype:
                raise ValueError(f"ReplayBuffer {key} 形状或 dtype 不匹配")
            if array.dtype.kind == "f" and not np.isfinite(array).all():
                raise ValueError(f"ReplayBuffer {key} 含非有限值")
            restored[key] = array.copy()
        if np.any((restored["action"] < 0) | (restored["action"] >= config.ACTION_COUNT)):
            raise ValueError("ReplayBuffer action 越界")
        if np.any(~restored["terminated"] & ~restored["next_action_mask"].any(axis=1)):
            raise ValueError("ReplayBuffer 非终止样本存在空掩码")
        restored_rng = np.random.default_rng()
        restored_rng.bit_generator.state = deepcopy(state["rng_state"])
        for key, value in restored.items():
            self._data[key][:size] = value
        self._size, self._position, self.rng = size, position, restored_rng
