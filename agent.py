"""普通 DQN 学习核心；不创建环境、不自行推进回合、不启动训练。"""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral, Real

import numpy as np
import torch
from torch import nn

import config
from network import DQNNetwork
from replay_buffer import ReplayBuffer


class DQNAgent:
    """掩码用于探索、贪心及 TD 目标；环境步调度由 train.py 管理。"""

    def __init__(self, seed: int = 0, device: str = "cpu", settings: dict | None = None) -> None:
        self.settings = dict(getattr(config, "DQN_CONFIG", {}))
        if settings is not None:
            self.settings.update(settings)
        required = {
            "learning_rate", "batch_size", "replay_capacity", "learning_starts", "train_every",
            "target_update_every", "gamma", "gradient_norm_clip", "epsilon_start", "epsilon_end",
            "epsilon_decay_fraction",
        }
        if not required <= self.settings.keys():
            raise ValueError(f"缺少 DQN 配置：{sorted(required - self.settings.keys())}")
        for key in ("batch_size", "replay_capacity", "train_every", "target_update_every"):
            value = self.settings[key]
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{key} 必须是正整数")
        if self.settings["batch_size"] > self.settings["replay_capacity"]:
            raise ValueError("batch_size 不能大于 replay_capacity")
        for key in ("learning_rate", "gradient_norm_clip"):
            if not np.isfinite(self.settings[key]) or self.settings[key] <= 0:
                raise ValueError(f"{key} 必须是有限正数")
        if not 0 <= self.settings["gamma"] <= 1:
            raise ValueError("gamma 必须位于 [0,1]")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed 必须是非负整数")
        self.device = torch.device(device)
        # CPU 初始化暂存/恢复外部 torch RNG；各 run 的网络初始化完全独立。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.online = DQNNetwork().to(self.device)
            self.target = DQNNetwork().to(self.device)
        self.target.requires_grad_(False)
        self.target.eval()
        self.sync_target_network()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.settings["learning_rate"])
        self.loss_function = nn.SmoothL1Loss()
        self.rng = np.random.default_rng(seed)
        replay_seed = np.random.SeedSequence([int(seed), 1])
        self.replay = ReplayBuffer(self.settings["replay_capacity"], seed=replay_seed)
        self.gradient_steps = 0

    def select_action(self, observation, action_mask, epsilon: float = 0.0) -> int:
        """只在合法动作中探索/取 argmax；epsilon=0 不消费探索随机数。"""
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (config.OBSERVATION_DIM,) or not np.isfinite(observation).all():
            raise ValueError("observation 必须是有限的 shape=(34,) 数组")
        mask = np.asarray(action_mask)
        if mask.shape != (config.ACTION_COUNT,) or mask.dtype != np.bool_:
            raise ValueError("action_mask 必须是 shape=(9,) 的布尔数组")
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            raise ValueError("非终止状态的动作掩码不能为空")
        if isinstance(epsilon, (bool, np.bool_)) or not isinstance(epsilon, Real) or not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon 必须是 [0,1] 中的有限实数")
        if epsilon > 0 and self.rng.random() < epsilon:
            return int(self.rng.choice(valid))
        with torch.no_grad():
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            q_values = self.online(observation_tensor)
            if not torch.isfinite(q_values).all():
                raise FloatingPointError("在线网络输出非有限 Q 值")
            legal = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            return int(q_values.masked_fill(~legal, -torch.inf).argmax().item())

    def _batch_tensors(self, batch: dict) -> dict[str, torch.Tensor]:
        """统一检查批量形状，尤其拒绝 reward/action 的 [B,1] 广播。"""
        keys = {"obs", "action", "reward", "next_obs", "terminated", "truncated", "next_action_mask"}
        if set(batch) != keys:
            raise ValueError("batch 必须包含且仅包含完整的七个交互字段")
        result = {key: torch.as_tensor(value, device=self.device) for key, value in batch.items()}
        if result["obs"].ndim != 2 or result["obs"].shape[1] != config.OBSERVATION_DIM:
            raise ValueError("obs 批量形状应为 [B,34]")
        count = result["obs"].shape[0]
        if count == 0:
            raise ValueError("批量不能为空")
        for key, value in result.items():
            expected = (count, config.OBSERVATION_DIM) if key in ("obs", "next_obs") else (
                (count, config.ACTION_COUNT) if key == "next_action_mask" else (count,)
            )
            if tuple(value.shape) != expected:
                raise ValueError(f"{key} 形状应为 {expected}，实际为 {tuple(value.shape)}")
            if key in ("terminated", "truncated", "next_action_mask"):
                if value.dtype != torch.bool:
                    raise TypeError(f"{key} 必须是布尔值")
            elif key == "action":
                if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                    raise TypeError("action 必须是整数")
                if torch.any((value < 0) | (value >= config.ACTION_COUNT)):
                    raise ValueError("action 超出九动作范围")
                result[key] = value.to(dtype=torch.int64)
            else:
                result[key] = value.to(dtype=torch.float32)
                if not torch.isfinite(result[key]).all():
                    raise ValueError(f"{key} 含非有限值")
        return result

    def _td_targets(self, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        """仅非 terminated 样本访问目标网络；truncated 仍使用截断前 next_obs。"""
        with torch.no_grad():
            targets = tensors["reward"].clone()
            bootstrap = ~tensors["terminated"]
            if bootstrap.any():
                mask = tensors["next_action_mask"][bootstrap]
                if not mask.any(dim=1).all():
                    raise ValueError("需要 bootstrap 的下一状态具有空掩码")
                q_next = self.target(tensors["next_obs"][bootstrap])
                if not torch.isfinite(q_next).all():
                    raise FloatingPointError("目标网络输出非有限 Q 值")
                best_next = q_next.masked_fill(~mask, -torch.inf).max(dim=1).values
                targets[bootstrap] += self.settings["gamma"] * best_next
            if not torch.isfinite(targets).all():
                raise FloatingPointError("TD 目标含非有限值")
            return targets

    def td_targets(self, batch: dict) -> torch.Tensor:
        """公开纯目标计算，方便独立手算验证，不更新任何网络参数。"""
        return self._td_targets(self._batch_tensors(batch))

    def learn(self, batch: dict | None = None) -> dict | None:
        """恰好执行一次优化；不足一个 batch 时返回 None，不作非法采样。"""
        if batch is None:
            if len(self.replay) < self.settings["batch_size"]:
                return None
            batch = self.replay.sample(self.settings["batch_size"])
        tensors = self._batch_tensors(batch)
        targets = self._td_targets(tensors)
        predictions = self.online(tensors["obs"]).gather(1, tensors["action"].unsqueeze(1)).squeeze(1)
        if predictions.shape != targets.shape:
            raise ValueError("当前 Q 值与 TD 目标必须同为 [B]，禁止隐式广播")
        loss = self.loss_function(predictions, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("DQN 损失非有限")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.online.parameters(), self.settings["gradient_norm_clip"], error_if_nonfinite=True
        )
        self.optimizer.step()
        self.gradient_steps += 1
        synced = self.gradient_steps % self.settings["target_update_every"] == 0
        if synced:
            self.sync_target_network()
        return {
            "loss": float(loss.detach().item()), "q_mean": float(predictions.detach().mean().item()),
            "target_mean": float(targets.mean().item()), "grad_norm": float(gradient_norm.item()),
            "gradient_steps": self.gradient_steps, "target_synced": synced,
        }

    def sync_target_network(self) -> None:
        """硬复制，不使用 Double DQN；target 始终不参与梯度更新。"""
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.target.eval()

    def state_dict(self) -> dict:
        """供上层回合边界 checkpoint 使用；不包含任何仿真状态。"""
        return {
            "version": 1, "settings": deepcopy(self.settings),
            "online": deepcopy(self.online.state_dict()), "target": deepcopy(self.target.state_dict()),
            "optimizer": deepcopy(self.optimizer.state_dict()), "replay": self.replay.state_dict(),
            "rng_state": deepcopy(self.rng.bit_generator.state), "torch_rng_state": torch.get_rng_state().clone(),
            "gradient_steps": self.gradient_steps,
        }

    def load_state_dict(self, state: dict) -> None:
        """恢复同配置学习状态；上层还须核对环境签名、游标和 seed。"""
        if state.get("version") != 1 or state.get("settings") != self.settings:
            raise ValueError("DQN 检查点版本或超参数不兼容，请建立新 run")
        self.replay.load_state_dict(state["replay"])
        self.online.load_state_dict(state["online"], strict=True)
        self.target.load_state_dict(state["target"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.rng.bit_generator.state = deepcopy(state["rng_state"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        self.gradient_steps = int(state["gradient_steps"])
        self.target.requires_grad_(False)
        self.target.eval()
