"""独立固定场景的贪心验证；不收集训练经验、不改变训练随机流。

仅使用正式 HighwayLaneEnv 的 34 维观测、九动作和项目奖励。
不运行预留 final-test；validation 分数只用于预先约定的 checkpoint 选择。
"""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import torch

import config
from environment import make_env
from metrics import EpisodeMetrics, aggregate_metrics
from scenarios import ScenarioSampler, reset_scenario


def validate_agent(agent, episodes: int | None = None, seed: int | None = None) -> dict:
    """每次从相同 validation 种子开始；返回全部物理指标及逐回合记录。

    验证场景/环境随机源与训练实例无共享。epsilon=0 的动作选择不得消费
    agent.rng；torch fork_rng 还保护可能由环境构建等触发的全局 Torch 状态。
    前向使用 no_grad，恢复原 online/target train 模式，不改变优化器和回放。
    """
    episodes = config.TRAINING_CONFIG["validation_episodes"] if episodes is None else episodes
    seed = config.TRAINING_CONFIG["validation_seed"] if seed is None else seed
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes 必须为正整数")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed 必须为非负整数")
    online_mode, target_mode = agent.online.training, agent.target.training
    device = next(agent.online.parameters()).device
    devices = [] if device.type == "cpu" else [device.index or 0]
    records = []
    q_values = []
    env = None
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            agent.online.eval()
            agent.target.eval()
            env = make_env(render_mode=None, invalid_action_mode="raise")
            sampler = ScenarioSampler(split="validation", seed=seed)
            maximum_steps = math.ceil(config.EPISODE_DURATION_SECONDS * config.POLICY_FREQUENCY_HZ) + 1
            for episode in range(episodes):
                scenario = sampler.next()
                observation, info = reset_scenario(env, scenario)
                initial_scenario = deepcopy(info["scenario"])
                metrics = EpisodeMetrics(env)
                for _ in range(maximum_steps):
                    mask = info["action_mask"]
                    action = agent.select_action(observation, mask, epsilon=0.0)
                    tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
                    values = agent.online(tensor).detach().cpu().numpy()
                    if not np.isfinite(values).all():
                        raise ValueError("验证网络输出非有限 Q 值")
                    q_values.append(float(np.max(values[mask])))
                    observation, reward, terminated, truncated, info = env.step(action)
                    if info["invalid_action"] or info["executed_action"] != action:
                        raise RuntimeError("验证动作与实际执行不一致，停止运行，不接受 hold 回退")
                    metrics.update(env, reward, terminated, truncated, info)
                    if terminated or truncated:
                        break
                else:
                    raise RuntimeError("正式环境超过固定回合时限仍未终止/截断，验证停止")
                records.append({**metrics.finish(), "episode": episode, "scenario": initial_scenario})
    finally:
        if env is not None:
            env.close()
        agent.online.train(online_mode)
        agent.target.train(target_mode)
    return {
        **aggregate_metrics(records),
        "mean_legal_max_q": float(np.mean(q_values)),
        "max_abs_legal_max_q": float(np.max(np.abs(q_values))),
        "episode_records": records,
        "validation_seed": seed,
    }
