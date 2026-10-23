# 阶段 3：普通 DQN 设计与正确性约定

本文依据当前项目源码编写。阶段 2 的 `DQNNetwork` 已复用，学习核心由项目自行实现，
没有使用 Stable-Baselines3，也没有加入 Double DQN、Dueling、PER 或其他算法。
训练只连接正式 `HighwayLaneEnv`：34 维 float32 观测、九动作、结构掩码及项目奖励。
stock、joint-debug、observation-debug 的原生调试奖励不用于驾驶训练。

本轮复用项目解释器 Python 3.11.16、PyTorch 2.14.1+cpu、Gymnasium 1.4.0、
HighwayEnv 1.12.1、NumPy 2.4.6；没有升级依赖或安装 CUDA。

参数起点和选模规则来自本轮用户指定的工程方案，不声称由 proposal 给出或已达最优。
各次实际 smoke、pilot 和 formal 的完成状态以运行目录 `status.json`、日志及
[训练协议](training_protocol.md)、[进度记录](progress.md)为准；代码完成不等于正式训练完成。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `network.py` | 只计算 `34→128→128→9` 原始 Q 值，22,153 参数 |
| `agent.py` | online/target、合法动作选择、TD 目标、一次梯度更新与同步 |
| `replay_buffer.py` | 有界环形经验池、复制存储、独立随机采样和恢复 |
| `scenarios.py` | 按划分采样场景、真实初始化、初态检查及场景游标 |
| `metrics.py` | 与奖励分开的物理指标、固定分母和 checkpoint 排序 |
| `validation.py` | 独立固定场景、epsilon=0 验证，隔离训练状态 |
| `checkpoints.py` | 兼容签名、源码哈希、推理/恢复类型检查和原子写入 |
| `train.py` | 有界 CPU 单环境训练、计数、日志、分级运行与恢复 |
| `config.py` | DQN、场景、验证、指标和运行频率集中配置 |

导入这些入口不会启动训练。正式环境本身仍不依赖学习模块或 PyTorch。

## 网络与普通 DQN 目标

online 参数记为 θ，独立 target 参数记为 θ⁻。二者结构相同：

```text
Linear(34,128,bias=True) → ReLU
→ Linear(128,128,bias=True) → ReLU
→ Linear(128,9,bias=True)
```

输出没有 Softmax/Sigmoid。初始化后立即将 online 参数复制到 target；
target 使用 `requires_grad_(False)`，且不属于 Adam 优化器的参数集合。

经验为 `(obs, action, reward, next_obs, terminated, truncated, next_action_mask)`。
令下一状态合法动作集合为 `A(next_obs)`，采用普通 DQN：

```text
terminated=True:
    y = reward

terminated=False（包括 truncated=True）:
    y = reward + 0.99 * max_{a ∈ A(next_obs)} Q_target(next_obs, a)
```

target 自己完成最大值选择和估值；没有改为 Double DQN 的 online 选动作、target 估值。
整个目标计算位于 `torch.no_grad()` 下。

真正碰撞/离路属于 terminated，因此不 bootstrap。无失败到 40 秒属于 truncated，
应使用**截断之前最后一次 step 返回的 next_obs 和 next_action_mask**继续 bootstrap。
训练循环先存该次经验，再在回合结束后 reset；reset 的观测不能替换最后状态。
这里不使用 `terminated OR truncated` 作为 bootstrap 屏蔽条件。

实现只对 `~terminated` 子批次调用 target 和求掩码最大值。全 terminal 批次直接返回奖励，
即使其下一动作掩码为空也不构造 `0 * (-inf)`。需要 bootstrap 的样本如果掩码为空，明确报错。
非法动作 Q 值填为 `-inf` 后再取最大值；对网络输出、目标、损失和梯度检查有限性。

当前动作值使用：

```python
predictions = online(obs).gather(1, action.unsqueeze(1)).squeeze(1)
```

`predictions` 和 `targets` 均须为 `[B]`。经验批中 action/reward/终止标记不能用 `[B,1]`
混入计算，否则明确拒绝，避免错误广播产生 `[B,B]` 损失。

## 超参数与三类步数

| 配置项 | 固定工程起点 |
| --- | --- |
| optimizer / learning_rate | Adam / `1e-4` |
| batch_size / replay_capacity | 64 / 100000 条转移 |
| learning_starts | 已收集 1000 个环境决策步后 |
| train_every / updates_per_trigger | 每 4 个环境步触发，一次触发执行 1 次梯度更新 |
| loss | `SmoothL1Loss` |
| gradient_norm_clip | 10，限制网络梯度范数，不改变车辆加速度 |
| gamma | 0.99 |
| target_update_every | 每 1000 次成功梯度更新后硬复制 |
| epsilon | 1.0 线性降至 0.05，在预算前 60% 完成下降 |
| execution | CPU、一个 Torch 线程、单环境、无渲染 |

定义 `t` 为当前动作执行前已完成的环境步、`B` 为本 run 固定计划预算：

```text
epsilon(t) = 1.0 + (0.05 - 1.0) * min(1, max(0, t/(0.60*B)))
```

该分母不会因为恢复或已提前停止的步数而改变。恢复时更改预算会被拒绝。

三种计数分别记录：

- **环境步**：一次联合动作及一次真实 `env.step()`；通常 0.2 秒。
- **物理帧**：HighwayEnv 15 Hz 更新；通常每个环境步对应 3 帧。
- **梯度更新**：一次 Adam `optimizer.step()`；仅满足学习起点、频率和批大小时发生。

当前实现先存第 1000 条经验，再在环境步 1000 执行第一次更新，此后为 1004、1008……。
因此 target 的首次第 1000 次更新出现在环境步 4996，不能把同步周期误解为每 1000 环境步。
每次成功更新返回 loss、当前 Q 均值、目标均值、裁剪前梯度范数及 target 是否同步；
updates.jsonl按配置每100次梯度更新记录一条，曲线明确标为这些实际采样日志。

## 掩码与经验回放

探索只从 `np.flatnonzero(mask)` 均匀采样。利用只在合法动作中取 online Q 最大值；
非终止状态空掩码直接报错。epsilon=0 不消费探索随机数，因此验证不会改变后续训练探索序列。

训练和验证均创建 `invalid_action_mode='raise'` 的正式环境，并在 step 后再次核对
`info['executed_action'] == selected_action` 且 `info['invalid_action'] == False`。
若执行不一致或发生 hold 回退，立即停止该 run，不把所选动作和另一实际动作混成错误经验。

ReplayBuffer 为容量固定的环形数组。写入每条观测和掩码时复制数据；采样返回独立数组，
环境复用输出数组或训练侧修改 batch 均不会改写历史经验。
采样使用独立、可设种子的 NumPy Generator，均匀不放回抽样。
经验不足一个 batch 时 `agent.learn()` 返回 None；训练循环在足量后才调用。
直接请求超出经验数量的 sample 会明确报错。

online 初始化、探索 RNG、ReplayBuffer RNG 和场景 RNG 分离。
每个正式种子从新网络、新 target、新优化器、新经验池和独立 RNG 开始，不能沿用 pilot 权重。

## 独立验证与统一物理指标

固定验证为 validation 命名空间、种子 0 的前 6 个场景。
每次重新构造独立场景采样器，保证同一协议下各 checkpoint 面对同一组初始条件。
训练开始前验证一次，之后每 5000 环境步及预算结束时验证；因此选模规则也可能选择 step 0，
不得隐藏这种没有改善的结果。

验证使用单独正式环境、epsilon=0、`torch.no_grad()` 和临时 eval 模式。
结束后恢复 online/target 原训练模式；不调用学习更新、不写回放、不修改训练场景游标。
`torch.random.fork_rng` 保护 Torch 全局随机状态。测试同时检查 NumPy 全局随机状态、
agent 完整状态、训练环境和下一训练场景均保持不变。
验证有最多 201 次 step 的有限防护；若超过正式 40 秒回合仍未结束，会报错而不是记作成功。

物理指标由 `EpisodeMetrics` 从真实车辆位置差沿固定道路方向累计，不能从奖励反推：

```text
failure = collision OR off_road
normalized_progress = longitudinal_distance / 1112
completion = 无 failure 且无 terminated，因时间上限 truncated 且已到 40 秒
```

1112 米固定为 `27.80 m/s × 40 s`。失败回合照常保留，不按较短回合时间修改分母；
进度不裁剪，负位移和超过 1112 米均如实记录。预算在回合中结束时标记 partial，不能记为 completion。

此外记录：累计项目奖励、环境步、物理帧、时长、停车占比、换道次数和调速次数。
停车为步后纵向速度 `<1m/s` 的物理时长比例，该阈值是工程诊断约定。
目标变化次数读取 `joint_debug.before/targets_applied`，重复 hold 不额外计数。
验证还记录合法最大 Q 的均值及绝对值幅度，辅助发现数值漂移，不能解释为驾驶性能。

选模规则预先固定：

```text
validation_score = mean_normalized_progress - 2 * failure_rate
```

score 高者优先；精确同分时选 collision_rate 较低者，仍同分选更早环境步的 checkpoint。
同时保存碰撞率、离路率、失败率、完成率、距离、进度、奖励、停车比例及目标变化次数，
不只保存 score。该规则是工程选模规则，不构成安全保证。

## 检查点、兼容与恢复边界

检查点仅加载本项目可信输出。使用 `torch.load(..., map_location='cpu', weights_only=True)`，
不加载来源不明的序列化对象。保存先写同目录临时文件、flush/fsync，再原子替换目标文件。

两种类型明确区分，不能混用：

| 类型 | 典型文件 | 内容和用途 |
| --- | --- | --- |
| `dqn_inference` | `best.pt`、`last.pt` | online 权重、配置、兼容签名、环境/梯度计数及已有验证记录；用于推理 |
| `dqn_resume_boundary` | `resume.pt` | online/target、Adam 状态、完整回放和其 RNG、探索 RNG、Torch CPU RNG、更新次数、场景游标/状态、训练计数、固定预算、下一验证步和最佳记录 |

`best.pt` 只按固定验证标准更新。`last.pt` 在验证后保存当前推理参数；若被中断或报错，
也尽量保存当前参数，但它不含完整仿真状态，不能宣称精确续接当前回合。

`resume.pt` 只在完整回合边界保存，包括初始化边界，以及达到每 5000 环境步保存间隔后的
首个结束边界。预算恰在回合边界结束时再保存一次；预算在回合中结束时，最后可恢复边界
可能早于最后推理权重，`status.json` 的 `last_resume_env_steps` 明确记录这一差别。
恢复后从该边界的下一场景继续。旧日志中边界之后的尾段不删除，使用 `segment` 和恢复事件
区分，避免把重放的尾段误计成连续完成的额外步数。

兼容记录包括：全部集中配置、34 维特征顺序和归一化、九动作映射、项目奖励、场景定义、
实际底层控制参数、项目运行源码 SHA256、关键 HighwayEnv 库源码 SHA256、Python 和依赖版本。
加载必须同时匹配签名和完整记录。run 恢复还要求 mode、seed、计划预算和超参数完全相同；
不兼容就拒绝，使用对应历史源码环境或新 run，不能静默给旧权重换环境。
正式模式另要求冻结协议的状态、兼容记录、预算和种子列表一致。

## 已执行的恢复检查与测试范围

已执行真实 **400 环境步、97 次梯度更新**的回合边界恢复测试：一份连续运行，另一份在
达到指定步数后的完整回合边界停止并恢复。两者使用相同预算，保证 epsilon 分母不变。
最终 online 推理权重逐位一致；最近共同完整边界的全部学习状态（online、target、optimizer、
ReplayBuffer 及 RNG）和场景采样器状态也一致，验证 score/failure_rate 和场景顺序一致。
该小测试使用较小经验池、较早学习起点及单回合验证，以验证机制；不冒充 smoke、pilot 或正式结果。

当前测试目录保留阶段 1/2 的 114 项，并新增 22 项学习核心、15 项场景、
11 项指标/独立验证、5 项训练与检查点测试，共 167 项。
最终全套已实际执行：**167通过、0失败、0跳过，41.276秒**。
执行命令为指定解释器的 `python -m unittest discover -s tests -v`。
分级训练的完成情况另见 [进度记录](progress.md)，不能由测试通过推断训练完成。

本阶段不运行 final-test，不使用预留最终测试结果决定随机化范围、预算或 checkpoint。
底层较强减速响应仍是已记录的环境限制，不能把网络梯度裁剪误当作物理加速度限幅。
是否改善、是否长期停车、是否收敛及三个正式种子的实际完成步数，都必须根据真实日志报告。
