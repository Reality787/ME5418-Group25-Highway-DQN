# ME5418 · Highway Lane Change and Speed Decisions

基于 HighwayEnv、Gymnasium 和 PyTorch 的课程项目：在三车道高速公路上，用普通 DQN 同时选择换道方向与目标速度。项目围绕 **Environment / Neural Network / Learning Agent** 三个模块组织。

当前代码已包含真实环境、34 维观测、九个联合动作、项目奖励和自行实现的 DQN。队友代码整合后实际全套测试为 **187 通过、0 失败、0 跳过**；三个正式种子的训练尚未全部完成。M2 网络已接入 DQN，M1 的演示策略、叠图和离屏渲染已复用；正式环境和冻结协议保持原样。详情见 [整合与兼容记录](docs/teammate_code_integration.md)、[当前进度](docs/progress.md)、[项目规格](docs/project_spec.md)和[训练协议](docs/training_protocol.md)。本轮只执行独立短程 smoke，没有继续正式训练。

## 1. Project Overview

任务是研究在相同观测、动作和执行器约束下，DQN 如何进行车道与速度决策。HighwayEnv 提供道路、车辆动力学、周围车辆行为、底层跟踪控制、碰撞检测与渲染；本项目实现联合动作适配、结构掩码、感知与观测、奖励、Gym 接口、网络及学习流程。

没有使用 Stable-Baselines3 替代 DQN 核心，也没有将算法升级为 Double/Dueling/PER。`baseline.py` 和 `evaluate.py` 仍为接口骨架，gap-based 对照和最终测试尚未完成。

## 2. Environment

正式入口是 `environment.make_env()` / `HighwayLaneEnv`。它返回真实 `Box(shape=(34,), dtype=float32)` 观测与 `Discrete(9)` 动作空间。

| 项目 | 当前设置 |
| --- | --- |
| 道路 / 车辆 | 3 条同向直车道，1 辆受控车，初始 20 辆周围车辆 |
| 决策 / 物理仿真频率 | 5 Hz / 15 Hz；正常一次决策为 0.2 秒、3 个物理更新 |
| 回合上限 / 期望速度 | 40 秒 / 27.80 m/s |
| 感知范围 | 车辆中心平面欧氏距离 ≤100 米 |
| `terminated` | 碰撞或离路 |
| `truncated` | 无失败达到时间上限；同时发生失败时以 `terminated` 为准 |

`reset()` 返回 `(observation, info)`；`step(action)` 返回 `(observation, reward, terminated, truncated, info)`。`info["action_mask"]` 是返回状态对应的九维布尔掩码。环境本身可独立于 PyTorch 运行。

`make_stock_env()` 及 `stock`、`joint-debug`、`observation-debug` 演示保留用于分层诊断；只有 **project** 使用正式项目奖励。调试环境不用于正式驾驶训练。

## 3. Observation: 34 Features

特征按真实车道编号 0、1、2 固定排列，不随自车位置重排。

| 索引 | 内容 | 归一化与范围 |
| --- | --- | --- |
| 0–2 / 3–5 | 车道 0 前车 / 后车：presence、bumper gap、relative speed | 0/1；gap÷100，`[0,1]`；相对速度÷40，`[-1,1]` |
| 6–8 / 9–11 | 车道 1 前车 / 后车，同上 | 同上 |
| 12–14 / 15–17 | 车道 2 前车 / 后车，同上 | 同上 |
| 18 | 自车纵向速度 | ÷40 m/s，`[-1,1]` |
| 19 | 自车横向速度 | ÷10 m/s，`[-1,1]` |
| 20 | 相对当前车道中心的横向偏移 | ÷2 m，`[-1,1]` |
| 21 / 22 | 当前车道 / 目标车道 | 编号÷2，`[0,1]` |
| 23 | 目标速度 | ÷27.80 m/s，`[0,1]` |
| 24 | 换道进行标记 | 0/1，复用控制器判据 |
| 25–33 | 上一实际执行联合动作 0–8 的 one-hot | reset 时全部为 0 |

先做 ≤100 米欧氏距离过滤，再选各车道最近前/后车，排除自车。按当前位置最近车道中心线归属，平局取较低车道号；纵向位置相等归前车，同距按道路车辆列表顺序确定。压线车辆暂只归入一条车道。

保险杠间距是纵向中心距离减去双方车身的道路方向投影半长，重叠截为 0：`half_extent=(length*abs(cos(angle))+width*abs(sin(angle)))/2`。相对速度统一为 `other_longitudinal_speed - ego_longitudinal_speed`，后车槽不反转符号。缺失车辆编码为 `(0,1,0)`。

具体排列、归一化尺度、裁剪和缺失编码是工程约定，不声称来自 proposal 原文公式。超界裁剪记录在 `info["observation_debug"]["clipped_features"]`；NaN/Inf 报错。观测读取不推进仿真、不消费随机数，也不读取其他车辆隐藏目标或意图。

## 4. Action: 9 Joint Actions

| 编号 | 动作 | 编号 | 动作 | 编号 | 动作 |
| --- | --- | --- | --- | --- | --- |
| 0 | Left + slower | 3 | Keep + slower | 6 | Right + slower |
| 1 | Left + hold | 4 | Keep + hold | 7 | Right + hold |
| 2 | Left + faster | 5 | Keep + faster | 8 | Right + faster |

目标速度档位为 `[0.00, 6.95, 13.90, 20.85, 27.80]` m/s。slower/hold/faster 对已有目标索引加 `-1/0/+1`；Keep 保留目标车道，hold 保留目标速度，均不依实际速度重新设目标。

同次决策先更新两个目标，再推进一次底层 `IDLE`；九动作编号不会直接传入原生五动作接口。实际运动复用 HighwayEnv 动力学。

掩码只限制道路边缘向外换道、换道未完成时新换道、速度索引越界。换道过程中 Keep 类动作仍可调速；没有 THW/TTC 或交通风险过滤。暂定完成条件为当前车道等于目标车道、中心横向误差 ≤0.10 m、相对航向误差 ≤0.02 rad，三者同时满足。

正式环境默认允许将结构非法请求整体回退为动作 4，并记录请求/实际动作；训练和验证使用严格 `raise` 模式、合法采样并核对实际执行动作，任何回退或不一致都停止运行。

## 5. Reward

正式环境替换原生奖励，使用五个项目分量：

```text
r = dt * (1.0 * progress - 2.0 * short_headway)
    - 10.0 * failure
    - 0.05 * lane_change_started
    - 0.02 * target_speed_changed

progress = clip(longitudinal_displacement / (27.80 * dt), 0, 1)
THW = bumper_gap / max(ego_longitudinal_speed, 0.1)
short_headway = clip(1 - min_relevant_THW / 1.5, 0, 1)
```

无相关可见前车时短时距项为 0。正常行驶检查当前车道，换道中检查当前与目标车道并去重。碰撞/离路同时发生只扣一次失败成本；持续换道不重复扣发起成本。

权重、THW 阈值与低速处理均是集中于 `config.py` 的初始工程配置，不是已证明最优的 proposal 数值。`info["reward_terms"]` 提供五项贡献，其和等于返回奖励；原生奖励只作为诊断保存。详见[项目规格](docs/project_spec.md)。

## 6. DQN Network

```text
Linear(34, 128, bias=True) → ReLU
→ Linear(128, 128, bias=True) → ReLU
→ Linear(128, 9, bias=True)
```

参数量为 **22,153**：`34*128+128 + 128*128+128 + 128*9+9`。输入是与网络参数同 dtype/device 的 `torch.Tensor`，默认 CPU/float32；支持 `[34]→[9]` 与 `[B,34]→[B,9]`。

当前 `DQNNetwork` 通过薄继承接口调用队友 `highway_m2.QNetwork`；六个权重键、初始化顺序及数值行为与整合前等价，未在训练主流程保留第二套网络。

九个输出是按固定动作编号排列的原始 Q 值，无 Softmax/Sigmoid。网络只做前向计算，掩码、探索、环境交互和优化由 Learning Agent 负责。见[网络设计](docs/network_design.md)。

## 7. Learning Agent

自行实现 online/target 网络、均匀 ReplayBuffer、masked epsilon-greedy、TD 更新、Adam、目标同步和检查点。初始化时复制 online 到独立 target；target 无梯度。

```text
terminated:      y = reward
otherwise:       y = reward + 0.99 * max_legal Q_target(next_obs)
```

时间截断仍 bootstrap，使用当前 step 的最终观测与掩码；先存经验再 reset。只对需 bootstrap 的样本求下一状态最大值，避免全终止批次的 `0*(-inf)`。探索、贪心和 TD 均使用同一结构掩码。

工程超参数：Adam `1e-4`、batch 64、回放容量 100000、1000 环境步开始学习、每 4 环境步 1 次更新、SmoothL1Loss、梯度范数裁剪 10、每 1000 梯度更新硬同步 target；epsilon 在预算前 60% 从 1.0 降至 0.05。

场景真正随机化交通密度、初始间距和周车速度。开发/训练/验证/最终测试使用隔离种子空间，预留 withheld 组合及 OOD 变化轴。独立验证不更新网络、不写回放、不改变训练随机流。

选模规则为 `mean_normalized_progress - 2*failure_rate`，同分优先较低碰撞率、再较早 checkpoint；失败回合保留，进度始终除以 1112 米。每次固定 6 个验证场景，这一规则不是安全保证。

`best.pt` / `last.pt` 为推理权重；`resume.pt` 含回放、优化器、随机数和计数，只能从最近完整回合边界恢复。加载检查特征顺序、动作、奖励、动力学、源码哈希与依赖版本。详见[DQN 设计](docs/dqn_design.md)和[冻结训练协议](docs/training_protocol.md)。

## 8. Project Structure

| 文件/目录 | 作用 |
| --- | --- |
| `config.py` | 场景、观测、动作、奖励、网络与训练集中配置 |
| `environment.py` / `project_environment.py` | 环境公开入口 / 正式 Gym 生命周期和整合 |
| `joint_actions.py` / `observation.py` / `reward.py` | 联合控制与掩码 / 34 维感知 / 奖励纯函数 |
| `network.py` / `network_demo.py` | Q 网络 / 数值、保存加载和真实环境输入验证 |
| `agent.py` / `replay_buffer.py` | DQN 学习核心 / 复制存储的有界经验池 |
| `train.py` / `scenarios.py` | 有界训练与恢复 / 随机化和数据划分 |
| `validation.py` / `metrics.py` | 独立验证 / 物理指标和选模规则 |
| `checkpoints.py` / `plot_training.py` | 兼容检查及保存加载 / 真实日志绘图 |
| `demo.py` / `validate_env.py` | 随机动作演示 / Gym 与环境验收 |
| `baseline.py` / `evaluate.py` | 待实现的基线与最终评测接口 |
| `tests/` / `docs/` | 单元与真实仿真回归 / 设计及执行记录 |
| `environment.yml` / `requirements*.txt` | 新建独立环境 / 已验证依赖版本 |
| `outputs/` | 本地生成模型、日志、视频，不作为源代码上传 |

调用主线：`train → DQNAgent → HighwayLaneEnv → observation/joint control/reward → HighwayEnv`。`validation` 使用独立环境；网络模块不依赖环境创建。

## 9. Installation

在仓库根目录执行。已有正确环境时直接激活；没有环境时用提交的环境文件创建独立 Conda 环境：

```bash
conda env create -f environment.yml
conda activate me5418-highway
python -c "import sys, torch; print(sys.executable); print(sys.version); print(torch.__version__)"
```

若同名环境已存在，先核对它，不直接覆盖或升级。CPU 网络依赖来源和精确版本见 `requirements-network.txt`；基础环境依赖见 `requirements.txt`。

| 组件 | 现有项目环境中已验证版本 |
| --- | --- |
| Python | 3.11.16 |
| PyTorch | 2.14.1+cpu |
| Gymnasium / HighwayEnv | 1.4.0 / 1.12.1 |
| NumPy / pygame-ce | 2.4.6 / 2.5.8 |
| Matplotlib / pandas | 3.11.2 / 3.0.6 |

这些版本来自现有环境的实际检查。本次没有重新安装依赖；`environment.yml` 是复现配置，不代表已经在全新机器或全新 Conda 环境完成安装验证。默认 CPU、无渲染、单环境，不需要 CUDA。

## 10. How to Run

以下命令均在包含本 README 的仓库根目录、已激活项目环境后执行。

```bash
# Environment：真实34维、九动作和项目奖励；随机合法动作，非训练模型。
python demo.py --mode project --render none --episodes 1 --seed 0
python validate_env.py --episodes 3 --seed 0

# 有桌面显示时才运行图形演示。
python demo.py --mode project --render human --episodes 1 --seed 0

# Neural Network：合成数值检查、state_dict保存加载，以及真实reset观测输入。
python network_demo.py --seed 0 --steps 30

# Learning Agent：1004个真实环境步的短流程演示，使用全新空目录。
python train.py --mode smoke --device cpu --seed 101 --steps 1004 \
  --run-dir outputs/training/learning_agent_cr_demo
```

短 Learning Agent 演示在环境步 1000 和 1004 各执行一次优化，还包含初始/结束独立验证。它展示动作选择、回放、TD 更新与保存流程，**不代表模型收敛或有效驾驶策略**。重复执行时换一个空 `--run-dir`，程序会拒绝覆盖已有运行记录。

`network_demo.py` 的合成部分是“网络数值验证，不是驾驶训练结果”；默认保存到临时文件并重载比较。`demo.py --mode project` 不会自动加载训练权重。所有脚本导入时均不启动训练或仿真。

正式训练及边界恢复依赖本地冻结协议和可信 checkpoint；这些大型运行产物不包含在代码仓库/Canvas 源码包中。接收方无需已有模型即可运行上面的三个模块演示。本次整理没有续训 seed 33，也没有运行 final-test。

## 11. Tests

```bash
python -m unittest discover -s tests -v
python demo.py --help
python network_demo.py --help
python train.py --help
```

保留的原有 167 项测试涵盖配置、真实 stock/联合控制/34 维观测、奖励与正式环境，网络数值与保存加载，以及回放、掩码、手算 TD、终止/截断区别、场景随机化、独立验证、checkpoint 和边界恢复。

本轮整合前实测 167 项通过（43.025 秒），新增 20 项兼容/真实仿真测试后 **187 通过、0 失败、0 跳过**（37.775 秒）。真实恢复测试仍对比连续运行与回合边界恢复后的权重和学习状态；新增测试核对 M2 与旧网络输出/梯度/更新等价及 M1 诊断接口。独立 1004 步 smoke 实际执行 2 次优化并可重载。测试通过不等于训练收敛。队友原 pytest 测试因缺 pytest 未执行，不能并入通过数；详细边界见[整合记录](docs/teammate_code_integration.md)。以前的 167 项源码/ZIP 交付检查是历史记录，见[交付记录](docs/delivery_validation.md)。

## 12. Current Progress

| 项目 | 已记录状态 |
| --- | --- |
| 正式环境、Q 网络、普通 DQN 和训练/验证代码 | 已实现 |
| 整合后全套测试（保留原有 167 项） | 187 通过，0 失败，0 跳过 |
| smoke / pilot | 2,000 步 / 20,000 步完成 |
| formal seed 11 | 50,000 步、12,251 次更新完成 |
| formal seed 22 | 50,000 步、12,251 次更新完成 |
| formal seed 33 | 日志至少 19,396 步；完整恢复点 15,178 步，未完成 |
| 三模型统一冻结、gap-based 对照、最终测试 | 尚未完成 / 尚未完成 / 未运行 |

seed 33 的历史 `status.json` 最后写入 `running`，不能当作当前仍有训练进程的证明；日志尾部也不等于可恢复模型。已有正式预算统一为每种子 50,000 步，seed 11/22 的最佳 checkpoint 分别选自 15,000/10,000 步。结果有波动，seed 11 最终固定验证出现 2/6 碰撞。

[进度记录](docs/progress.md)保留真实训练与中断情况。历史文档引用的 `outputs/training/stage3/` 日志、曲线和权重属于本地实验产物，默认不随此代码仓库/Canvas 源码包提供；不要把历史产物链接理解为已上传文件。

## 13. Limitations

- 底层速度控制为 `a=(target_speed-speed)/0.6`，常规没有固定加速度限幅；目标 13.90、实际 25 m/s 时，0.2 秒后约 21.696 m/s。当前冻结模型保留这一较强响应，未擅自改物理参数。未来 DQN/基线必须共享相同控制限制。
- 结构动作掩码不做交通安全过滤；奖励和 THW 近似、单车道归属、换道容差及归一化仍需更广场景验证。
- 六个固定验证场景不足以证明泛化、收敛、真实驾驶安全或优于规则方法；存在反复调速等行为问题。
- seed 33 未完成，三个最终模型尚未统一冻结；未使用 withheld/OOD 最终测试决定范围或选择模型。
- 图形演示依赖桌面显示；新机器安装、课程最终报告与视频要求应单独核对。代码包不自动等于全部课程提交材料。

## 14. References and AI Assistance

运行接口与外部库来源：

- [HighwayEnv 官方文档](https://highway-env.farama.org/)：道路、交通行为、控制、动力学与渲染。
- [Gymnasium Env 接口](https://gymnasium.farama.org/api/env/)：reset/step、终止/截断与空间约定。
- [PyTorch 官方文档](https://docs.pytorch.org/docs/stable/index.html)：网络、自动微分与优化器。
- 课程最终九动作 proposal：`ME5418_Group25_Proposal(9-actions).pdf`；具体资料读取记录见[项目规格](docs/project_spec.md)。课程资料不作为开源依赖，也不擅自向仓库/提交包重复发布。
- Lecture 6b 的三个模块划分按用户提供的课程要求组织；本次整理未声称直接读取工作区之外的课件或其他 ChatGPT 附件。

代码、测试、排错和文档使用过 AI 辅助生成及审查。团队应阅读、理解并人工核对实现，按实际情况记录每位成员的贡献，并依据课程要求披露 AI 使用；仓库没有编造成员身份、贡献比例或课程合规结论。HighwayEnv/Gymnasium/PyTorch 的功能归第三方实现，不计为本项目自行编写的学习核心。协作约定见 [CONTRIBUTING.md](CONTRIBUTING.md)。

本次 Learning Agent C&R 的英文报告源稿见 [报告](docs/learning_agent_cr_report.md)，解压运行说明见 [提交包 README](docs/learning_agent_submission_README.md)，交付核验见 [交付记录](docs/delivery_validation.md)。PDF 生成脚本位于 `tools/build_learning_agent_report.py`，使用独立文档工具环境中的 ReportLab 和 Linux Liberation Sans 字体；不属于 DQN 运行依赖。ZIP 打包脚本 `tools/build_learning_agent_submission.py` 按明确清单打包并核对指定 Git 提交，不打包训练产物。

## 15. Teammate Integration

来源和 22 个导入文件的哈希见 [队友代码](teammates/README.md)。M1 的 34 维排列与九动作编号相同，但速度归一化、奖励和加速度限幅不同；它作为独立诊断环境保留，未替换冻结正式环境。新展示入口默认连接正式环境并复用 M1 的策略/叠图，保存 PNG，无需视频编码依赖：

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python teammate_demo.py \
  --environment project --policy random --seed 25 --steps 20 \
  --output /tmp/me5418-teammate-demo-new.png
```

输出必须是尚不存在的文件。这是演示策略，不是训练模型性能。三个种子的 9 份旧 checkpoint 在线权重均通过 15 条真实观测的 Q 等价检查；原训练文件哈希保持不变。由于源码签名改变，旧 best/last 需显式调用 `legacy_checkpoint.load_legacy_inference`，旧 resume 继续使用原冻结代码和协议；不静默兼容恢复、不重写文件。完整命令、兼容条件、测试证据及 Stage 4 尚缺事项见 [整合记录](docs/teammate_code_integration.md)。
