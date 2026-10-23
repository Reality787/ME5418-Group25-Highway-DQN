# ME5418 项目规格与待确认项

项目：**Deep Reinforcement Learning for Highway Lane Change and Speed Decisions**。

## 资料实际读取情况

本项目最初检查时没有 PDF；当前项目目录名为 `highway`。
随后用户明确提供了 `ME5418_Group25_Proposal(9-actions).pdf`。
阶段 0 已读取这份 PDF 的全部两页文字，并查看两页渲染图。
页脚标明这是 2026 年 10 月 6 日修订的九动作版本；第 1 页给出场景，
第 2 页给出 RL Cast、动作表、网络、奖励和评测计划。
固定配置与用户给定规格一致，未发现冲突。源 PDF 保持原位，未复制或修改。

**Lecture 6b 尚未提供，也未直接读取**；三个模块的划分依据用户说明，
没有伪造课件页码。没有访问其他 ChatGPT 对话中的附件，
没有使用旧版五动作 proposal。阶段 1A 按用户要求查阅了相关官方接口文档。
本轮没有重新读取 PDF，沿用阶段 0 的核对记录；实际检查了项目文件和本地已安装库源码。

本文将 PDF 作为项目规格资料。文档中的实施计划不视为扩大本次任务的指令。
阶段 1B-3 已完成项目奖励、正式 Gym 环境、测试及环境提交包。
当前授权阶段 2：Q 网络、数值测试、保存加载和正式环境输入对接；不实现经验回放或启动驾驶训练。
阶段 2 按用户许可补装缺少的 CPU PyTorch，未改变已有依赖。详见 network_design.md 和 progress.md。
本页保留历阶段事实与验证历史；旧节中的“当时/本阶段”边界不代表当前功能仍缺失。
阶段 1B-3 的实现和工程来源见末节，实际验收记录见 environment_validation.md。

## 一、proposal 已确定的内容（已核对最终 PDF）

### 仿真与场景

使用开源 HighwayEnv 提供道路、车辆动力学、交通行为和底层控制。
项目环境采用 Gymnasium 接口是本次用户明确指定的工程要求。

| 项目 | 已确定值 |
| --- | --- |
| 道路 | 三条同向直车道 |
| 受控车辆 | 1 辆 |
| 初始周围车辆 | 20 辆 |
| 回合最长时间 | 40 秒 |
| 期望速度 | 27.80 m/s |
| 感知范围 | 100 米 |
| 决策频率 | 5 Hz |
| 仿真频率 | 15 Hz |
| terminated | 碰撞或离开道路 |
| truncated | 达到时间上限 |

### 34 维观测

| 组成 | 维数 | 内容 |
| --- | --- | --- |
| 交通特征 | 18 | 三条车道的前车/后车，每个位置包含 presence、bumper gap、relative speed |
| 自车/控制特征 | 7 | 纵向速度、横向速度、横向偏移、当前车道、目标车道、目标速度、换道状态 |
| 动作历史 | 9 | 上一联合动作的 one-hot 特征 |

proposal 只确定组成与维数，没有给出完整归一化、缺失车辆编码或初始动作历史规则。
阶段 1B-2 按本轮用户约定及明确的工程选择落实这些细节，见后文；不归因为 proposal。

### 固定的九个联合动作

| 编号 | 名称 |
| --- | --- |
| 0 | Left + slower |
| 1 | Left + hold |
| 2 | Left + faster |
| 3 | Keep + slower |
| 4 | Keep + hold |
| 5 | Keep + faster |
| 6 | Right + slower |
| 7 | Right + hold |
| 8 | Right + faster |

目标速度档位为 `[0.00, 6.95, 13.90, 20.85, 27.80]` m/s。
slower、hold、faster 分别使**已有目标速度索引**变化 -1、0、+1。
Keep 保留已有目标车道；hold 保留已有目标速度。
期望速度不是已经确定的初始速度或初始目标速度。

未来实现时，必须在**同一次决策中更新两个目标，再推进一个决策间隔**。
不得通过连续两次 `env.step()` 拼接出一个联合动作。

结构动作掩码只限制以下情况：

1. 道路边缘向外换道。
2. 换道未完成时发起新换道。
3. 目标速度索引越界。

掩码同时用于探索、贪心动作选择和 TD 目标。
不擅自增加基于交通间隙的安全过滤器。

### 网络与学习

采用 PyTorch DQN，网络为 `34 → 128 → 128 → 9`，隐藏层使用 ReLU，
折扣因子 `gamma = 0.99`。输出为九个动作的 Q 值。
后续自行实现经验回放、epsilon-greedy、目标网络和 TD 更新，
不用现成训练库替代本项目计划自行实现的 DQN 核心。

### 奖励与主要对照

奖励包含归一化进度、碰撞/离路惩罚、短车头时距惩罚、换道发起成本和目标速度变化成本。
一次联合决策可以同时产生两种动作成本。权重将使用开发/验证场景确定，
PDF 没有给定最终权重或具体计算公式；阶段 1B-3 采用用户新指定的初始工程公式，见后文。
PDF 将加速度/jerk 惩罚列为可靠学习之后的可选项，
不把它们加入本阶段的固定奖励配置。

主要对照是 gap-based controller。DQN 与该基线共享观测、动作、
执行器限制和初始场景。基线可依据交通间隙制定策略，但不得改变共享结构掩码。

### PDF 已明确的后续实验计划（仅记录，不执行）

第 2 页还明确了以下计划，不能把它们描述成已有实验结果：

- 训练随机化交通密度、初始间隙和周围车辆速度；具体范围尚未给出。
- 验证后冻结来自 3 个训练种子的策略，在新的范围内场景、保留组合和部分范围外条件下测试；
  每个场景组、每个训练种子使用 100 个匹配回合，不在测试时重新调参。
- 已检查的场景转为开发数据，最终测试使用新的随机种子；基线规则在开发数据上确定后冻结。
- 主要指标为碰撞率、完成情况，以及相对“全时域按目标速度行驶距离”的进度。
  学习曲线、种子间不确定性、车头时距、换道及加速度/jerk 用于辅助分析。
- 训练预算由验证曲线确定，不能据此捏造固定训练步数或宣称 DQN 已优于基线。

以上评测规模仅记录在文档中，本阶段不建立评测循环或运行配置。

## 二、这次建议的工程组织方式

当前目录直接作为项目根目录，不建立嵌套 `highway_project/`。
以用户指定的 Lecture 6b 三模块划分组织代码：

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| Environment | `environment.py` | 场景、观测、联合动作、结构掩码、奖励和终止接口 |
| Neural Network | `network.py` | 输入观测，输出九个 Q 值 |
| Learning Agent | `agent.py` | 动作选择、TD 更新、目标网络同步、保存/加载 |

辅助文件包含固定配置、经验池、基线、三个命令入口、配置测试和输出目录。
这属于本次工程组织安排，不声称是 proposal 或课件指定的目录结构。

阶段 0 当时使用普通 Python 类与标准库类型标注表达接口，未实现方法抛出
`NotImplementedError`，当时未继承 Gymnasium Env 或 PyTorch Module。
阶段 1B-3 的正式 HighwayLaneEnv 已继承 Gymnasium Env；阶段2的DQNNetwork已继承PyTorch Module。
学习器、经验池和基线仍保留骨架；网络支持单条与批量张量，详见network_design.md。

接口约定：

- `reset(seed=..., options=...)`：未来返回 `(obs, info)`。
- `step(action)`：未来返回 `(obs, reward, terminated, truncated, info)`。
- `get_action_mask()`：未来返回长度 9 的布尔序列，True 表示动作可用。
- `render()` / `close()`：未来负责渲染与资源释放。
- 经验池保存 `obs`、`action`、`reward`、`next_obs`、`terminated`、
  `truncated`、`next_action_mask`，不把两个结束标志合并。
- 真实终止不进行 bootstrap；时间截断需单独处理。
  若以后使用自动重置，应保留实际最终观测，避免把下一回合初始状态用于 TD 目标。

计划的交互流程为：命令入口创建环境及控制器 → 环境返回观测和结构掩码 →
控制器选择联合动作 → 环境一次执行动作 → 训练入口将转移写入经验池 →
agent 采样并学习。该流程只是后续设计，当前没有运行。

## 三、后续仍需确认的实现细节（TODO）

- 源文件核对：Lecture 6b 原文仍未提供；最终九动作 proposal 已核对。
- 观测：阶段 1B-2 已落实排列、固定归一化、缺失/历史编码、速度符号、
  投影 bumper gap 与坐标规则；后续验证尺度适用性、压线单槽简化与航向速度近似。
- 初始状态：自车速度、车道、目标车道、目标速度索引及场景随机化范围。
- 执行器：阶段 1B-1 已实现目标适配、严格非法动作处理和暂定几何完成判据；
  最终执行器限幅及暂定容差在更多场景中的适用性仍待确认。
  **训练前确认底层控制限制，并确保 DQN/基线一致**，不得从旧版 proposal 导入限值。
- 掩码：阶段 1B-1 已落实三个结构限制，并已在正式环境复用；后续 DQN 探索、贪心和 TD 目标仍需使用。
- 奖励：阶段 1B-3 已落实本轮指定的初始公式、权重、THW 与单次事件计数；
  后续仍需在开发/验证场景验证和调参，不能声称参数已最优。
- 学习：学习率、训练步数、经验池容量、批大小、预热、更新频率、epsilon 调度、
  优化器、损失函数和目标网络同步周期。
- TD：时间截断 bootstrap 策略、最终观测获取方式，以及终止样本的掩码处理。
- 基线与评测：驾驶规则阈值、指标的具体公式、场景组构造、随机种子的具体取值，
  以及相同初始场景的复现方案。训练种子数量和每组匹配回合数已在上一节记录。
- 工程：PyTorch 的版本与兼容性、训练设备、检查点格式和日志格式。
  阶段 1A 的原生仿真依赖版本及 none/human 渲染已验证，详见 README.md。

未确定的项目参数保持 TODO，不从旧版 proposal 搬入数值；本轮必需的工程选择在下方单独标注。

## 阶段 1A：真实原生仿真检查

新增的 make_stock_env(render_mode=None) 调用真实 gymnasium.make("highway-v0")，
读取 config.py 中的车道数、车辆数、受控车数、时长和频率，并开启 offroad_terminal。
阶段 1A 当时将 HighwayLaneEnv 保留为独立 TODO 接口；最终 34 维观测和九动作配置不变。

当前版本（HighwayEnv 1.12.1）的原生检查环境实际提供：

- Kinematics 观测，Box(-inf, inf, (5, 5), float32)，列为 presence/x/y/vx/vy。
- Discrete(5)：0=LANE_LEFT、1=IDLE、2=LANE_RIGHT、3=FASTER、4=SLOWER。
- 原生目标速度 [20, 25, 30] m/s、初始速度 25 m/s、感知距离 200 m。
- 原生奖励及执行器逻辑，未实现项目奖励或项目结构掩码。

上述表格记录的是 stock 模式；阶段 1B-1 已在独立 joint-debug 模式实现联合动作及五档速度。
阶段 1B-2 又在 observation-debug 实现了项目观测与 100 米感知，当时项目奖励仍待实现。
不能把原生动作编号套入项目九动作映射，
也不能把原生观测伪造为 34 维或把原生空间声明为 Discrete(9)。

demo.py 仅使用真实原生 action_space 采样。每回合使用 seed + 回合下标重置，
并在 reset 重建空间之后对动作空间设置同一种子。任一终止/截断标志出现后结束回合，
finally 负责关闭环境；回合循环和步数循环均有界。human 模式启用实时播放，
不改变 5 Hz 决策和 15 Hz 仿真频率。

本阶段通过 15 项 unittest（7 项配置、8 项真实仿真），三回合无窗口演示及一回合
human 演示。X11 实际窗口的两帧渲染表面已查看，车辆位移和窗口资源释放已检查。
独立的零周围车辆 IDLE 场景验证了时间上限截断；正式场景仍为 20 辆周围车辆。
这些结果只是仿真连接检查，不是训练结果或模型性能。命令、版本和逐回合记录见 README.md。

官方接口参考：
[Quickstart](https://highway-env.farama.org/quickstart/)、
[Highway 环境](https://highway-env.farama.org/environments/highway/)、
[Gymnasium Env](https://gymnasium.farama.org/api/env/)。
同时核对了本机 me5418-highway 环境的实际包源码，未复制训练库示例。

## 阶段 1B-1：联合动作控制与工程选择

本节保留阶段 1B-1 的实现及验证历史：当时只完成九个联合动作、五档目标速度和结构掩码。
当时 34 维观测、100 米感知尚未实现，当时自定义奖励及 DQN 也未实现；后续进展见后文。

### proposal 已确定且本阶段已落实的部分

九动作编号和五档速度保持 config.py 中的固定定义。slower/hold/faster 只对
已有目标索引加 -1/0/+1；Keep 保留已有目标车道，hold 保留已有目标速度。

JointActionController.apply() 校验整个动作后，同时写入
target_lane_index、speed_index、target_speed，然后 JointDebugEnv 只调用一次
内层 step(IDLE)。内层 IDLE 编号从原生 actions_indexes["IDLE"] 读取，当前为 1。
项目动作 0～8 不直接传给原生五动作接口。路面车辆实际运动仍由 HighwayEnv
ControlledVehicle 控制与 Vehicle 动力学产生，不靠直接修改实际速度或瞬移。

本机 MDPVehicle.act("FASTER"/"SLOWER") 会依据实际车速重新选档，因此不能用于
本项目的纵向分量。IDLE 与每个物理帧的 act(None) 不覆盖已设置的目标。
一次联合 step 在当前 15/5 Hz 配置下推进 0.2 秒和 3 个物理步。

结构掩码仅使用自车控制状态与道路拓扑，不读取交通间距、THW/TTC 或预测碰撞：
道路边缘向外、尚未完成换道时再发起左/右换道、目标索引越界时屏蔽整个动作。
换道期间 Keep 类动作仍按速度边界可用，Keep + hold 始终满足结构条件。

### 本阶段工程选择：暂定，待更多场景验证

| 项目 | 当前明确实现 | 理由 |
| --- | --- | --- |
| 换道完成 | 当前车道等于目标车道，且目标中心横向误差 ≤ 0.10 m，航向误差 ≤ 0.02 rad，三者同时满足 | 最近车道编号会在车辆仅跨过半个车道时变化；还需几何对齐 |
| 容差配置 | LANE_CHANGE_LATERAL_TOLERANCE_M=0.10；LANE_CHANGE_HEADING_TOLERANCE_RAD=0.02 | 0.10 m 约为原生 4 m 车道宽的 2.5%，0.02 rad 约 1.15°；不是 proposal 数值 |
| 初始目标索引 | JOINT_INITIAL_TARGET_SPEED_RULE="nearest_actual_speed_ties_lower" | 只在 reset 时选最接近实际初速的档位，等距选低档，减少初始控制突变；浮点平局只容纳机器精度误差 |
| 显式初始化 | reset(options={"initial_target_speed_index": 0}) 等 0～4 值优先 | 0 是有效索引，不使用 truthy 默认值；默认原生实际初速 25 对应目标 27.80 |
| 初始目标车道 | reset 后采用当前车道 | 不带入上一回合未完成的换道 |
| 非法动作 | 错误类型抛 TypeError；越界或被掩码禁止抛 ValueError | 在目标、时间或车辆状态修改前拒绝；不替换为动作 4，不执行部分分量 |
| 图形输入 | joint-debug 的独立 viewer 禁用方向键驾驶，保留画面操作与关闭 | 原生键盘会直接调用五动作并绕过掩码；stock viewer 不变 |

“暂定，待验证”表示这些选择虽已通过本阶段可控测试，仍需在后续更多速度、
交通条件和整合场景中检验；不宣称来自 proposal。
换道状态按当前几何关系计算，不依赖可能残留的布尔锁。
如果换道中减速停车，未对齐时保持锁定，Keep + faster 可恢复运动。

库的 ControlledVehicle 构造器使用 target_speed or speed，会误处理目标 0。
本实现在原生 reset 构造结束后显式赋五档、目标索引和目标速度，从而保留零目标。
真实测试验证了从 25 m/s 接近停车，再经 Keep + faster 起步；不要求有限步内精确为零。

### 复用的底层参数与实际限幅

以下来自已安装的 highway-env 1.12.1 源码和实际车辆属性，
不是 proposal 给出的数值，本阶段未改写这些参数：

| 参数或行为 | 实际值 |
| --- | --- |
| TAU_ACC / TAU_HEADING / TAU_LATERAL / TAU_PURSUIT | 0.6 / 0.2 / 0.6 / 0.1 秒 |
| KP_A / KP_HEADING / KP_LATERAL | 1/0.6 / 5 / 1/0.6 |
| 纵向控制 | acceleration = KP_A × (target_speed − actual_speed) |
| 常规固定加速度限幅 | 无；本阶段未自行增加数值上限 |
| 航向参考偏差限幅 | ±π/4 |
| 转向角限幅 | ±π/3 rad |
| MIN_SPEED / MAX_SPEED | -40 / 40 m/s；实际越界后修正加速度，不是直接裁剪实际速度 |
| 碰撞后的底层处理 | steering=0，acceleration=-speed |
| 车辆 LENGTH / WIDTH | 5 / 2 m |

最终项目的加速度等执行器限值仍待确认，不从旧版 proposal 导入。
源码核对对象包括 DiscreteMetaAction、MDPVehicle、ControlledVehicle、
Vehicle.clip_actions/step 和 AbstractEnv.step/_simulate，未修改 site-packages。

### 调试接口与观测边界

make_joint_debug_env() 返回真实 Gymnasium Wrapper，对外为 Discrete(9)，
内层 env.unwrapped.action_space 仍是原生 Discrete(5)，仅用 IDLE 推进。
原生 Kinematics 观测仍为 (5, 5)，感知仍为 200 m，奖励仅为原生调试输出。
阶段 1B-1 当时将 HighwayLaneEnv 保留为未实现接口，不将调试包装器冒充正式环境。

get_action_mask() 返回 shape=(9,) 的 bool 数组；reset 和 step 的 info["action_mask"]
始终是返回状态的掩码。info["joint_debug"] 包含 action_id/action_name/native_action_id
以及 before、targets_applied、after 三份快照。每份快照记录时间、物理步数、
当前/目标车道、实际/目标速度、目标索引、换道状态、几何误差和掩码。
reset 没有执行动作，动作字段及 before/targets_applied 为 None。

JointActionController 可在后续最终环境中直接复用。演示从同一 get_action_mask()
的有效编号中采样，以每回合的 seed 初始化 NumPy Generator。
每回合最多运行 duration×policy_frequency 再加一个浮点容差决策步，循环有界。
所有环境通过 finally 关闭，Ctrl+C 或关闭 joint-debug 窗口会停止演示。

### 本阶段实际验证

使用 ${CONDA_PREFIX}/bin/python（Python 3.11.16）。
依赖版本沿用阶段 1A，无安装、升级或系统环境修改。

- unittest：33 项通过，0 失败、0 跳过；原有 15 项全部保留，新增 18 项联合动作测试。
- 参数化子测试覆盖全部九动作、两种分量边界、错误类型/越界/掩码拒绝、
  目标与实际速度分离、reset、可复现性、停车再起步以及交通间距独立性。
- 真实换道过程中，车道编号已改变但尚未居中时仍锁定，实际收敛后能够再次换道。
- stock 无窗口 3 回合通过；seed 0/1/2 仍分别为 10/52/47 次决策，与阶段 1A 一致。
- joint-debug 无窗口 3 回合通过；seed 0/1/2 均为 200 次决策，到达 40 秒 truncated。
- joint-debug human 命令已运行，最后完整日志到 22.8 秒，之后收到关闭/中断，
  退出码 130；未将其记作完整 40 秒图形回合通过。
- 另行在真实 X11 窗口检查并查看前后渲染帧，确认车辆运动；注入方向键事件后
  联合目标未被覆盖，窗口关闭事件正确中断，viewer/display 均已释放。

上述结果是动作控制与仿真连接测试，不是训练结果、策略性能或安全性结论。

## 阶段 1B-2：观测与感知工程约定

### 已确定要求与工程选择的区别

proposal 确定 18+7+9 共 34 维以及 100 米感知。以下固定索引、中心欧氏距离、
缺失/历史编码和速度符号来自本轮用户明确要求；车道单槽归属、并列排序、
车身投影以及固定归一化尺度为本轮工程实现选择，暂定待后续验证。
这些规则均有实际代码与数值测试，不宣称来自课件或 proposal。

新增 `observation.py`：`ObservationBuilder` 从真实 road/vehicle 状态构造输入；
`ObservationDebugEnv` 包装已有 `JointDebugEnv`，不复制动作逻辑。
`environment.make_observation_debug_env()` 是独立调试入口，不是完成后的 HighwayLaneEnv。
stock 仍返回原生五动作与 (5,5) 观测，joint-debug 仍返回九动作与 (5,5) 观测；
仅 observation-debug 返回真实九动作与 34 维项目观测。三者均保留原生调试奖励。

### 固定 34 维名称、单位与范围

固定按实际车道编号 0、1、2 排列，不随自车换道重排槽位。
对每一维计算 `scaled = raw / scale`，再裁剪到下表 Box 区间，最终转为 float32。
配置中的 `OBSERVATION_FEATURE_NAMES/SCALES/LOW/HIGH/UNITS` 与表一致。

| 索引 | 英文特征名 | 物理含义 | 原始单位 | 裁剪前归一化 | Box 范围 |
| --- | --- | --- | --- | --- | --- |
| 0 | lane_0_front_presence | 车道 0 前车 | 0/1 | 保持 0/1 | [0,1] |
| 1 | lane_0_front_bumper_gap | 车道 0 前车 | m | gap/100 | [0,1] |
| 2 | lane_0_front_relative_speed | 车道 0 前车 | m/s | (other_v_long − ego_v_long)/40 | [-1,1] |
| 3 | lane_0_rear_presence | 车道 0 后车 | 0/1 | 保持 0/1 | [0,1] |
| 4 | lane_0_rear_bumper_gap | 车道 0 后车 | m | gap/100 | [0,1] |
| 5 | lane_0_rear_relative_speed | 车道 0 后车 | m/s | (other_v_long − ego_v_long)/40 | [-1,1] |
| 6 | lane_1_front_presence | 车道 1 前车 | 0/1 | 保持 0/1 | [0,1] |
| 7 | lane_1_front_bumper_gap | 车道 1 前车 | m | gap/100 | [0,1] |
| 8 | lane_1_front_relative_speed | 车道 1 前车 | m/s | (other_v_long − ego_v_long)/40 | [-1,1] |
| 9 | lane_1_rear_presence | 车道 1 后车 | 0/1 | 保持 0/1 | [0,1] |
| 10 | lane_1_rear_bumper_gap | 车道 1 后车 | m | gap/100 | [0,1] |
| 11 | lane_1_rear_relative_speed | 车道 1 后车 | m/s | (other_v_long − ego_v_long)/40 | [-1,1] |
| 12 | lane_2_front_presence | 车道 2 前车 | 0/1 | 保持 0/1 | [0,1] |
| 13 | lane_2_front_bumper_gap | 车道 2 前车 | m | gap/100 | [0,1] |
| 14 | lane_2_front_relative_speed | 车道 2 前车 | m/s | (other_v_long − ego_v_long)/40 | [-1,1] |
| 15 | lane_2_rear_presence | 车道 2 后车 | 0/1 | 保持 0/1 | [0,1] |
| 16 | lane_2_rear_bumper_gap | 车道 2 后车 | m | gap/100 | [0,1] |
| 17 | lane_2_rear_relative_speed | 车道 2 后车 | m/s | (other_v_long − ego_v_long)/40 | [-1,1] |
| 18 | ego_longitudinal_speed | 自车道路纵向速度 | m/s | v_long/40 | [-1,1] |
| 19 | ego_lateral_speed | 自车道路横向速度 | m/s | v_lateral/10 | [-1,1] |
| 20 | ego_lateral_offset | 相对当前车道中心的有符号横向偏移 | m | offset/2 | [-1,1] |
| 21 | current_lane | 几何当前车道 0/1/2 | lane_id | lane_id/2 | [0,1] |
| 22 | target_lane | 已有目标车道 0/1/2 | lane_id | lane_id/2 | [0,1] |
| 23 | target_speed | 已有目标速度 | m/s | target_speed/27.80 | [0,1] |
| 24 | lane_change_in_progress | 现有控制器的换道进行状态 | 0/1 | 保持 0/1 | [0,1] |
| 25 | previous_action_0 | 上一联合动作是否为 0 | 0/1 | 保持 0/1 | [0,1] |
| 26 | previous_action_1 | 上一联合动作是否为 1 | 0/1 | 保持 0/1 | [0,1] |
| 27 | previous_action_2 | 上一联合动作是否为 2 | 0/1 | 保持 0/1 | [0,1] |
| 28 | previous_action_3 | 上一联合动作是否为 3 | 0/1 | 保持 0/1 | [0,1] |
| 29 | previous_action_4 | 上一联合动作是否为 4 | 0/1 | 保持 0/1 | [0,1] |
| 30 | previous_action_5 | 上一联合动作是否为 5 | 0/1 | 保持 0/1 | [0,1] |
| 31 | previous_action_6 | 上一联合动作是否为 6 | 0/1 | 保持 0/1 | [0,1] |
| 32 | previous_action_7 | 上一联合动作是否为 7 | 0/1 | 保持 0/1 | [0,1] |
| 33 | previous_action_8 | 上一联合动作是否为 8 | 0/1 | 保持 0/1 | [0,1] |

presence 的原始值即 0/1；间距先处理纵向重叠为 0。
没有该槽车辆时，raw 为 `(0, 100, 0)`，归一化为 `(0, 1, 0)`；
raw gap=100 是缺失哨兵，不应当作测得距离。有车时即使 gap/100 接近 1，presence 仍为 1。

各固定尺度的理由：

- 纵向速度 40 m/s 对应库名义速度边界，允许负值；27.80～40 m/s 的实际速度不会被提前压到 1。
- 横向速度 10 m/s 是宽松的固定工程范围，给直路换道留出余量；不是统计拟合或车辆限制。
- 相对速度 40 m/s 保留快慢差的符号；若超过范围则明确诊断，不重新缩放。
- 横向偏移 2 m 对应原生 4 m 车道的半宽；道路法向朝编号增加的方向为正。
- gap 100 m、车道编号 2、目标速度 27.80 m/s 使用明确感知界、三车道编号上界与最大目标档。
- 这些尺度都不是动作加速度限幅，也不是每回合更新的标准化统计量。

`observation_space` 为逐项范围对应的 `Box(shape=(34,), dtype=float32)`；所有输出必须有限。
若有限原始值超出范围，则裁剪，并在 `info["observation_debug"]["clipped_features"]`
逐项提供 `index/name/raw_value/unclipped_normalized/normalized_value/low/high`。
观测诊断还提供完整特征名、原始值与归一化值。
NaN/Inf 是错误，抛出 ValueError，不通过 `nan_to_num`、静默替换或裁剪掩盖。

### 感知、车道归属与距离

当前实现明确限定三条同向平行 StraightLane；其他道路几何不在本阶段支持范围内。
以车道 0 的道路纵向单位向量 `t` 和横向单位向量 `n` 为共同参考；三车道平行。
从仿真世界读取车辆是模拟传感，输出遵守以下规则：

1. 排除自车。用双方当前位置计算中心平面欧氏距离 `d = ||p_other − p_ego||₂`。
   `d <= 100 m` 可见，`d > 100 m` 不可见；先过滤，再选择前后车。
2. 车道归属仅取当前位置到中心线横向距离最小的一条车道；等距取较低车道编号。
   不使用缓存车道、周围车目标车道、意图或规划轨迹。
   这是单槽近似：压线/换道车辆不会同时填充两车道，越过中心分界时槽位会离散变化。
3. 纵向位置差 `ds = s_other − s_ego`，`ds >= 0` 归前车，`ds < 0` 归后车；
   每个槽选择 `|ds|` 最小的可见车辆。选择依据是纵向中心距离，不是按保险杠间距排序。
   `|ds|` 并列时使用 `road.vehicles` 列表中先出现的车；同一世界顺序下确定可复现。
4. 双方都是中心定位的矩形车身。对于车长 L、车宽 W、航向与道路之差 θ，
   道路纵向投影半长 `h = (L*|cos θ| + W*|sin θ|)/2`。
   输入保险杠间距 `gap = max(0, |ds| − h_ego − h_other)`。
   零航向、双方长 5 m 时，中心距 20 m 对应 gap=15 m，而不是 20 m。
   该投影包络考虑非零航向，但不是二维碰撞最短距离，也不检查两车的横向车身重叠。
   纵向投影重叠时 gap=0，presence 仍为 1。
5. `v_long = vehicle.velocity · t`，`v_lateral = vehicle.velocity · n`；
   前后槽统一 `relative_speed = other_v_long − ego_v_long`，后车不翻转符号。
   本机 `vehicle.velocity = speed * [cos(heading), sin(heading)]`，未计转向侧偏 β；
   物理积分中的位移方向使用 heading+β。因此本轮观测是航向投影速度，不能称为
   包含侧偏的精确位置差分速度。该近似在后续训练前仍需确认。

感知之外的有限位置车辆在可见性过滤后，不再读取其速度、航向或目标信息来构造特征。
可见车辆只用物理位置、速度、航向和尺寸；隐藏目标速度/车道不会进入输入。
自车目标速度/车道属于已定义的控制状态特征，可以读取。
100 米过滤作用于本轮输出，不更改底层 stock 原生 200 米观测配置。

### 历史时序、换道标记及只读性

`reset()` 清空上一动作，25–33 全 0，这是回合初始特殊编码。
`step(a_t)` 先让内层严格校验并完成真实动作，再记录外部联合动作 a_t，
因此返回的新状态第 `25+a_t` 项为 1，其余历史位为 0；不记录底层 IDLE 编号。
被拒绝动作在内层抛错，外层不会更新历史；下次 reset 也不会携带上一回合历史。

换道标记直接调用 `JointActionController.is_lane_change_in_progress()`，
与掩码使用同一判定，没有第二套容差。当前车道特征按物理位置几何归属构造。
`info["action_mask"]` 和 `info["joint_debug"]` 保留内层的当前状态掩码与控制诊断。
`info["observation_debug"]` 单独存诊断，绝不拼入 34 维策略输入。

`ObservationBuilder.build(previous_action)` 是只读构造：不 step、不改变实际或目标状态、
不消费环境或采样器随机数；没有上一动作的内部副作用。历史由包装器维护。
`get_observation()` 可重复读取同一状态。包装前后相同动作都只推进 0.2 秒/3 个物理步。

### 一个真实构造的已知场景

在独立测试实例中设定自车中心 `(200, 4.5)`、航向 0、实际速度 20 m/s，
目标车道 1、目标速度 13.90 m/s。三条车道中心 y=0/4/8；所有车 L=5 m、W=2 m。
周围车均航向 0、位于各自车道中心，`(lane, 相对纵向位置, speed)` 为：
`(0,+20,10), (0,-30,30), (1,+40,15), (1,-50,25), (2,+60,0), (2,-70,20)`。
全部在 100 米内。本例没有执行上一动作，历史全 0。
自车偏离车道 1 中心 0.5 米，大于既有 0.10 米容差，所以换道进行标记为 1；
它是控制器几何未对齐状态，不额外推断驾驶意图。

下面来自该真实仿真状态的观测构造结果（2026-10-23 恢复后再次核对）：

| 索引 | 特征名 | 原始值 | 归一化值 |
| --- | --- | ---: | ---: |
| 0 | lane_0_front_presence | 1 | 1.000000 |
| 1 | lane_0_front_bumper_gap | 15 | 0.150000 |
| 2 | lane_0_front_relative_speed | -10 | -0.250000 |
| 3 | lane_0_rear_presence | 1 | 1.000000 |
| 4 | lane_0_rear_bumper_gap | 25 | 0.250000 |
| 5 | lane_0_rear_relative_speed | 10 | 0.250000 |
| 6 | lane_1_front_presence | 1 | 1.000000 |
| 7 | lane_1_front_bumper_gap | 35 | 0.350000 |
| 8 | lane_1_front_relative_speed | -5 | -0.125000 |
| 9 | lane_1_rear_presence | 1 | 1.000000 |
| 10 | lane_1_rear_bumper_gap | 45 | 0.450000 |
| 11 | lane_1_rear_relative_speed | 5 | 0.125000 |
| 12 | lane_2_front_presence | 1 | 1.000000 |
| 13 | lane_2_front_bumper_gap | 55 | 0.550000 |
| 14 | lane_2_front_relative_speed | -20 | -0.500000 |
| 15 | lane_2_rear_presence | 1 | 1.000000 |
| 16 | lane_2_rear_bumper_gap | 65 | 0.650000 |
| 17 | lane_2_rear_relative_speed | 0 | 0.000000 |
| 18 | ego_longitudinal_speed | 20 | 0.500000 |
| 19 | ego_lateral_speed | 0 | 0.000000 |
| 20 | ego_lateral_offset | 0.5 | 0.250000 |
| 21 | current_lane | 1 | 0.500000 |
| 22 | target_lane | 1 | 0.500000 |
| 23 | target_speed | 13.9 | 0.500000 |
| 24 | lane_change_in_progress | 1 | 1.000000 |
| 25 | previous_action_0 | 0 | 0.000000 |
| 26 | previous_action_1 | 0 | 0.000000 |
| 27 | previous_action_2 | 0 | 0.000000 |
| 28 | previous_action_3 | 0 | 0.000000 |
| 29 | previous_action_4 | 0 | 0.000000 |
| 30 | previous_action_5 | 0 | 0.000000 |
| 31 | previous_action_6 | 0 | 0.000000 |
| 32 | previous_action_7 | 0 | 0.000000 |
| 33 | previous_action_8 | 0 | 0.000000 |

随后真实执行 Keep + hold，只推进一个 0.2 秒决策间隔、3 个物理步，
返回观测第 29 项（25+动作 4）变为 1。测试夹具仅用于建立初始状态，不用假的 step 或奖励。

### 底层速度控制诊断：不修改控制参数

本轮重新只读检查已安装 highway-env 1.12.1 的
`ControlledVehicle.speed_control/act`、`Vehicle.clip_actions/step` 与项目联合控制器。
项目联合控制器仅改变目标，实际加速度指令在 `speed_control(target_speed)` 中产生：

`a = KP_A * (target_speed − speed) = (target_speed − speed) / 0.6`。

`ControlledVehicle.act()` 将该值写入 `action["acceleration"]`；每个物理帧的
`road.act()` 重新计算速度误差。`Vehicle.step(dt)` 先调用 `clip_actions()`，
然后通过显式 Euler 积分 `speed += action["acceleration"] * dt`，这里 dt=1/15 秒。

上一阶段动作 0 把目标从 20.85 改为 13.90 m/s，初始实际速度 25 m/s。
在无碰撞、未超出速度界的三个物理帧中：

| 帧 | 帧前速度 m/s | a 指令 m/s² | 帧后速度 m/s |
| --- | ---: | ---: | ---: |
| 1 | 25.000000000 | -18.500000000 | 23.766666667 |
| 2 | 23.766666667 | -16.444444444 | 22.670370370 |
| 3 | 22.670370370 | -14.617283951 | 21.695884774 |

这与真实日志 `t=0.2s, speed=21.695885` 一致；可写成
`v_3 = 13.9 + (25 − 13.9)*(1 − (1/15)/0.6)^3`。
0.2 秒的平均减速度约 -16.5206 m/s²，说明当前响应强；此处只诊断，不擅自修改。

实际限幅/覆盖的位置：

- `ControlledVehicle.steering_control()` 将航向参考相对偏差裁到 ±π/4；
  steering_control/act 对转向角使用 `MAX_STEERING_ANGLE=π/3`。
- `Vehicle.clip_actions()` **没有常规固定加速度上下限**。
  仅当 speed>40 时令 `a=min(a, 40−speed)`，当 speed<−40 时令 `a=max(a, −40−speed)`；
  这是速度越界后的修正，不是直接裁剪实际速度，也不是常规 ±a_max。
- crashed 时 `clip_actions()` 覆盖 steering=0、acceleration=−speed。
- 项目的五档目标速度和结构掩码限制目标索引，不等于限制加速度指令。

本轮没有修改这些公式、第三方源码或执行器限值，也没有导入旧版 proposal 的限值。
**TODO：训练前确认底层控制限制，并确保 DQN/基线一致**；同时确认航向速度近似与固定归一化适用性。

### 本阶段实际验证

使用 `${CONDA_PREFIX}/bin/python`，Python 3.11.16、
Gymnasium 1.4.0、HighwayEnv 1.12.1、NumPy 2.4.6，依赖未安装或升级。
完整演示在会话中断前已执行；2026-10-23 恢复后再次实际跑全部测试，
并复查已知场景和一步时序，没有重新开图形窗口。

- 56 项 unittest 全通过，0 失败、0 跳过；原有 33 项保留，新增 23 项。
  恢复后复测耗时 4.312 秒。具体覆盖真实 reset/step 空间、物理数值、最近槽位、自车排除、
  99.9/100.0/100.1 米与横向欧氏边界、含长宽航向的 gap、缺失/重叠、符号、隐藏目标独立性、
  历史/拒绝/reset、换道一致性、裁剪诊断、只读性、可复现性和包装时间不变。
- 15 个 Python 文件语法检查通过；`demo.py --help` 通过；禁用 site 导入 environment/demo
  不强制加载仿真依赖，没有导入即启动回合。
- stock 无窗口 seed 0/1/2：10/52/47 次决策，原生奖励 8.282599/38.433912/37.939393，
  terminated=True、truncated=False，与阶段 1A/1B-1 一致。
- joint-debug 与 observation-debug 各自无窗口 seed 0/1/2：每回合 200 次决策、40 秒，
  terminated=False、truncated=True；各对应奖励 148.462411/146.315262/146.737987。
- observation-debug human：真实 X11，seed 0 完整 200 次决策/40 秒，退出码 0，
  原生调试奖励 148.462411，terminated=False、truncated=True。
  本轮没有额外截图或人工视觉判读；该记录只声称真实桌面模式完整运行。

这些检查只验证仿真/控制/观测，不代表训练结果、模型性能或安全性。
完整可复制命令见 README.md；`--observation-steps N` 控制每回合前 N 步完整表，
默认 N=2，初始表总会输出，任何步骤的裁剪仍会报告。

## 阶段 1B-2 的历史边界

阶段 1A stock、阶段 1B-1 joint-debug 保留；阶段 1B-2 在 observation-debug 中
实现真实 34 维观测、100 米感知、历史及诊断，并复用现有九动作/掩码。
该阶段结束时尚未实现自定义项目奖励、正式 HighwayLaneEnv、DQN、经验回放、基线、训练或正式评测。
该阶段没有安装依赖、改写第三方源码或系统环境，完成阶段 1B-2 后停止。
后续新增实现与本轮验收范围见下一节。

## 阶段 1B-3：正式 Gym 环境和初始项目奖励

本节记录本轮用户新授权的实现；上方阶段 1A/1B-1/1B-2 是历史过程与检查记录。
本轮先保留并自检既有 34 维观测、100 米感知、九动作、结构掩码及动作历史，
没有复制第二套控制器或感知实现，也没有将原生观测展平补零。

### 正式环境结构与接口

`from environment import HighwayLaneEnv, make_env` 对外公开正式入口。
实现放在 `project_environment.py`，`HighwayLaneEnv` 继承 `gymnasium.Env`，
内部持有现有 `ObservationDebugEnv`，后者继续组合 `JointDebugEnv` 和 HighwayEnv。
正式 `env.unwrapped` 仍是正式环境本身，Gym checker 不会绕过到原生五动作环境。

- `reset(seed=None, options=None) → (float32[34], info)`。
- `step(action) → (float32[34], float, bool, bool, info)`，动作空间为 `Discrete(9)`。
- `get_action_mask()` 返回共享控制器生成的新 `(9,)` 布尔副本。
- 支持 `render_mode=None/human/rgb_array`；metadata 声明 human/rgb_array。
- `render()` 转交底层渲染，rgb_array 采用离屏渲染，不接受原生键盘驾驶。
- `close()` 可重复；结束或关闭后 step 明确抛 `ResetNeeded`，必须 reset。

正常 step 仍只推进一次决策间隔，即 3 个物理步 / 15 Hz = 0.2 秒。
包装器不额外 step；观测、奖励和诊断只读取这次真实状态。reset 清除上一动作、
换道状态、累计奖励、累计步数和非法回退计数；实例随机数种子继续传递给真实仿真，
不重置全局随机数。同 seed 和相同动作序列应得到相同结果。
观测、掩码和 info 中数组均返回独立数据，后续步骤不得回写历史输出。

正式默认配置保持三车道、一辆受控车、20 辆周车、40 秒、5/15 Hz 和项目 100 米感知。
reset 不允许改变车道数、受控车数、时长、频率、离路终止或替换动作/观测接口；
允许测试实例显式覆盖周车数和初始车道等原有夹具参数。
默认 seeded 场景初始化不是随机化训练：训练密度、初始间距、周车速度的随机范围仍待确认。

### 本轮奖励公式：初始工程配置

以下公式与数值来自本轮用户明确要求，**不是最终 proposal 原文参数，也不是已验证最优值**。
全部权重和阈值集中在 `config.py`，纯函数 `reward.compute_reward()` 只接收物理量和事件标记，
不访问环境、不修改状态、不消费随机数、不产生额外仿真推进。

```text
r = dt * (1.0*p - 2.0*h)
    - 10.0*failure - 0.05*lane_change_started - 0.02*target_speed_changed

p = clip(longitudinal_displacement / (27.80*dt), 0, 1)
THW = bumper_gap / max(ego_longitudinal_speed, 0.1)
h = clip(1 - min_relevant_THW / 1.5, 0, 1)
```

| 配置名 | 值 | 单位 / 来源说明 |
| --- | ---: | --- |
| `REWARD_PROGRESS_WEIGHT` | 1.0 | 连续进度权重，用户本轮初始工程配置 |
| `REWARD_HEADWAY_WEIGHT` | 2.0 | 连续短时距权重，同上 |
| `REWARD_FAILURE_COST` | 10.0 | 单次失败成本，同上 |
| `REWARD_LANE_CHANGE_COST` | 0.05 | 单次实际换道发起成本，同上 |
| `REWARD_TARGET_SPEED_COST` | 0.02 | 单次实际目标索引变化成本，同上 |
| `REWARD_THW_THRESHOLD_S` | 1.5 | s，THW 参考阈值，同上 |
| `REWARD_MIN_SPEED_MPS` | 0.1 | m/s，THW 分母下限，同上 |

`dt` 从这一次真实物理更新数的增量除以仿真频率得到，通常为 0.2 秒。
`dt<=0` 明确报错，不以任意分母替代。位移由前后真实位置差投影到道路纵向计算。
progress 在 [0,1] 内裁剪，负向位移贡献为 0；**总奖励不裁成非负数**。
没有存活奖励、非法动作惩罚、加速度/jerk 项，也没有额外隐含碰撞奖励。

THW 采用 step 后状态及同一 `ObservationBuilder` 的原始物理量；不会把归一化向量
反缩放后当作真实间距或速度。正常行驶选当前车道；换道未完成时同时选当前和目标车道，
两个编号相同时去重。只使用这两条车道中通过相同 100 米过滤的最近前车，取最小 THW；
无相关可见前车时 h=0。零速分母使用 0.1；重叠间距为 0 时 h=1。
采用 step 后瞬时时距、当前/目标车道集合和单车道槽位均属初始工程近似，后续需验证。

事件成本从实际接受动作前与目标写入后的控制状态比较：目标车道变化才视为发起换道，
目标速度索引变化才扣速度变化成本。进行中的换道不重复扣发起成本，实际车速变化
不代表目标变化，保持和兼容回退均不会误扣目标成本。一个联合动作可以同时扣两项成本。
碰撞或离路合并为一个 failure 布尔，二者同时出现也只扣一次 10.0。

`info["reward_terms"]` 是五项加权贡献：`progress/headway/failure/lane_change/target_speed`，
求和等于返回 reward。`reward_diagnostics` 提供 dt、纵向位移、原始纵向速度、检查车道、
前车原始间距、进度比例、最小 THW、h 和实际事件标记；这些诊断不拼进 34 维策略输入。
原生奖励被项目奖励替换，只以 `native_reward_diagnostic` 留作对照诊断；
原生分项如存在则命名为 `native_reward_terms_diagnostic`，不与项目分项相加。
reset 时项目分项全为 0，`reward_diagnostics={}`，尚没有动作或原生单步奖励。

独立手算验收例：

| 条件 | 独立预期 |
| --- | ---: |
| dt=0.2、p=1、无前车/失败/事件 | 0.2 |
| 同上，实际同时发起换道并改变速度目标 | 0.2-0.05-0.02=0.13 |
| dt=0.2、p=1、THW=0.75 s、无失败/事件 | h=0.5，r=0 |

纯函数测试使用这些独立数值，不调用被测函数生成预期答案。
正式环境测试另设置确定物理初态后真实 step，核对没有重复添加原生奖励、
未裁剪速度用于 THW、一次性事件成本及碰撞/离路处理。

### 终止与非法动作约定

碰撞或离开道路设 `terminated=True`；40 秒时间上限且无失败设 `truncated=True`。
失败恰与时限同时发生时，失败优先：`terminated=True, truncated=False`，不能算作完成回合。
失败原因分别记录为 `collision` / `off_road`；可同时报告两原因，但 failure 成本只计一次。
随机交通中的碰撞是有效终止输出，不以随机演示全部无碰撞作为接口验收条件。

正式环境构造参数 `invalid_action_mode` 支持：

| 模式 / 情况 | 明确行为 |
| --- | --- |
| 默认 `hold`，有效编号但结构掩码禁止 | 整个请求改执行动作 4，保留两个既有目标并正常推进一步 |
| `raise`，有效编号但结构掩码禁止 | 在状态、目标、时间、历史及计数变化前报错 |
| 两种模式，类型错误、bool/非整数、编号不在 0～8 | 均报错，不采用兼容回退 |
| 两种模式，结构合法动作 | 相同执行路径和结果；不因交通风险改变动作 |

默认 hold 仅为了兼容不使用掩码的通用 Gym checker。回退不执行合法的部分分量，
不添加额外奖励/惩罚；`requested_action/executed_action/invalid_action` 明确记录，
动作历史对应实际动作 4。`invalid_action_count` 为回合累计回退次数。
正式随机演示和未来 DQN 仍应按掩码选动作，不能依赖回退隐藏策略错误。
joint-debug 和 observation-debug 保持原有严格拒绝行为。

### 调试模式、检查和提交材料

| 入口 | 实际观测 | 实际动作 | 奖励 |
| --- | --- | --- | --- |
| `make_stock_env()` | 原生 (5,5) | 原生 5 动作 | 原生调试奖励 |
| `make_joint_debug_env()` | 原生 (5,5) | 项目 9 动作 | 原生调试奖励 |
| `make_observation_debug_env()` | 项目 (34,) | 项目 9 动作 | 原生调试奖励 |
| `HighwayLaneEnv` / `make_env()` | 项目 (34,) | 项目 9 动作 | 本节项目奖励 |

`demo.py --mode project` 打印解释器、版本、空间、场景配置和奖励权重，
默认前两步的奖励分项及每回合步数、时长、累计项目奖励、碰撞/离路/截断和回退次数。
`--reward-steps N` 调节详细输出；旧模式和 `--observation-steps` 继续可用。
`validate_env.py` 对正式默认 hold 模式执行 Gymnasium `check_env`，再运行掩码随机交互、
空间/有限值/奖励求和验证；失败返回非零退出码。raise 模式和严格掩码另有真实仿真测试。
不修改第三方 checker、不筛选有利 seed，也不将依赖缺失跳过计为通过。

本阶段实际运行全部 **94 项测试通过，0 失败、0 跳过**，耗时 6.425 秒；保留旧 56 项，
新增奖励纯函数 15 项、正式环境 23 项。`validate_env.py --episodes 3 --seed 0`
四项检查通过，退出码 0；正式 Gym checker 使用默认 hold 模式，未绕过或修改 checker。
stock、joint-debug、observation-debug 三回合回归均通过，结果与历史一致。
project 无窗口 seed 0/1/2 各为 200 步、40 秒，累计项目奖励分别为
21.152381 / 19.001978 / 18.582316，非法回退均为 0。
project human seed 0 完整 200 步、40 秒、退出码 0，累计项目奖励 21.152381。
保存并查看了 `docs/demo_frames/project_t0.png` 和 `project_t2.png` 两张真实
rgb_array 渲染帧（600×150），可见自车换道；它们不是屏幕截图或视频。
以上奖励为随机动作验收输出，不是模型性能。

本阶段实际命令、返回码、checker 警告、图形验证范围、媒体文件和解压复验记录
统一写入 [environment_validation.md](environment_validation.md)。
当直接创建的 Gym Env 没有注册 spec 时，checker 不能自动重建所有渲染模式，
因此其相关提示需保留，并单独记录真实 human/rgb_array 验证，不把跳过自动渲染检查算通过。

提交包 `dist/ME5418_Group25_Gym_Environment.zip` 只收环境所需源码、测试、配置、README、
依赖与验证说明及可用演示帧；排除个人路径设置、缓存、虚拟环境、`.git`、课程 PPT、
无关附件、模型和大型日志。运行命令使用项目相对路径，不依赖原开发机绝对路径。
解压后必须在临时目录用现有解释器实际重跑测试和无窗口演示；只有通过才称解压复现通过。
这不是全新 Ubuntu/全新依赖安装验证，图形静态帧也不是视频。

### 控制限制、复用与后续工作

本轮保持上节诊断：`a=(target_speed-speed)/0.6`，无常规固定加速度上下限，
存在转向限幅、速度越界修正和碰撞制动。目标 13.90、实际 25 m/s 的三帧响应仍为
`25 → 23.766667 → 22.670370 → 21.695885`，没有改小动力学响应来隐藏强制动。
**训练与行为解释前确认底层控制限制，并确保 DQN/基线一致。**
现有工程不能证明真实驾驶安全性。

HighwayEnv 提供道路、交通行为、底层控制、动力学、碰撞与渲染；本项目实现九动作适配、
结构掩码、34 维观测/100 米感知、项目奖励、Gym 整合及验收工具。
代码、测试与文档使用 AI 辅助生成及审查；团队仍须理解、人工核对并按课程要求披露。
没有使用现成训练库替代计划自行实现的 DQN，未启动训练或正式性能评测。

尚待完成：学习模块、经验回放、gap-based 基线、训练、基线比较、泛化实验、奖励调参、
训练随机化范围、控制限幅与暂定观测尺度/换道容差的适用性验证。
Lecture 6b 原文、Canvas 提交格式、原创代码比例、视频及报告要求尚未完整核对；
不能保证此环境包单独满足全部评分要求。

## 阶段1B-3完成边界（历史记录）

阶段 1B-3 已实现项目奖励纯函数和正式 34 维 / 九动作 Gym 环境；
stock、joint-debug、observation-debug、既有控制/观测及原有测试保留。
本轮验收和提交包的实际结果以 environment_validation.md 为准，不把未执行项写为通过。
DQN、经验回放、基线控制器、训练、正式性能评测和泛化实验仍未实现。
本轮没有安装依赖、修改第三方源码或系统环境，环境验收与提交材料完成后停止。


## 当前阶段2完成边界

Q网络、20项新增测试、CPU state_dict保存加载、真实正式环境输入对接已实现并验证；
与原94项合计114项通过、0失败、0跳过。配置读取34→128→128→9，ReLU和bias，22153参数。
网络只计算原始Q值，不承担掩码、动作选择或环境推进。CPU PyTorch2.14.1+cpu已按授权安装，
原环境源码、控制器、观测、奖励、旧测试和依赖版本保持不变。
详见 [网络设计](network_design.md) 与 [进度记录](progress.md)。
**网络已验证，但DQN和驾驶训练尚未完成。** 不继续实现经验回放或启动正式训练。

交付整理说明：历史个人绝对路径已改成可移植表示；`${CONDA_PREFIX}` 指当时实际验证的 `me5418-highway` 环境路径。历史源码和实验结论未因此改动。
