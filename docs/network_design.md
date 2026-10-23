# 阶段 2：PyTorch Q 网络设计与验证

日期：2026-10-23。当前完成 **Q 网络、数值测试、state_dict 保存加载及正式环境输入对接**。
**网络已验证，但 DQN 和驾驶训练尚未完成。** 没有实现经验回放、探索、目标网络、TD 更新或基线。
本轮实际资料是工作区 config、观测/环境源码、测试及既有文档；没有重新读取外部 proposal、
课件或其他 ChatGPT 对话附件。网络规格沿用当前九动作、34 维配置。

## 输入、输出与职责

`DQNNetwork` 继承 `torch.nn.Module`，尺寸读取 `config.py`。
模型默认使用 CPU / float32，输入为同 dtype/device 的 `torch.Tensor`：

| 输入 | 输出 | 含义 |
| --- | --- | --- |
| `[34]` | `[9]` | 单条观测对应九个原始 Q 值 |
| `[B,34]`，B≥1 | `[B,9]` | 批量观测各自对应九个原始 Q 值 |

观测固定顺序：0–17 是三条车道各最近可见前/后车的 presence、bumper gap、relative speed；
18–24 是七个自车/控制特征；25–33 是上一实际联合动作的 one-hot，reset 时全零。
100 米过滤、归一化、缺失编码继续由既有观测模块完成，网络不再次归一化、不补齐数据。
完整特征表见 [项目规格](project_spec.md) 和 [README](../README.md)。

输出索引固定为：

| 编号 | 动作 | 编号 | 动作 | 编号 | 动作 |
| --- | --- | --- | --- | --- | --- |
| 0 | Left + slower | 3 | Keep + slower | 6 | Right + slower |
| 1 | Left + hold | 4 | Keep + hold | 7 | Right + hold |
| 2 | Left + faster | 5 | Keep + faster | 8 | Right + faster |

这些输出是未来经 DQN 学习后的动作价值估计，当前随机初始化输出没有驾驶策略含义。
不是概率分布，不做 Softmax/Sigmoid，也不在网络内应用动作掩码或挑选动作。
未来 Learning Agent 应在探索、贪心选择和 TD 目标中使用环境掩码，属于后续任务。

调用方将 NumPy 观测显式转换为张量：`torch.from_numpy(observation)`。
forward 只做输入检查及 Linear/ReLU，不修改共享 NumPy 存储，不 detach、转 NumPy 或搬至 CPU。
非 Tensor、错误维数、空批量、dtype/device 不匹配分别明确报错，不自动猜测或转换。
显式 `model.double()` 后可使用 float64 输入，梯度保留；本阶段实际运行/加载均在 CPU。
有限环境输入得到有限输出；错误物理 NaN/Inf 已由观测模块拒绝。

## 网络结构与独立参数核对

```text
Linear(34, 128, bias=True)
ReLU
Linear(128, 128, bias=True)
ReLU
Linear(128, 9, bias=True)
```

| 层 | 权重 | bias | 合计 |
| --- | ---: | ---: | ---: |
| 第一层 | 34×128=4352 | 128 | 4480 |
| 第二层 | 128×128=16384 | 128 | 16512 |
| 输出层 | 128×9=1152 | 9 | 1161 |
| 总计 | 21888 | 265 | **22153** |

没有 CNN、LSTM、Dueling、Dropout、BatchNorm 或额外输出激活。
导入 network 不构造模型、环境或优化器。初始化使用 PyTorch Linear 的默认初始化，
种子由调用方显式管理；forward 不消费随机数或修改参数。

## CPU 依赖与实际变更

指定解释器：`${CONDA_PREFIX}/bin/python`，Python 3.11.16。
初检发现 torch 不存在；按照用户授权和 [PyTorch 官方 CPU 安装说明](https://pytorch.org/get-started/locally/)，
先用官方 CPU wheel 源 dry-run，锁住既有核心依赖并审查计划，然后仅向项目 Conda 环境安装。
未使用 sudo、系统 Python、CUDA、云资源；没有升级或删除已有依赖。

新增包（安装报告与前后 metadata 对比所得）：

| 包 | 实际版本 |
| --- | --- |
| torch | **2.14.1+cpu** |
| filelock | 3.32.3 |
| fsspec | 2026.7.0 |
| networkx | 3.6.1 |
| sympy | 1.14.0 |
| mpmath | 1.3.0 |
| Jinja2 | 3.1.6 |
| MarkupSafe | 3.0.3 |

已有包版本全部未变，包括 gymnasium 1.4.0、highway-env 1.12.1、numpy 2.4.6、
pygame-ce 2.5.8、matplotlib 3.11.2、pandas 3.0.6、farama-notifications 0.0.6。
`torch.version.cuda is None`；安装前后 `pip check` 都报告 `No broken requirements found`。
没有将 torchvision/torchaudio 作为本项目依赖。

实际安装形式（`python` 是上面的指定解释器）：

```bash
python -m pip install --dry-run --report /tmp/me5418-stage2/torch-plan.json \
  --constraint /tmp/me5418-stage2/core-constraints.txt \
  --index-url https://download.pytorch.org/whl/cpu --only-binary=:all: torch
python -m pip install --report /tmp/me5418-stage2/torch-installed.json \
  --constraint /tmp/me5418-stage2/torch-constraints.txt \
  --index-url https://download.pytorch.org/whl/cpu --only-binary=:all: 'torch==2.14.1+cpu'
```

临时约束和报告用于这次审查，不是运行所必需的文件。可移植的可选安装入口为：

```bash
python -m pip install -r requirements-network.txt
```

`requirements-network.txt` 使用 CPU 源、固定 torch 版本，并通过 `-c requirements.txt`
约束已验证的核心环境版本。原 `requirements.txt` 继续只服务环境，不强制加载 PyTorch。
新机器/新依赖环境安装未验证；已有项目环境无需再次安装。

## 演示与保存加载

在项目根目录运行：

```bash
python network_demo.py --help
python network_demo.py --seed 0 --steps 30
# 可选保留文件；若同名文件已存在会明确拒绝覆盖，请换一个未使用的路径。
python network_demo.py --seed 0 --steps 30 --checkpoint outputs/models/my_network_check.pt
```

默认检查点在临时目录中，保存/读回检查后自动清理。指定路径时使用独占创建，
不覆盖已有文件。保存的是纯 `state_dict`，使用 `weights_only=True`、`map_location='cpu'`
和 `load_state_dict(..., strict=True)` 加载，并以 eval 模式比较同一批输入。

```python
import torch
from network import DQNNetwork

model = DQNNetwork()
state = torch.load("outputs/models/network_numeric_check_seed0.pt",
                   map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()
```

本轮实际保留的 `outputs/models/network_numeric_check_seed0.pt` 是**合成数值检查参数**，
不是驾驶模型，也没有保存优化器状态。它已按既有 .gitignore 被忽略，没有提交或推送。
合成检查使用固定32条输入、目标 `0.25*x[:,:9]+0.1`、MSE、Adam lr=0.001、30步。
这些只是短小数值实验设置，不是已经确定的 DQN 训练超参数。

实际 seed=0 输出：

```text
参数量：22153
单输入 (34,) -> (9,)
批量输入 (32,34) -> (32,9)
MSE：0.085491151 -> 0.001677406
参数差 L2：2.108985424
state_dict 保存及 CPU 加载：通过；同一输入输出最大差=0
```

上述全部标注“网络数值验证，不是驾驶训练结果”。没有环境回合驱动的优化、奖励目标或 TD 更新。

## 真实环境输入集成

另建**独立随机初始化**的网络，避免把合成任务优化后的网络说成未训练初始网络。
默认演示真实调用 `make_env(render_mode=None).reset(seed=0)`，返回 float32[34]，
检查 observation_space、Discrete(9) 和结构掩码，再通过 `torch.from_numpy` 输入网络。
环境在 finally 中关闭；这个集成演示不调用 step、不执行驾驶动作。

实际得到按 0～8 排列的 Q 值：

```text
[+0.02147857, +0.00120005, -0.06696703,
 -0.19061418, -0.07547047, +0.06604509,
 +0.09047119, -0.09612814, +0.00862855]
```

掩码独立为 `[True, True, False, True, True, False, False, False, False]`，
没有用它改写九个原始 Q 值。不把这些数值解释为有效驾驶策略。
测试还检查了一次真实 step(4) 返回的项目观测输入；前向前后比较整组车辆状态、控制目标、
时间/物理步数、观测、掩码、动作历史、累计奖励及 RNG，均不发生改变。
独立子进程阻断 torch/network/学习模块导入时，正式环境仍能真实 reset/step。
若使用 `--skip-environment`，脚本明确输出环境集成“未验证”；不会用 stock 或伪造观测代替。

## 实际验证命令与结果

以下均使用指定解释器执行：

```bash
python -m unittest discover -s tests -v
python network_demo.py --seed 0 --steps 30 --checkpoint outputs/models/network_numeric_check_seed0.pt
python validate_env.py --episodes 3 --seed 0
python demo.py --mode project --render none --episodes 1 --seed 0 --reward-steps 0
```

- 开始前环境基线：94 通过、0失败、0跳过，6.216秒。
- 最终全部测试：**114 通过、0失败、0跳过**，9.722秒；原94项全部保留，新增20项。
- 新测试涵盖独立参数公式、真实原始Q、单/批量、有限梯度、输入梯度、参数更新、固定回归下降、
  CPU保存加载、错误输入、非连续张量、输入共享存储、参数/RNG只读及正式环境输入集成。
- `network_demo`：上述 loss/参数变化/CPU往返/真实环境9Q 均通过，退出码0。
- `validate_env`：正式环境 check_env、三回合掩码交互、raise拒绝、RGB/close 四组通过，退出码0。
- project 无窗口回归：seed0完整200步/40秒，累计项目奖励21.152381，terminated=False、
  truncated=True，非法回退0，与阶段1B-3一致；该值不是网络策略成绩。
- checker仍报告未注册 spec 导致不能自动重建其他渲染模式的既有警告，未屏蔽。
  RGB检查实际通过；本阶段不重跑human窗口，阶段1B-3的完整图形结果保留在原验证记录。

代码语法、仅标准库 --help 及最终文件保护检查见 [进度记录](progress.md)。
保存加载只验证 CPU；没有安装或验证 CUDA。既有环境ZIP仍是阶段1B-3快照，本轮未重打包。

## 后续边界

下一个学习模块入口是 `replay_buffer.py`，随后才是 Learning Agent、epsilon-greedy、
目标网络与 TD 更新；需要后续授权，当前未实现。继续保留 `config.py` 中的训练超参数 TODO。
训练前仍需确认底层执行器限制，统一 DQN/基线设置；随机初始化 Q 值不是驾驶安全证明。

交付整理说明：历史个人绝对路径已改成可移植表示；`${CONDA_PREFIX}` 指当时实际验证的 `me5418-highway` 环境路径。历史源码和实验结论未因此改动。
