# ME5418 Group 25：M1 Environment 与 M2 Neural Network

当前项目以队友提交的 **M1 原始环境和 M2 原始网络** 为准。两个模块的核心 Python 源码、原演示和原测试直接采用 ZIP 原件，不再为旧 DQN checkpoint 调整归一化、奖励、动力学或网络。

组员：Zeyu Shen、Zhang Lingyue、Xiong Peiting。

当前没有完整 Learning Agent。原 Codex 环境、Stage 3 DQN/回放/训练恢复、Stage 4 基线和评测骨架、Stage 5 报告及构建脚本已撤出活动项目，并在项目外完整归档。旧正式训练结果不是当前 M1 环境下的实验结果。

## 代码与入口

| 路径 | 来源和用途 |
| --- | --- |
| `highway_m1/__init__.py`、`highway_m1/environment.py` | M1 原件；唯一正式环境、观测、奖励、联合动作、结构掩码和受限自车 |
| `highway_m2/__init__.py`、`highway_m2/network.py` | M2 原件；QNetwork、掩码工具、网络权重保存加载 |
| `environment.py` | 仅转交构造参数的 `make_env()`，不覆盖 M1 默认值 |
| `network.py` | 直接导出 M2 QNetwork 等符号，无第二套 DQNNetwork |
| `demo.py`、`verify.py` | M1 原演示/录像、Gym 检查及可复现回合验证；verify仅适配缺少可选依赖时的版本报告 |
| `nn_demo.py` | M2 原前向/反向/固定批次拟合/保存加载数值演示 |
| `environment_demo.py` | 新增的无视频依赖入口，复用 M1 原环境、策略和叠图 |
| `teammates/m1/`、`teammates/m2/` | 原说明、原依赖记录和原 pytest 测试 |
| `tests/test_active_interfaces.py` | 新增标准库验收，独立核对 M1 物理数值及 M2 数据流 |
| `teammates/SOURCE_MANIFEST.json` | 22 个采用文件与原 ZIP 的 SHA256 对照 |

## 当前接口

M1 返回 `float32[34]` 观测和 `Discrete(9)` 动作空间。特征顺序是三车道各前/后车的 presence、gap、relative_vx（18维），自车/目标状态（7维），上一实际动作 one-hot（9维）。纵速、横速和相对速度均除以 **35**；距离除以100；横偏除以当前车道半宽；车道÷2；目标速度÷27.8。保持原 M1 的裁剪与缺失编码。

动作编号依次为 Left/Keep/Right，各嵌套 Slower/Hold/Faster；4 为 Keep/Hold。目标档为 `[0,6.95,13.90,20.85,27.80]` m/s。`env.action_mask()` 和 `info['action_mask']` 的 True 表示合法。结构非法动作在 M1 中整体回退为4，并记录 requested/executed/invalid；越界动作报错。演示只选合法动作。

默认3车道、20周车、1自车，40秒、5Hz/15Hz，初速20.85 m/s。加速度采用 **[-5,+3] m/s²**，速度名义范围[0,35]。奖励使用 M1 原公式：进度不额外乘dt，碰撞/离路各-20，短时距系数-0.5，换道/调速成本-0.03/-0.01。只有这一套默认环境与奖励。

M2 是带 bias 的 `34→128→ReLU→128→ReLU→9`，22,153参数，无输出激活。原 forward 会把输入转为float32，设备和初始化沿用原 PyTorch 行为。本轮验证 CPU；原保存加载只支持网络推理，不包含完整训练恢复。

## 运行

使用现有解释器，无需重新创建或升级已有环境：

```bash
cd "/home/aileenz/文档/ChatGPT/highway"
PYTHON="/home/aileenz/miniconda3/envs/me5418-highway/bin/python"

# M1完整有限回合，无窗口，不训练。
"$PYTHON" environment_demo.py --policy scripted --episodes 1 --seed 25 --render none

# M1原验证入口：Gym检查、重复seed及RGB渲染。
"$PYTHON" verify.py --seed 25 --output /tmp/me5418-m1-verification-new.json

# M2原数值演示：真实M1观测、有限固定批次优化，不是DQN驾驶训练。
"$PYTHON" nn_demo.py --seed 25 --pool-size 200 --batch-size 32 --fit-steps 10 \
  --output /tmp/me5418-m2-numerical-new

# 标准库验收（不会导入缺少pytest的原测试文件）。
"$PYTHON" -m unittest discover -s tests -v
```

无显示环境下导出真实场景PNG：

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy "$PYTHON" environment_demo.py \
  --policy hold --episodes 1 --seed 25 --render rgb_array \
  --output /tmp/me5418-m1-frame-new.png
```

每次使用新的输出目录/文件。原 `demo.py` 录像入口保留，需要 imageio/imageio-ffmpeg；当前未安装，未验证视频导出。桌面显示可用时可运行 `environment_demo.py --render human`，本轮不把离屏渲染称为桌面窗口验证。

## 依赖与测试边界

本轮使用 Python3.11.16、torch2.14.1+cpu、gymnasium1.4.0、highway-env1.12.1、numpy2.4.6、pygame-ce2.5.8、Pillow12.3.0、matplotlib3.11.2。没有安装或升级依赖。

`requirements.txt` / `requirements-network.txt` 记录实际运行依赖；`requirements-optional.txt` 中 pytest/imageio 的版本来自队友原清单，尚未在本机验证。`environment.yml` 是精简复现配置，不代表已完成新建环境验证。

本轮结果：**22通过、0失败、0跳过**（1.065秒）；M1原Gym/可复现验证及40秒PNG演示、M2前向/反向/10次固定批次优化与保存加载均通过。队友原 pytest 测试保持原样，但当前缺 pytest，未执行、未验证；不把队友原报告的历史74/93项结果计作本轮通过。安装可选测试依赖后，可从根目录执行 `python -m pytest teammates/m1/tests teammates/m2/tests`。

## 历史与下一步

旧代码可通过 `backup/pre-m1-m2-authoritative-20261023` 分支恢复；完整工作区、含姓名修改的报告、提交材料、模型/日志、Git bundle 和哈希清单位于项目外 `../highway-legacy-archive/`。详细目录、撤下清单、M1工程选择与proposal边界见 [权威整合记录](docs/m1_m2_authoritative_integration.md)。

后续如重新开发DQN，应直接围绕 M1 的观测/奖励/执行动作和 M2 的Q输出设计回放、探索、TD及恢复协议，并重新训练；不把旧模型迁入新环境后称为同一实验。M2数值演示自带的固定TD目标和短优化仅用于模块验证，不等于当前已有完整Learning Agent。

原 PR #1 采用旧环境优先方案，不适合直接合并为本次目标。新分支供团队独立审查，不自动合并master。仓库当前Public；请团队确认课程代码公开是否合适，未擅自更改可见性。保留队友原README中的AI辅助披露；本次整理、验收与中文说明由Codex协助完成，不编造成员分工。
