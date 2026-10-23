# Learning Agent C&R 与私有仓库交付核验

本记录对应 2026-10-24 的交付整理，不续训历史正式模型，不执行 final-test。
GitHub 上传目标为 `Reality787/ME5418-Group25-Highway-DQN`（Private）。
最终 ZIP 的 Git commit、生成时间、逐文件 SHA256 和实际测试摘要由
`SUBMISSION_RECORD.json` 记录，避免在提交自身中嵌入循环引用的 commit 值。

## 实际环境与执行边界

实际使用项目 Conda 环境 `me5418-highway` 的 `bin/python`，Python 3.11.16。
PyTorch 2.14.1+cpu（CUDA 为 None）、Gymnasium 1.4.0、HighwayEnv 1.12.1、
NumPy 2.4.6、pygame-ce 2.5.8、Matplotlib 3.11.2、pandas 3.0.6。
`environment.yml` 来自实际 `conda env export --no-builds`，含 32 项 Conda 包和
25 项 pip 包，移除个人 prefix。Conda 清单、pip 元数据和依赖闭包核对通过，
`python -m pip check` 返回 No broken requirements found。没有安装或升级依赖。

解压验证清空 PYTHONPATH，14 个运行模块的实际导入位置均在独立解压目录。
**在已有依赖环境中完成解压复现**；没有在全新 Conda 或全新 Ubuntu 中重新安装。
PDF 使用独立文档工具环境中的 ReportLab 生成，没有修改项目 Conda 环境。

## 实际命令与结果

命令中的 python 均由上述已验证解释器执行。

| 检查 | 结果 |
| --- | --- |
| 源码 `python -m unittest discover -s tests -v` | 167 通过、0 失败、0 跳过；44.414 秒 |
| 解压后全部 32 个 Python 文件 AST 语法检查 | 通过 |
| 解压后相同 unittest 命令 | 167 通过、0 失败、0 跳过；42.196 秒 |
| train/demo/network_demo/plot_training 的 `--help` | 全部退出码 0 |
| 两页英文 PDF | 实际 2 页；两页渲染逐页可视检查均无裁切、重叠；正文字体已嵌入 |
| ZIP CRC、清单和运行模块 SHA256 | 通过；核心文件与冻结源码相同 |

真实短演示（从全新网络/回放池/优化器开始）：

```bash
python train.py --mode smoke --device cpu --seed 101 --steps 1004 --run-dir outputs/training/learning_agent_cr_demo
```

退出码 0，status=completed；1004 环境步、3012 物理帧、2 次梯度更新、28 个完整回合。
训练段实际耗时约 24.38 秒（含初始/最终各 6 个固定场景验证，不含 Python 启动）。
`last.pt` 在 CPU 上通过兼容性检查并重载；与相同 seed 的初始参数相比，
参数变化 L2 为 **0.019947045667035052**，输出 9 个 Q 值均有限。
训练真实经验先进入 ReplayBuffer，更新发生于第 1000、1004 步。
第 1000 步日志的 loss 为 0.459471；两次更新均通过代码中的有限性检查。

```text
train mode=smoke seed=101 env_steps=1000/1004 updates=1 epsilon=0.0500 mean_recent_loss=0.459471
completed seed=101 env_steps=1004 updates=2
```

这不是有效驾驶模型：六个固定验证场景初始碰撞率 5/6，结束为 6/6；
normalized progress 从 0.077033 到 0.097853，选模 score 从 -1.589634 到 -1.902147。
不能将小样本进度变化解释为安全性或收敛。best 因此保留初始验证模型，
参数更新核验使用 last。完整 resume 边界仍在第 0 步，不能把 last 当作精确续训状态。

## 冻结运行源码与历史训练状态

14 个兼容性关键源码与 `outputs/training/stage3/frozen_source/` 逐字节一致；
兼容签名仍为 `c98a0123f46fa245cbaf1a9a6140dd2ae99b6ca916f8141047bc4cc746657989`。
本轮只整理说明、环境快照与交付工具，没有修改 DQN、控制、观测、奖励或训练逻辑。
历史 config 的阶段性 TODO 注释没有为了风格统一而改写，避免破坏源码哈希兼容。

以下为本地状态、JSONL 日志及可信 checkpoint 实际核对值：

| 运行 | 环境步 / 更新 | 可恢复边界 | 状态 |
| --- | --- | --- | --- |
| smoke seed101 | 2000 / 251 | 0 | 已完成 |
| pilot seed202 | 20000 / 4751 | 15155 | 已完成 |
| formal seed11 | 50000 / 12251 | 45808 | 已完成；best=15000 步 |
| formal seed22 | 50000 / 12251 | 45663 | 已完成；best=10000 步 |
| formal seed33 | 更新日志至少 19396 / 4600 | 15178 / 3545 更新 | 中断未完成；best/last=15000 步 |

seed33 的旧 status 最后写到 running/19000 步，并不代表当前有进程在训练。
seed11 最后 6 个验证场景有 2 次碰撞，历史选模保留更早模型。所有历史模型、
日志、曲线和冻结协议留在原 outputs 中，没有删除、覆盖或打包上传。
三模型统一冻结、gap-based 对照、最终测试与泛化评价仍未完成。

## 文件范围与安全审计

复用项目自身 Git 根目录；原分支 master、无提交、无跟踪文件、无远程。
明确审查源码/tests/docs/环境快照/工具清单，没有发现高置信令牌、密码、私钥或
含认证信息 URL。个人绝对路径已从共享文档移除，历史原稿备份留在被忽略的
`submission/history/`。未上传课程 PDF/PPT、其他课程文件或个人编辑器配置。

忽略 outputs 训练产物、dist、submission、.vscode、虚拟环境、缓存、.env、pt/pth；
保留模型/日志/视频目录的 .gitkeep。GitHub 包含开发文档和生成工具；Canvas 包
只包含运行源码、全部 tests、依赖文件、精简 README、两页报告及源稿/提交记录。
Canvas ZIP 不包含 Git、已有 checkpoint、回放池、视频或完整训练日志。

报告由实际代码、执行记录和官方库文档支撑；不声称读取了本轮不可用的课件附件。
组员姓名、具体个人贡献与课程 AI 声明合规性仍需团队确认，未虚构人工检查经历。
