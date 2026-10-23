# Contributing

本仓库用于 ME5418 小组协作。当前共享分支为 `master`；保留已有历史，使用独立功能分支和 Pull Request 讨论、检查与合并。

## 开始修改

```bash
git switch master
git pull --ff-only
git switch -c feature/short-description
```

分支名使用具体任务，例如 `feature/gap-baseline` 或 `fix/observation-boundary`。先读取 `README.md`、相关 `docs/`、现有代码和测试，再做有界修改。不要把未提交的他人工作当作可丢弃内容；不要向共享分支 force-push，也不要用 reset/clean 清理他人的文件。

## 接口与实验约定

- 正式环境维持 34 维 `float32` 观测、固定 0–8 九动作映射和项目奖励；`stock` 等诊断接口不得冒充正式项目环境。
- 掩码供探索、贪心和 TD 目标共用。真实终止不 bootstrap，时间截断使用最后观测继续 bootstrap；不能用下一回合 reset 观测替换。
- 代码采用英文命名，关键约定、公式和非显然实现添加中文注释。修改物理控制、观测顺序/归一化、奖励或训练参数时同步文档、测试和版本记录。
- 正式训练协议已冻结。改变运行源码或依赖可能使 checkpoint 兼容检查拒绝加载；需要新实验时使用新 run 目录和明确的新协议，不隐藏配置差异或覆盖旧结果。
- 任何长时间训练、依赖升级、最终测试、模型发布应作为独立任务明确说明范围。本次交付整理不自动续训或运行最终测试。

## 提交前验证

在项目环境和仓库根目录执行：

```bash
python -m unittest discover -s tests -v
python demo.py --mode project --render none --episodes 1 --seed 0
python network_demo.py --seed 0 --steps 30
```

按变更范围追加针对性检查。若缺依赖、缺显示环境或检查中断，在 PR 中写明“未执行/未验证”与原因；不能通过删除、跳过旧测试伪造全通过。训练机制的小型测试与驾驶模型性能要区分。

提交前检查 `git diff` 与 `git status`，仅暂存本次相关文件：

```bash
git add <reviewed-files>
git commit -m "Describe the concrete change"
git push -u origin feature/short-description
```

随后创建指向 `master` 的 PR。PR 说明具体问题、最终行为、实际执行的验证及局限；由组员审查后合并。不要为提高指标而悄悄丢弃失败回合或修改预留最终测试。

## 不提交的内容

不上传个人凭证、`.env`、访问令牌、Conda/虚拟环境、缓存、个人编辑器设置、模型权重、回放池、原始大型日志、视频或生成的 ZIP。课程 proposal、课件和作业页面按课程授权处理，不随意复制上传。保留必要的输出目录 `.gitkeep`；实验结果需要共享时先约定渠道与文件范围。

## 真实贡献与 AI 使用

每位成员在 PR 或团队记录中填写实际完成、审查和验证的工作，不代填身份、工作量或贡献比例。使用 AI 时如实记录用途和人工检查情况，例如辅助实现、测试设计、排错或文档编辑；AI 生成代码仍需理解并验证。

遵循课程实际公布的协作、引用和 AI 披露要求。本文件不代替课程政策，也不宣称已经取得未核实的许可。引用第三方库时清楚区分复用的仿真能力与团队实现的控制适配、观测、奖励和 DQN 核心。
