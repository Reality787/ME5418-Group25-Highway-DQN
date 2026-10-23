# M1/M2 协作约定

当前开发分支为 `m1-m2-authoritative`，M1 Environment 和 M2 QNetwork 是权威来源。原模块保持 ZIP 原件，新增代码仅提供入口与验收；不要重新引入旧默认环境、奖励或归一化。

修改前检查 Git 状态并阅读 README、来源清单和权威整合记录。若确需修改原模块，记录源码差异、理由和实际测试；不得仅为让测试通过而改预期。队友原文档和AI披露保留，成员分工由本人如实补充。

验收命令为 `python -m unittest discover -s tests -v`、`python verify.py`、`python nn_demo.py --pool-size 200 --batch-size 32 --fit-steps 10`。后两项使用新输出位置；数值优化不是驾驶训练。缺pytest时原pytest套件记为未验证，不安装伪造测试替代品。

旧完整DQN/训练/评测/交付代码在历史分支和项目外归档中，不是当前运行依赖。将来开发Learning Agent需要新的明确任务、M1/M2接口协议和新训练，不默认恢复旧50,000步实验。任何正式训练、依赖变更或最终评测需单独明确范围。

不上传模型、运行日志、个人配置、凭据、课程提交PDF或历史ZIP；提交前查看diff和暂存文件。通过新分支PR审查，不force push或自动合并master。原PR #1的默认环境与此方案不同，不应直接合并替代本分支。
