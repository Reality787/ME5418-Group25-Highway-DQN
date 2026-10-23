# ME5418项目进度：阶段3

本页依据实际status.json快照更新。代码完成、测试通过、训练完成、模型冻结分别记录。
正式环境/网络与此前所有114项测试保留；阶段2历史结果见network_design.md，环境见environment_validation.md。

| 验收事项 | 实际状态 |
| --- | --- |
| DQN/回放/训练/验证/恢复代码 | 已完成，普通DQN，无SB3/Double/Dueling/PER |
| 全部测试 | 167通过、0失败、0跳过，41.276秒 |
| 真实边界恢复 | 400步、97更新；连续与恢复最终权重及共同边界完整状态一致 |
| 环境输入/奖励 | 正式34维/九动作/掩码/项目奖励，非stock或调试奖励 |
| 最终测试 | 未运行；final-test命名空间被本阶段生成器拒绝 |
| 正式协议 | 已冻结，见frozen_protocol.json |
| 三个模型冻结 | 尚未完成三模型冻结 |

| 阶段/种子 | 环境步 | 梯度更新 | 实际状态 |
| --- | ---: | ---: | --- |
| smoke seed 101 | 2000/2000 | 251 | completed |
| pilot seed 202 | 20000/20000 | 4751 | completed |
| formal seed 11 | 50000/50000 | 12251 | completed |
| formal seed 22 | 50000/50000 | 12251 | completed |
| formal seed 33 | 至少19396/50000（日志） | 至少4600 | 会话中断，未完成 |

## 固定验证集选出的checkpoint（不是最终测试性能）

每次均为validation命名空间seed0的固定6个场景。失败回合保留、进度除1112；score不是安全保证。

| run/seed | 选择步数 | collision | off-road | failure | completion | 平均距离m | 平均进度 | 停车比例 | score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke/101 | 2000 | 0.500000 | 0.000000 | 0.500000 | 0.500000 | 235.908 | 0.212148 | 0.156709 | -0.787852 |
| pilot/202 | 15000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 717.892 | 0.645586 | 0.000000 | 0.645586 |
| formal/11 | 15000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 737.461 | 0.663184 | 0.000000 | 0.663184 |
| formal/22 | 10000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 736.123 | 0.661981 | 0.000000 | 0.661981 |
| formal/33（当前最佳，预算未完成） | 15000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 733.561 | 0.659677 | 0.000000 | 0.659677 |

## 文件与运行记录

实现agent.py、replay_buffer.py、train.py；新增scenarios.py、metrics.py、validation.py、checkpoints.py、plot_training.py和五份测试。
config.py集中工程参数；原环境/观测/联合控制/奖励/network.py与所有旧测试保持原样。
未升级依赖、未修改系统Python或第三方源码、未安装CUDA、未提交/推送Git。

每个outputs/training/stage3/<mode>_seed<seed>/目录包含：config.json、status.json、scenarios.jsonl、
episodes.jsonl、updates.jsonl、validation.jsonl、best.pt、last.pt、resume.pt。
config包含解释器/依赖版本、特征/动作/归一化/奖励/控制配置、项目与库源码SHA256。
若恢复，另有events.jsonl与segment标识，旧日志尾段保留为审计记录。

best/last是推理checkpoint；resume只代表最近完整回合边界。预算中途结束时，
status.last_resume_env_steps可能小于已训练步数，不能从last假装逐步精确恢复。
完整恢复命令由各status.resume_command记录，正式运行必须同时提供冻结协议。

当前底层强减速仍为a=(target-speed)/.6，无常规固定加速度限幅，未修改以隐藏问题。
尚未完成：基线比较、预留最终测试、泛化/收敛证明；不保证DQN优于规则方法。
详细设计与范围见[dqn_design.md](dqn_design.md)和[training_protocol.md](training_protocol.md)。


## 执行中断后的磁盘核对

seed11、22各完成50,000环境步、12,251次更新。seed33预算未完成：
status最后快照19,000步，updates日志最后记录19,396步/4,600次更新；实际未落盘的尾部无法确认。
原会话句柄不可访问，当前执行环境未找到对应进程；本轮没有重新启动训练。
原status的running字段是历史快照，另见[中断记录](../outputs/training/stage3/interruption_record.json)。

seed33：best.pt与last.pt均为15,000步推理权重；resume.pt为15,178步/3,545次更新的完整回合边界，
已经用当前冻结兼容签名及CPU weights_only实际加载核对。不能把19,396步日志当作可恢复的模型状态。
从边界恢复还需执行34,822步，其中会重放边界之后的旧尾段；旧日志保留，恢复段由segment区分。
恢复时仍传总预算50,000，不能改成剩余步数，否则会改变epsilon方案并被兼容检查拒绝。

在项目目录中可使用以下命令恢复（本轮未执行）：

```bash
${CONDA_PREFIX}/bin/python train.py --mode formal --seed 33 --steps 50000 --run-dir "outputs/training/stage3/formal_seed33" --resume "outputs/training/stage3/formal_seed33/resume.pt" --protocol "outputs/training/stage3/frozen_protocol.json"
```

![实际正式曲线，seed33未完成](../outputs/training/stage3/formal_curves_partial.png)
[开发曲线](../outputs/training/stage3/development_curves.png)也已保存。
曲线仅连接实际记录点；loss按每100次梯度更新记录。最终测试未运行，三个模型尚未统一冻结导出。
seed11最终验证2/6碰撞，最佳在15,000步；seed22最佳在10,000步。固定六场景验证不能证明收敛或安全。

实际新增/修改包括agent.py、replay_buffer.py、train.py、config.py，scenarios.py、metrics.py、validation.py、
checkpoints.py、plot_training.py，五份新测试、README、DQN设计/训练协议/本进度文档和输出忽略规则。
此前167项全套检查通过，0失败/0跳过；没有因中断重新声称三种子训练完成。

## 2026-10-24：Learning Agent C&R 交付整理

源码与独立解压包的 167 项测试均全部通过；1004 步真实 smoke 完成 2 次更新，CPU 保存重载通过。两页英文报告、实际 Conda environment.yml 和提交包构建工具已准备。14 个冻结运行文件没有改动。详细结果见 [交付核验](delivery_validation.md)；最终 ZIP 记录对应 Git commit。此次仅做短流程检查，未续训 seed33、未运行最终测试；历史训练状态保持上述记录。
