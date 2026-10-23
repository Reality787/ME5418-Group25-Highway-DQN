# 阶段3训练协议

本轮使用正式HighwayLaneEnv：34维float32、九动作、100米感知、项目奖励。
无渲染、CPU单训练环境；验证在独立环境顺序执行。不得使用stock或原生调试奖励训练。
所有工程范围/超参数不是proposal原文或性能保证。最终测试保持未执行。

## 预先指定的学习方案

普通DQN，online与独立target都是34→128→128→9，ReLU、bias、22153参数。
Adam lr=1e-4；batch64；均匀回放100000；1000环境步开始学习，每4环境步一次更新；
SmoothL1Loss；梯度范数裁剪10；每1000次梯度更新硬复制target；gamma=.99。
epsilon从1线性到.05，在每run计划预算的前60%完成下降，之后保持。
每一步是0.2秒决策，通常3个1/15秒物理更新；梯度次数单独记录。
超参数集中在config.DQN_CONFIG/TRAINING_CONFIG，正式训练不可在途中修改。

真实终止 y=r；其余（含时间截断）y=r+.99*max_legal Q_target(next_obs)。
使用当前step返回的最终obs/mask，然后才reset。全terminal批次不访问target，不以0*(-inf)屏蔽。
探索/贪心/TD均使用结构掩码；严格环境raise，实际执行不一致或hold回退立即停止。

## 真实场景随机化与分区（训练前定义）

density均匀[.8,1.2]；initial_spacing_m均匀[20,35]；traffic_speed_mps均匀[18,28]。
每车初始/目标速度为速度中心±2m/s均匀扰动。参数逐回合实际写入交通初始化。
密度控制后续车队中心间距35/density米，再乘[.9,1.1]扰动；初始间距控制每道最近前后车中心距离。
两参数作用不同，保险杠间距须减车身半长。20周车随机顺序分为7/7/6，前后交错。
固定自车25m/s原生初速、最近目标档、三车道、40秒。初始化后沿用原生IDM/MOBIL交通动力学。
只按初始几何检查同道无车身重叠、在路上、未碰撞，不看未来结果筛除场景。
每回合scenarios.jsonl保留采样值、ID/seed、实际21车位置速度及间距摘要。

命名空间development=1、train=2、validation=3、final-test=4。
env_seed=namespace*10^12+run_seed*10^6+cursor，run_seed/cursor各限制[0,10^6)。
训练/开发/验证的RNG和种子严格隔离；每次验证固定validation seed0的前6个场景。
Development/train/validation全部排除组合区域：density>=1.1且spacing<=23且speed>=26。
预留最终测试分组：ID非留出、上述withheld组合、OOD密度[1.25,1.4]（spacing27.5/speed23固定）、
OOD速度中心[29,32]（density1/spacing27.5固定）。生成器本阶段拒绝final-test，未用于调参选模。
范围是基于本地真实几何/速度检查的初始工程选择，未做无上限搜索。

## 验证与选模（训练前指定）

每5000环境步及初始/最终步独立验证6回合：epsilon0/no_grad，不进回放，不更新网络，
不改变训练场景、探索/回放RNG。验证结果既用于选模也可用于pilot有限开发，不是最终泛化证据。
所有失败回合保留；进度为沿路纵向距离/1112米，分母不因早失败变小。
completion仅无失败运行到40秒；记录collision/offroad/failure/completion率、距离、进度、reward，
停车时长比例（步后纵速<1m/s）、换道发起/调速次数，以及合法最大Q值。
selection score = mean_normalized_progress - 2*failure_rate；同分选较低collision_rate，再较早checkpoint。
初始step0同样参加预先约定的选模，不能隐瞒模型没有改善。score不是驾驶安全保证。

## 分级预算与冻结

先smoke seed101、2000环境步，再pilot seed202、20000环境步，均属于development。
正式候选预算为每seed50000/100000/200000环境步。根据pilot记录选择统一预算B后，
写出机器可校验frozen_protocol.json，记录原因、环境/观测/奖励/动力学、全部配置、依赖版本和源码hash。
formal仅允许seed11/22/33按同一冻结协议从头初始化网络、回放、优化器和随机数。
本段不提前声称pilot/formal已完成；执行状态见progress.md。

## 底层控制版本

复核HighwayEnv1.12.1本地源码：a=(target-speed)/.6，TAU_ACC=.6、KP_A=1/.6。
每1/15秒重算，目标13.9/实际25时三帧为23.7666667→22.6703704→21.6958848。
常规无固定加速度限幅；speed越过±40才约束加速度，crashed时转向0、a=-speed；
转向限幅±pi/3，航向参考偏差±pi/4。TAU_HEADING=.2、TAU_LATERAL=.6、TAU_PURSUIT=.1。
第一帧-18.5m/s²是当前模型强响应，作为已知建模限制保留，不称为真实驾驶安全/舒适控制。
本轮冻结这一实际版本，不静默改物理参数。若将来修改限幅，需新环境版本、新run并与基线统一。

## checkpoint与恢复语义

best.pt/last.pt是inference checkpoint：online权重、必要配置、兼容签名及指标。
resume.pt是最近完整回合边界：online/target/optimizer/replay、各RNG、计数、epsilon预算进度、
场景采样器游标、选模记录、下次验证步。每经过5000步后的首个回合边界建立恢复点，
运行首尾恰在边界或显式边界停止时也保存。CPU weights_only加载；只加载可信自产文件。
加载严格检查34维顺序/归一化、九动作、奖励、动力学、源代码hash和依赖版本。

预算结束或程序能捕获中断时，last.pt尽量保留当前推理权重；resume.pt仍是最近完整检查点。
若进程被强制结束或原会话不可访问，异常处理可能未执行，只能以磁盘实际检查点为准。
没有序列化完整中途仿真状态，因此不能从last.pt宣称逐步精确续训。
--resume会回退到最近完整边界；旧日志尾段保留、以segment和env_steps标识重新执行记录。
连续/边界恢复一致性必须有真实测试，不能仅凭文件存在判断。

## 本轮实际冻结记录

冻结时间（UTC）：`2026-10-23T15:32:44.478178+00:00`。统一正式预算 **50,000 环境步/seed**，种子 **11、22、33**。

pilot在20000环境步完成4751次更新，数值有限；固定验证score从-1.921723提高到最佳0.645586（step15000）。最终进度0.642293、失败率0.000000、停车比例0.000000、平均调速14.667次。采用候选中最小的50000步作为本轮有界比较预算，是pilot的2.5倍；已有可学习信号且没有数值/接口错误，但少量固定验证场景不足以证明收敛或为100000/200000成本提供充分依据。三种子同预算从头运行，保留反复调速等行为问题，不据此声称最优或安全；不调奖励、不读取最终测试。

兼容签名：`c98a0123f46fa245cbaf1a9a6140dd2ae99b6ca916f8141047bc4cc746657989`。
机器协议位于 `outputs/training/stage3/frozen_protocol.json`，14个运行源码和依赖清单保存在同级 `frozen_source/`。三个初始online权重SHA256均不同；每个正式run重新创建空回放与Adam状态，不加载pilot参数。

## 执行中断记录

正式seed11/22各完成50000步；seed33日志至少19396步，恢复点15178步。统一50000步协议保持不变，未运行最终测试。
恢复命令、准确可恢复范围及未完成曲线见[progress.md](progress.md)。三个模型尚未统一冻结导出。
