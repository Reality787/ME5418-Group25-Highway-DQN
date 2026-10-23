"""ME5418 已确定的配置；仅依赖 Python 标准库。

配置依据用户提供的最终九动作规格，已核对最终 proposal PDF；课件尚未提供。
最终稿未确定的参数保留 TODO；阶段 1B-1/1B-2/1B-3 必需的工程选择在末尾单独标注。
"""

# 场景：三条同向直车道，一辆受控车辆，初始 20 辆周围车辆。
ROAD_LAYOUT = "straight"
TRAFFIC_DIRECTION = "same_direction"
LANE_COUNT = 3
CONTROLLED_VEHICLE_COUNT = 1
INITIAL_SURROUNDING_VEHICLE_COUNT = 20
EPISODE_DURATION_SECONDS = 40
DESIRED_SPEED_MPS = 27.80
PERCEPTION_RANGE_METERS = 100
POLICY_FREQUENCY_HZ = 5
SIMULATION_FREQUENCY_HZ = 15

# 碰撞或离路是真实终止；时间上限属于截断。
TERMINATION_EVENTS = ("collision", "off_road")
TRUNCATION_EVENTS = ("time_limit",)

# 观测：交通特征 + 自车/控制特征 + 上一联合动作 one-hot。
TRAFFIC_FEATURE_DIM = 18
EGO_CONTROL_FEATURE_DIM = 7
PREVIOUS_ACTION_FEATURE_DIM = 9
OBSERVATION_DIM = 34

# 联合动作编号必须保持固定，禁止改为旧版五动作映射。
ACTION_COUNT = 9
ACTION_NAMES = {
    0: "Left + slower",
    1: "Left + hold",
    2: "Left + faster",
    3: "Keep + slower",
    4: "Keep + hold",
    5: "Keep + faster",
    6: "Right + slower",
    7: "Right + hold",
    8: "Right + faster",
}
TARGET_SPEEDS_MPS = (0.00, 6.95, 13.90, 20.85, 27.80)
SPEED_INDEX_DELTAS = {"slower": -1, "hold": 0, "faster": 1}
# Keep 保留已有目标车道，hold 保留已有目标速度；不是重置为当前实测状态。

# 阶段 2 已实现 PyTorch Q 网络 34 → 128 → 128 → 9；完整 DQN 学习器尚未实现。
NETWORK_INPUT_DIM = OBSERVATION_DIM
HIDDEN_LAYER_SIZES = (128, 128)
NETWORK_OUTPUT_DIM = ACTION_COUNT
HIDDEN_ACTIVATION = "ReLU"
GAMMA = 0.99

# 观测排列、归一化、缺失编码和动作历史初始化见下方阶段 1B-2 配置。
# TODO：最终场景初始自车状态、目标状态及随机化范围；调试初始化规则见下方。
# 项目奖励已有本轮用户指定的初始工程公式；后续需验证权重，不称为最优。
# 阶段3训练工程起点见DQN_CONFIG/TRAINING_CONFIG；正式预算由pilot后冻结协议指定。
# 阶段3使用线性epsilon、Adam、SmoothL1Loss和按梯度次数硬同步。
# 阶段3时间截断保留最后观测并bootstrap；真实终止不bootstrap。
# TODO：基线规则阈值、评测指标具体公式、场景生成和随机种子取值。
# proposal 中的评测规模与主要指标已记录在 docs/project_spec.md，仅供后续规划。

# 阶段 1B-1 工程选择：以下值暂定，待验证，不来自 proposal。
# 中心误差 0.10 m 约为原生 4 m 车道宽度的 2.5%；航向误差 0.02 rad
# 约为 1.15 度，避免车辆刚跨入目标车道或斜着经过中心时过早解锁。
LANE_CHANGE_LATERAL_TOLERANCE_M = 0.10
LANE_CHANGE_HEADING_TOLERANCE_RAD = 0.02
# 仅 reset 时选最接近实际初速的目标档，等距时取较低档，以减小初始控制突变。
# 显式 initial_target_speed_index（包含 0）优先；step 永远基于已有目标索引。
JOINT_INITIAL_TARGET_SPEED_RULE = "nearest_actual_speed_ties_lower"

# 阶段 1B-2 工程选择：固定归一化，不来自 proposal，不用回合统计量拟合。
# 纵向速度保留符号，40m/s 对应库名义速度边界；27.80～40 的实际超速仍可区分。
# 横向10m/s 为宽松固定尺度；相对40m/s 表示快慢差，超出后明确报告裁剪。
# 横向偏移2m 为原生4m车道的半宽；道路法向朝车道编号增大的方向为正。
OBS_LONGITUDINAL_SPEED_SCALE_MPS = 40.0
OBS_LATERAL_SPEED_SCALE_MPS = 10.0
OBS_RELATIVE_SPEED_SCALE_MPS = 40.0
OBS_LATERAL_OFFSET_SCALE_M = 2.0
OBS_GAP_SCALE_M = float(PERCEPTION_RANGE_METERS)
OBS_LANE_SCALE = float(LANE_COUNT - 1)
OBS_TARGET_SPEED_SCALE_MPS = DESIRED_SPEED_MPS
OBS_SIGNED_BOUNDS = (-1.0, 1.0)
OBS_UNSIGNED_BOUNDS = (0.0, 1.0)
# 下列顺序固定为实际车道0/1/2，不能随自车移动重排。
OBSERVATION_FEATURE_NAMES = tuple(
    f"lane_{lane}_{side}_{feature}"
    for lane in range(LANE_COUNT)
    for side in ("front", "rear")
    for feature in ("presence", "bumper_gap", "relative_speed")
) + (
    "ego_longitudinal_speed", "ego_lateral_speed", "ego_lateral_offset",
    "current_lane", "target_lane", "target_speed", "lane_change_in_progress",
) + tuple(f"previous_action_{action}" for action in range(ACTION_COUNT))
OBSERVATION_SCALES = (
    (1.0, OBS_GAP_SCALE_M, OBS_RELATIVE_SPEED_SCALE_MPS) * 6
    + (OBS_LONGITUDINAL_SPEED_SCALE_MPS, OBS_LATERAL_SPEED_SCALE_MPS,
       OBS_LATERAL_OFFSET_SCALE_M, OBS_LANE_SCALE, OBS_LANE_SCALE,
       OBS_TARGET_SPEED_SCALE_MPS, 1.0)
    + (1.0,) * ACTION_COUNT
)
OBSERVATION_LOW = tuple(
    OBS_SIGNED_BOUNDS[0] if index in (2, 5, 8, 11, 14, 17, 18, 19, 20)
    else OBS_UNSIGNED_BOUNDS[0] for index in range(OBSERVATION_DIM)
)
OBSERVATION_HIGH = (OBS_UNSIGNED_BOUNDS[1],) * OBSERVATION_DIM
OBSERVATION_UNITS = (
    ("0/1", "m", "m/s") * 6
    + ("m/s", "m/s", "m", "lane_id", "lane_id", "m/s", "0/1")
    + ("0/1",) * ACTION_COUNT
)
# 缺失交通槽为(0,1,0)，raw gap=100是缺失哨兵，不代表测到了车。
# reset 上一动作九维全0；成功 step 后记录外部联合动作的 one-hot。
# TODO：训练前确认底层控制限制，并确保 DQN/基线一致；本轮不修改执行器。

# 阶段 1B-3：本轮用户指定的初始工程奖励配置，暂定，非 proposal 原文/最优参数。
REWARD_PROGRESS_WEIGHT = 1.0
REWARD_HEADWAY_WEIGHT = 2.0
REWARD_FAILURE_COST = 10.0
REWARD_LANE_CHANGE_COST = 0.05
REWARD_TARGET_SPEED_COST = 0.02
REWARD_THW_THRESHOLD_S = 1.5
REWARD_MIN_SPEED_MPS = 0.1
# 连续项乘实际dt；失败/目标变化是单次事件，不添加存活或加速度/jerk项。

# 阶段3：用户指定的普通DQN工程起点，非proposal确定超参数；正式运行前冻结。
DQN_CONFIG = {
    "optimizer": "Adam", "learning_rate": 1e-4, "batch_size": 64,
    "replay_capacity": 100000, "learning_starts": 1000, "train_every": 4,
    "updates_per_trigger": 1, "loss": "SmoothL1Loss", "gradient_norm_clip": 10.0,
    "target_update_every": 1000, "gamma": GAMMA,
    "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay_fraction": 0.60,
}
TRAINING_CONFIG = {
    "device": "cpu", "torch_threads": 1, "validation_every": 5000,
    "validation_episodes": 6, "validation_seed": 0,
    "log_every_updates": 100, "checkpoint_every": 5000, "progress_every": 1000,
    "smoke_steps": 2000, "pilot_steps": 20000,
    "formal_budget_candidates": (50000, 100000, 200000),
    "formal_seeds": (11, 22, 33), "smoke_seed": 101, "pilot_seed": 202,
    "validation_failure_weight": 2.0, "progress_denominator_m": 1112.0,
    "stopped_speed_threshold_mps": 1.0,
}
# CPU无渲染单环境；预算是环境决策步，不是15Hz物理帧或梯度次数。
# terminated停止bootstrap；truncated保留最后观测/掩码并继续bootstrap。

# 阶段3场景工程起点：范围不是proposal指定，正式训练前冻结。
SCENARIO_CONFIG = {
    "version": "scenario-v1-density-headway-speed",
    "density_range": (0.8, 1.2),
    "initial_spacing_range_m": (20.0, 35.0),
    "traffic_speed_range_mps": (18.0, 28.0),
    "traffic_speed_jitter_mps": 2.0,
    "platoon_base_spacing_m": 35.0,
    "platoon_spacing_jitter_fraction": 0.1,
    "withheld": {"density_min": 1.1, "initial_spacing_max_m": 23.0,
                 "traffic_speed_min_mps": 26.0},
    # 仅声明最终测试区域，本阶段生成器拒绝final-test，不能用于选范围或选模。
    "ood_density_range": (1.25, 1.4),
    "ood_traffic_speed_range_mps": (29.0, 32.0),
    "seed_namespaces": {"development": 1, "train": 2, "validation": 3,
                        "final-test": 4},
    "namespace_stride": 1_000_000_000_000,
    "run_seed_stride": 1_000_000,
    "layout_rng_tag": 71_903,
    "sampling_rng_tag": 81_907,
    "max_parameter_draws": 10000,
}
