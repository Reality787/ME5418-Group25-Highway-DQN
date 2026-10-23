"""M1 原始动作/特征名称；从 environment.py 原样移出，避免网络导入仿真。"""

SPEED_TARGETS = (0.0, 6.95, 13.90, 20.85, 27.80)
LATERAL_COMMANDS = ("Left", "Keep", "Right")
SPEED_COMMANDS = ("Slower", "Hold", "Faster")
ACTION_NAMES = tuple(
    f"{lateral}/{speed}"
    for lateral in LATERAL_COMMANDS
    for speed in SPEED_COMMANDS
)
OBSERVATION_NAMES = tuple(
    f"lane_{lane}_{feature}"
    for lane in range(3)
    for feature in (
        "front_presence", "front_gap", "front_relative_vx",
        "rear_presence", "rear_gap", "rear_relative_vx",
    )
) + (
    "ego_vx", "ego_vy", "ego_lateral_offset", "ego_current_lane",
    "ego_target_lane", "ego_target_speed", "ego_manoeuvre_in_progress",
) + tuple(f"previous_action_{index}" for index in range(9))

