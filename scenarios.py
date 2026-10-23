"""可复现、分区隔离的真实训练场景；仅在 reset 初始化时安排交通。

密度控制车队相邻车辆的间距；initial_spacing_m 控制最靠近自车的前/后车
位置。两者独立生效，不把只换seed称为三个场景参数都已随机化。
最终测试区域仅预留，本模块禁止在阶段3采样或运行 final-test。
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Integral, Real

import numpy as np

import config


def _integer(value, name: str, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} 必须是整数")
    value = int(value)
    if not 0 <= value < maximum:
        raise ValueError(f"{name} 必须在 [0,{maximum}) 内")
    return value


def is_withheld(parameters: dict) -> bool:
    """组合留出区域：高密度且小初始间距且快交通；不查看运行结果。"""
    bounds = config.SCENARIO_CONFIG["withheld"]
    return bool(parameters["density"] >= bounds["density_min"]
                and parameters["initial_spacing_m"] <= bounds["initial_spacing_max_m"]
                and parameters["traffic_speed_mps"] >= bounds["traffic_speed_min_mps"])


def _validate_parameters(parameters: dict) -> dict:
    ranges = {"density": "density_range", "initial_spacing_m": "initial_spacing_range_m",
              "traffic_speed_mps": "traffic_speed_range_mps"}
    if set(parameters) != set(ranges):
        raise ValueError("场景参数必须恰好包含 density、initial_spacing_m、traffic_speed_mps")
    result = {}
    for name, range_name in ranges.items():
        value = parameters[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} 必须是有限实数")
        low, high = config.SCENARIO_CONFIG[range_name]
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} 超出已冻结场景范围 [{low},{high}]")
        result[name] = float(value)
    if is_withheld(result):
        raise ValueError("该组合属于预留最终测试区域，阶段3禁止使用")
    return result


class ScenarioSampler:
    """独立 RNG + 可恢复游标；验证用 seed=0 重建即可得到相同固定场景集。"""

    def __init__(self, split: str, seed: int) -> None:
        settings = config.SCENARIO_CONFIG
        if split not in ("development", "train", "validation"):
            raise ValueError("阶段3仅允许 development/train/validation；final-test保持未使用")
        self.split = split
        self.seed = _integer(seed, "run seed", settings["run_seed_stride"])
        self.namespace = settings["seed_namespaces"][split]
        self.cursor = 0
        self.rng = np.random.default_rng(np.random.SeedSequence(
            [settings["sampling_rng_tag"], self.namespace, self.seed]))

    def next(self) -> dict:
        """先拒绝预留参数组合，再分配不重叠的场景种子；拒绝不看未来碰撞。"""
        settings = config.SCENARIO_CONFIG
        _integer(self.cursor, "scenario cursor", settings["run_seed_stride"])
        for _ in range(settings["max_parameter_draws"]):
            parameters = {
                "density": float(self.rng.uniform(*settings["density_range"])),
                "initial_spacing_m": float(self.rng.uniform(*settings["initial_spacing_range_m"])),
                "traffic_speed_mps": float(self.rng.uniform(*settings["traffic_speed_range_mps"])),
            }
            if not is_withheld(parameters):
                break
        else:
            raise RuntimeError("未能在固定上限内采到非留出场景；禁止无限重试")
        env_seed = (self.namespace * settings["namespace_stride"]
                    + self.seed * settings["run_seed_stride"] + self.cursor)
        scenario = {
            "version": settings["version"], "split": self.split,
            "namespace": self.namespace, "run_seed": self.seed,
            "cursor": self.cursor, "env_seed": env_seed,
            "scenario_id": f"{self.split}/{self.seed}/{self.cursor}",
            "parameters": parameters,
        }
        self.cursor += 1
        return scenario

    def state_dict(self) -> dict:
        return {"version": config.SCENARIO_CONFIG["version"], "split": self.split,
                "seed": self.seed, "cursor": self.cursor,
                "rng_state": deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state: dict) -> None:
        """只恢复同版本同分区同run；先验证副本，失败时不更改当前状态。"""
        if (state["version"] != config.SCENARIO_CONFIG["version"]
                or state["split"] != self.split or state["seed"] != self.seed):
            raise ValueError("场景采样器版本、split或run seed不兼容")
        cursor = _integer(state["cursor"], "scenario cursor",
                          config.SCENARIO_CONFIG["run_seed_stride"])
        replacement = np.random.default_rng()
        replacement.bit_generator.state = deepcopy(state["rng_state"])
        self.rng, self.cursor = replacement, cursor


def _validate_scenario(scenario: dict) -> dict:
    settings = config.SCENARIO_CONFIG
    split = scenario["split"]
    if split not in ("development", "train", "validation"):
        raise ValueError("阶段3不运行final-test场景")
    seed = _integer(scenario["run_seed"], "run seed", settings["run_seed_stride"])
    cursor = _integer(scenario["cursor"], "scenario cursor", settings["run_seed_stride"])
    namespace = settings["seed_namespaces"][split]
    expected = namespace * settings["namespace_stride"] + seed * settings["run_seed_stride"] + cursor
    if (scenario["version"] != settings["version"] or scenario["namespace"] != namespace
            or scenario["env_seed"] != expected
            or scenario["scenario_id"] != f"{split}/{seed}/{cursor}"):
        raise ValueError("场景版本、ID或种子命名空间不一致")
    return _validate_parameters(scenario["parameters"])


def _initial_summary(raw, parameters: dict) -> dict:
    """记录实际物理状态，不以奖励或后续是否碰撞判断初始有效性。"""
    reference = raw.road.network.get_lane(raw.vehicle.lane_index)
    ego_s = reference.local_coordinates(raw.vehicle.position)[0]
    rows, lane_groups = [], {lane: [] for lane in range(config.LANE_COUNT)}
    for index, vehicle in enumerate(raw.road.vehicles):
        s = float(reference.local_coordinates(vehicle.position)[0])
        lane = int(vehicle.lane_index[2])
        row = {"vehicle_index": index, "is_ego": vehicle is raw.vehicle,
               "lane": lane, "longitudinal_position_m": s,
               "relative_longitudinal_position_m": s - ego_s,
               "position_m": vehicle.position.tolist(), "speed_mps": float(vehicle.speed),
               "target_speed_mps": float(vehicle.target_speed)}
        rows.append(row)
        lane_groups[lane].append((s, float(vehicle.LENGTH), vehicle))
    gaps = []
    for vehicles in lane_groups.values():
        vehicles.sort(key=lambda item: item[0])
        gaps.extend(b[0] - a[0] - (a[1] + b[1]) / 2
                    for a, b in zip(vehicles, vehicles[1:]))
    minimum = float(min(gaps))
    if minimum <= 0 or any(not vehicle.on_road or vehicle.crashed
                           for vehicle in raw.road.vehicles):
        raise RuntimeError("初始场景几何不合法：同车道重叠、离路或已碰撞；不重采样掩盖错误")
    other_speeds = [row["speed_mps"] for row in rows if not row["is_ego"]]
    return {"vehicle_count": len(rows), "surrounding_vehicle_count": len(rows) - 1,
            "minimum_same_lane_bumper_gap_m": minimum,
            "mean_surrounding_speed_mps": float(np.mean(other_speeds)),
            "platoon_center_spacing_nominal_m": (
                config.SCENARIO_CONFIG["platoon_base_spacing_m"] / parameters["density"]),
            "vehicles": rows}


def reset_scenario(env, scenario: dict):
    """正式reset后只设置初始化的真实周车状态，再只读构造obs，不推进仿真。

    不改变自车目标/初速/车道或执行器参数。所有方法共享相同初始有效性规则；
    密度和速度不是在step期间注入的外力。场景之后沿用原生IDM/MOBIL交通。
    """
    from project_environment import HighwayLaneEnv

    if not isinstance(env, HighwayLaneEnv):
        raise TypeError("训练场景只接受正式HighwayLaneEnv，不能使用stock或调试奖励环境")
    parameters = _validate_scenario(scenario)
    settings = config.SCENARIO_CONFIG
    _, info = env.reset(seed=scenario["env_seed"], options={"config": {
        "vehicles_count": config.INITIAL_SURROUNDING_VEHICLE_COUNT,
        "vehicles_density": parameters["density"], "initial_lane_id": None,
    }})
    raw = env.simulator.unwrapped
    ego = raw.vehicle
    others = [vehicle for vehicle in raw.road.vehicles if vehicle is not ego]
    if len(others) != config.INITIAL_SURROUNDING_VEHICLE_COUNT:
        raise RuntimeError("正式场景必须恰好有20辆周车")
    rng = np.random.default_rng(np.random.SeedSequence(
        [settings["layout_rng_tag"], scenario["namespace"], scenario["run_seed"], scenario["cursor"]]))
    others = [others[index] for index in rng.permutation(len(others))]
    start, end, _ = ego.lane_index
    reference = raw.road.network.get_lane((start, end, 0))
    ego_s = reference.local_coordinates(ego.position)[0]
    lane_counts = [0] * config.LANE_COUNT
    offsets = {(lane, side): parameters["initial_spacing_m"]
               for lane in range(config.LANE_COUNT) for side in (-1, 1)}
    interval = settings["platoon_base_spacing_m"] / parameters["density"]
    jitter = settings["platoon_spacing_jitter_fraction"]
    for index, vehicle in enumerate(others):
        lane_id = index % config.LANE_COUNT
        rank = lane_counts[lane_id]
        lane_counts[lane_id] += 1
        side = 1 if rank % 2 == 0 else -1
        if rank >= 2:
            offsets[lane_id, side] += interval * rng.uniform(1 - jitter, 1 + jitter)
        lane = raw.road.network.get_lane((start, end, lane_id))
        s = ego_s + side * offsets[lane_id, side]
        vehicle.position = lane.position(s, 0.0)
        vehicle.heading = lane.heading_at(s)
        # 仅初始化阶段同时设置周车实际速度和期望速度；后续由原生动力学推进。
        speed = parameters["traffic_speed_mps"] + rng.uniform(
            -settings["traffic_speed_jitter_mps"], settings["traffic_speed_jitter_mps"])
        vehicle.speed = float(speed)
        vehicle.on_state_update()
        vehicle.target_lane_index = vehicle.lane_index
        vehicle.target_speed = float(speed)
    summary = _initial_summary(raw, parameters)
    observation, diagnostics = env.get_observation()
    info = deepcopy(info)
    info["observation_debug"] = diagnostics
    info["action_mask"] = env.get_action_mask()
    info["joint_debug"]["after"] = env.get_debug_state()
    info["scenario"] = {**deepcopy(scenario), "initial_summary": summary}
    return observation, info
