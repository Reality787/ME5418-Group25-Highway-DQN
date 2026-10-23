"""阶段 1B-1：在真实 HighwayEnv 上验证九动作和纯结构掩码。

测试只覆盖各自独立实例的场景配置；不替换物理更新、观测或奖励。
依赖缺失的情况明确记为“未验证”，不能当作仿真通过。
"""

import copy
import importlib
import math
import unittest

import config
from environment import make_joint_debug_env, make_stock_env


DEPENDENCY_ERROR = None
try:
    gym = importlib.import_module("gymnasium")
    np = importlib.import_module("numpy")
    importlib.import_module("highway_env")
    from highway_env.vehicle.kinematics import Vehicle
    from joint_actions import JointActionController
except ModuleNotFoundError as exc:
    DEPENDENCY_ERROR = f"未验证：缺少真实仿真依赖 {exc.name}"


@unittest.skipIf(DEPENDENCY_ERROR is not None, DEPENDENCY_ERROR or "")
class JointActionTests(unittest.TestCase):
    """固定测试初态，再调用真实 reset/step 检查控制效果。"""

    def make_env(self, *, speed_index=2, lane_id=1, seed=31):
        """周围车辆清零只作用于本次独立测试实例。"""
        env = make_joint_debug_env(render_mode=None)
        self.addCleanup(env.close)
        obs, info = env.reset(
            seed=seed,
            options={
                "config": {"vehicles_count": 0, "initial_lane_id": lane_id},
                "initial_target_speed_index": speed_index,
            },
        )
        self.assertEqual(len(env.unwrapped.road.vehicles), 1)
        self.assertTrue(env.observation_space.contains(obs))
        return env, info

    def assert_mask(self, env, expected):
        """同时核对布尔类型、形状和具体可行动作。"""
        mask = env.get_action_mask()
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.dtype, np.dtype(bool))
        self.assertEqual(mask.shape, (9,))
        np.testing.assert_array_equal(mask, expected)
        self.assertTrue(mask[4], "非终止控制状态必须允许 Keep + hold")

    def vehicle_snapshot(self, env):
        """捕捉可见状态，确保错误动作没有偷偷更新任何车辆或时间。"""
        raw = env.unwrapped
        return {
            "time": raw.time,
            "steps": raw.steps,
            "debug": copy.deepcopy(env.get_debug_state()),
            "vehicles": [
                {
                    "position": tuple(vehicle.position),
                    "heading": vehicle.heading,
                    "speed": vehicle.speed,
                    "lane_index": vehicle.lane_index,
                    "target_lane_index": getattr(vehicle, "target_lane_index", None),
                    "target_speed": getattr(vehicle, "target_speed", None),
                    "speed_index": getattr(vehicle, "speed_index", None),
                    "action": copy.deepcopy(vehicle.action),
                    "crashed": vehicle.crashed,
                }
                for vehicle in raw.road.vehicles
            ],
            "random_state": copy.deepcopy(raw.np_random.bit_generator.state),
        }

    def assert_snapshot_equal(self, before, after):
        # debug 的掩码是数组，先单独比较，其他字段按实际标量和容器核对。
        before_debug = before.pop("debug")
        after_debug = after.pop("debug")
        np.testing.assert_array_equal(
            before_debug.pop("action_mask"), after_debug.pop("action_mask")
        )
        self.assertEqual(before_debug, after_debug)
        self.assertEqual(before, after)

    def test_real_external_nine_actions_and_native_observation(self):
        """外部确实为九动作；内层保留真实五动作和原生观测。"""
        env, info = self.make_env()
        self.assertIsInstance(env.action_space, gym.spaces.Discrete)
        self.assertEqual(env.action_space.n, 9)
        self.assertEqual(env.unwrapped.action_space.n, 5)
        self.assertEqual(env.observation_space.shape, (5, 5))
        self.assertNotEqual(env.observation_space.shape, (config.OBSERVATION_DIM,))
        self.assert_mask(env, [True] * 9)
        np.testing.assert_array_equal(info["action_mask"], env.get_action_mask())
        result = env.step(4)
        self.assertEqual(len(result), 5)
        obs, reward, terminated, truncated, info = result
        self.assertTrue(env.observation_space.contains(obs))
        self.assertTrue(math.isfinite(reward))
        self.assertIsInstance(terminated, (bool, np.bool_))
        self.assertIsInstance(truncated, (bool, np.bool_))
        self.assertIsInstance(info, dict)
        np.testing.assert_array_equal(info["action_mask"], env.get_action_mask())

    def test_all_nine_actions_update_both_targets_once(self):
        """逐个新建中间车道/中间速度档初态，参数化验证全部九动作。"""
        for action_id in range(9):
            with self.subTest(action=action_id):
                env, _ = self.make_env()
                vehicle = env.unwrapped.vehicle
                initial_lane = vehicle.target_lane_index
                initial_position = vehicle.position.copy()
                before = env.get_debug_state()
                result = env.step(action_id)
                after = env.get_debug_state()
                lane_delta = action_id // 3 - 1
                speed_delta = action_id % 3 - 1
                self.assertEqual(
                    vehicle.target_lane_index,
                    (*initial_lane[:2], initial_lane[2] + lane_delta),
                )
                self.assertEqual(vehicle.speed_index, 2 + speed_delta)
                self.assertAlmostEqual(
                    vehicle.target_speed, config.TARGET_SPEEDS_MPS[2 + speed_delta]
                )
                self.assertAlmostEqual(after["simulation_time"] - before["simulation_time"], 0.2)
                self.assertEqual(after["physics_steps"] - before["physics_steps"], 3)
                self.assertFalse(result[2] or result[3])
                self.assertTrue(env.observation_space.contains(result[0]))
                # 物理仿真确实使车辆移动，而不是只改写一份目标字典。
                self.assertGreater(vehicle.position[0], initial_position[0])
                info = result[4]
                idle_id = env.unwrapped.action_type.actions_indexes["IDLE"]
                self.assertEqual(info["action"], action_id)
                self.assertEqual(info["native_action"], idle_id)
                self.assertEqual(info["joint_debug"]["action_id"], action_id)
                self.assertEqual(info["joint_debug"]["action_name"], config.ACTION_NAMES[action_id])
                self.assertEqual(info["joint_debug"]["native_action_id"], idle_id)

    def test_target_speed_changes_use_previous_index_not_actual_speed(self):
        """目标 20.85、实际 25 时，减速必须降到 13.90，不能按实际速度选档。"""
        for action_id, expected_index in ((3, 2), (4, 3), (5, 4)):
            with self.subTest(action=action_id):
                env, _ = self.make_env(speed_index=3)
                vehicle = env.unwrapped.vehicle
                self.assertAlmostEqual(vehicle.speed, 25.0)
                self.assertAlmostEqual(vehicle.target_speed, 20.85)
                env.step(action_id)
                self.assertEqual(vehicle.speed_index, expected_index)
                self.assertAlmostEqual(vehicle.target_speed, config.TARGET_SPEEDS_MPS[expected_index])

    def test_keep_preserves_pending_lane_target_and_adjusts_speed(self):
        """换道期间 Keep 不取消目标车道；三个纵向分量仍基于目标索引。"""
        env, _ = self.make_env()
        env.step(1)  # Left + hold：开始一次真实换道。
        pending_lane = env.unwrapped.vehicle.target_lane_index
        self.assertTrue(env.get_debug_state()["lane_change_in_progress"])
        for action_id, expected_index in ((4, 2), (3, 1), (5, 2)):
            with self.subTest(action=action_id):
                self.assertTrue(env.get_action_mask()[action_id])
                env.step(action_id)
                self.assertEqual(env.unwrapped.vehicle.target_lane_index, pending_lane)
                self.assertEqual(env.unwrapped.vehicle.speed_index, expected_index)
                self.assertAlmostEqual(env.unwrapped.vehicle.target_speed, config.TARGET_SPEEDS_MPS[expected_index])

    def test_lane_edges_and_speed_boundaries_combine_in_mask(self):
        """横向或纵向任一分量无效，整个动作都必须被屏蔽。"""
        for lane_id in (0, 1, 2):
            for speed_index in (0, 2, 4):
                with self.subTest(lane=lane_id, speed_index=speed_index):
                    env, _ = self.make_env(speed_index=speed_index, lane_id=lane_id)
                    expected = []
                    for action_id in range(9):
                        next_lane = lane_id + action_id // 3 - 1
                        next_speed_index = speed_index + action_id % 3 - 1
                        expected.append(0 <= next_lane < 3 and 0 <= next_speed_index < 5)
                    self.assert_mask(env, expected)

    def test_lane_change_lock_waits_for_center_and_then_releases(self):
        """车道编号已经改变但尚未居中时仍锁定，实际收敛后恢复换道。"""
        env, _ = self.make_env(speed_index=3)
        env.step(1)
        self.assert_mask(env, [False, False, False, True, True, True, False, False, False])
        saw_lane_index_change_before_center = False
        completed = False
        for _ in range(100):  # 有界运行最多 20 秒，小于回合时限。
            state = env.get_debug_state()
            if state["current_lane"] == state["target_lane"] and abs(state["lateral_error"]) > config.LANE_CHANGE_LATERAL_TOLERANCE_M:
                saw_lane_index_change_before_center = True
                self.assertTrue(state["lane_change_in_progress"])
                self.assertFalse(np.any(env.get_action_mask()[[0, 1, 2, 6, 7, 8]]))
            if not state["lane_change_in_progress"]:
                completed = True
                self.assertEqual(state["current_lane"], state["target_lane"])
                self.assertLessEqual(abs(state["lateral_error"]), config.LANE_CHANGE_LATERAL_TOLERANCE_M)
                self.assertLessEqual(abs(state["heading_error"]), config.LANE_CHANGE_HEADING_TOLERANCE_RAD)
                break
            _, _, terminated, truncated, _ = env.step(4)
            self.assertFalse(terminated or truncated)
        self.assertTrue(saw_lane_index_change_before_center, "未覆盖车道编号先改变、几何尚未对齐的中间状态")
        self.assertTrue(completed, "正常底层控制收敛后不应永久锁定换道")
        self.assertTrue(env.get_action_mask()[8])
        old_target_lane = env.unwrapped.vehicle.target_lane_index[2]
        env.step(8)
        self.assertEqual(env.unwrapped.vehicle.target_lane_index[2], old_target_lane + 1)
        self.assertEqual(env.unwrapped.vehicle.speed_index, 4)

    def test_heading_alignment_is_required_even_at_lane_center(self):
        """仅坐标居中而航向未对齐不能解锁；只在夹具中设置真实初态。"""
        env, _ = self.make_env()
        vehicle = env.unwrapped.vehicle
        vehicle.heading = 2 * config.LANE_CHANGE_HEADING_TOLERANCE_RAD
        vehicle.on_state_update()
        self.assertEqual(vehicle.lane_index, vehicle.target_lane_index)
        self.assertTrue(env.get_debug_state()["lane_change_in_progress"])
        self.assertFalse(env.get_action_mask()[1])
        self.assertTrue(env.get_action_mask()[4])
        env.step(4)  # 真正推进控制器，验证夹具没有替代动力学。

    def test_zero_target_is_retained_and_vehicle_stops_then_restarts(self):
        """零档不被 truthy 默认值覆盖；通过真实纵向控制接近停车后再起步。"""
        env, _ = self.make_env(speed_index=0)
        vehicle = env.unwrapped.vehicle
        self.assertAlmostEqual(vehicle.speed, 25.0)
        self.assertEqual(vehicle.speed_index, 0)
        self.assertEqual(vehicle.target_speed, 0.0)
        self.assert_mask(env, [False, True, True, False, True, True, False, True, True])
        for _ in range(40):
            _, _, terminated, truncated, _ = env.step(4)
            self.assertFalse(terminated or truncated)
            self.assertEqual(vehicle.target_speed, 0.0)
        self.assertLess(abs(vehicle.speed), 0.1)
        stopped_speed = vehicle.speed
        env.step(5)
        self.assertEqual(vehicle.speed_index, 1)
        self.assertAlmostEqual(vehicle.target_speed, 6.95)
        for _ in range(10):
            env.step(4)
        self.assertGreater(vehicle.speed, stopped_speed + 1.0)
        self.assertAlmostEqual(vehicle.target_speed, 6.95)

    def test_invalid_types_and_ranges_do_not_change_simulation(self):
        """错误类型与越界编号必须在改目标和推进时间之前被拒绝。"""
        env, _ = self.make_env()
        invalid_cases = [
            (-1, ValueError), (9, ValueError), (99, ValueError),
            (True, TypeError), (False, TypeError), (np.bool_(True), TypeError),
            (4.0, TypeError), (np.float64(4), TypeError),
            ("4", TypeError), (None, TypeError), ([4], TypeError),
            (np.array(4), TypeError), (np.array([4]), TypeError),
        ]
        for action, error_type in invalid_cases:
            with self.subTest(action=repr(action)):
                before = self.vehicle_snapshot(env)
                with self.assertRaises(error_type):
                    env.step(action)
                self.assert_snapshot_equal(before, self.vehicle_snapshot(env))
        # 合法的 NumPy 整数仍应符合离散动作接口。
        env.step(np.int64(4))

    def test_masked_actions_do_not_partially_execute(self):
        """一个分量无效时不能只执行另一个分量；换道期间同样严格拒绝。"""
        cases = ((0, 2, 2), (1, 0, 0), (2, 4, 6), (1, 4, 8))
        for lane_id, speed_index, action_id in cases:
            with self.subTest(lane=lane_id, speed_index=speed_index, action=action_id):
                env, _ = self.make_env(speed_index=speed_index, lane_id=lane_id)
                self.assertFalse(env.get_action_mask()[action_id])
                before = self.vehicle_snapshot(env)
                with self.assertRaises(ValueError):
                    env.step(action_id)
                self.assert_snapshot_equal(before, self.vehicle_snapshot(env))
        env, _ = self.make_env()
        env.step(1)
        before = self.vehicle_snapshot(env)
        with self.assertRaises(ValueError):
            env.step(8)
        self.assert_snapshot_equal(before, self.vehicle_snapshot(env))

    def test_reset_rebinds_vehicle_and_clears_pending_lane_change(self):
        """reset 创建新车辆并重置目标及掩码，不沿用上一回合的换道状态。"""
        env, _ = self.make_env(speed_index=3)
        old_vehicle = env.unwrapped.vehicle
        env.step(0)
        self.assertTrue(env.get_debug_state()["lane_change_in_progress"])
        _, info = env.reset(seed=32, options={"initial_target_speed_index": 1})
        vehicle = env.unwrapped.vehicle
        self.assertIsNot(vehicle, old_vehicle)
        self.assertEqual(vehicle.target_lane_index, vehicle.lane_index)
        self.assertFalse(env.get_debug_state()["lane_change_in_progress"])
        self.assertEqual(vehicle.speed_index, 1)
        self.assertAlmostEqual(vehicle.target_speed, 6.95)
        self.assertEqual(env.get_debug_state()["simulation_time"], 0.0)
        self.assert_mask(env, [True] * 9)
        np.testing.assert_array_equal(info["action_mask"], env.get_action_mask())
        # 下一次真正执行的动作必须作用于新车辆。
        env.step(5)
        self.assertEqual(vehicle.speed_index, 2)
        self.assertEqual(old_vehicle.speed_index, 2)

    def test_default_reset_uses_nearest_speed_tier(self):
        """默认原生实际速度 25 m/s 对应最近的 27.80 档；属于工程初始化规则。"""
        env, _ = self.make_env(speed_index=0)
        env.reset(seed=33)
        vehicle = env.unwrapped.vehicle
        expected_index = min(
            range(len(config.TARGET_SPEEDS_MPS)),
            key=lambda index: abs(config.TARGET_SPEEDS_MPS[index] - vehicle.speed),
        )
        self.assertEqual(vehicle.speed_index, expected_index)
        self.assertAlmostEqual(vehicle.target_speed, config.TARGET_SPEEDS_MPS[expected_index])
        self.assertEqual(expected_index, 4)
        env.step(4)

    def test_structure_mask_is_independent_of_traffic_gaps(self):
        """加入近车、改变间距或移除车辆，不改变自车结构状态对应的掩码。"""
        env, _ = self.make_env()
        raw = env.unwrapped
        ego = raw.vehicle
        original_mask = env.get_action_mask().copy()
        other = Vehicle(raw.road, ego.position + np.array([6.0, 0.0]), speed=0.0)
        raw.road.vehicles.append(other)
        np.testing.assert_array_equal(env.get_action_mask(), original_mask)
        other.position = ego.position + np.array([0.0, 4.0])
        other.on_state_update()
        np.testing.assert_array_equal(env.get_action_mask(), original_mask)
        other.position = ego.position + np.array([150.0, 0.0])
        other.on_state_update()
        np.testing.assert_array_equal(env.get_action_mask(), original_mask)
        raw.road.vehicles.remove(other)
        np.testing.assert_array_equal(env.get_action_mask(), original_mask)
        env.step(1)
        pending_mask = env.get_action_mask().copy()
        raw.road.vehicles.append(other)
        np.testing.assert_array_equal(env.get_action_mask(), pending_mask)
        env.step(4)  # 交通测试也推进真实仿真，远处车辆不会触发碰撞。

    def test_seed_and_legal_action_sequence_are_reproducible(self):
        """两个独立真实环境执行相同合法序列，观察、奖励和控制状态一致。"""
        first, _ = self.make_env(seed=71, speed_index=3)
        second, _ = self.make_env(seed=71, speed_index=3)
        # 初始向左减速，完成换道后再向右加速；总时长小于 40 秒。
        actions = [0] + [4] * 40 + [8] + [4] * 40
        for decision, action_id in enumerate(actions):
            with self.subTest(decision=decision, action=action_id):
                self.assertTrue(first.get_action_mask()[action_id])
                self.assertTrue(second.get_action_mask()[action_id])
                first_result = first.step(action_id)
                second_result = second.step(action_id)
                np.testing.assert_array_equal(first_result[0], second_result[0])
                self.assertEqual(first_result[1:4], second_result[1:4])
                self.assertFalse(first_result[2] or first_result[3])
                np.testing.assert_array_equal(first_result[4]["action_mask"], second_result[4]["action_mask"])
                np.testing.assert_array_equal(first.unwrapped.vehicle.position, second.unwrapped.vehicle.position)
                self.assertEqual(first.unwrapped.vehicle.speed, second.unwrapped.vehicle.speed)
                self.assertEqual(first.unwrapped.vehicle.target_lane_index, second.unwrapped.vehicle.target_lane_index)
                self.assertEqual(first.unwrapped.vehicle.target_speed, second.unwrapped.vehicle.target_speed)

    def test_debug_info_distinguishes_before_applied_and_after(self):
        """日志包含动作前、目标写入后、仿真后三个时点；动作写入本身不推进时间。"""
        env, reset_info = self.make_env()
        initial_debug = reset_info["joint_debug"]
        self.assertIsNone(initial_debug["before"])
        self.assertIsNone(initial_debug["targets_applied"])
        self.assertIsNone(initial_debug["action_id"])
        info = env.step(0)[4]
        debug = info["joint_debug"]
        before = debug["before"]
        applied = debug["targets_applied"]
        after = debug["after"]
        self.assertEqual(before["simulation_time"], 0.0)
        self.assertEqual(applied["simulation_time"], before["simulation_time"])
        self.assertEqual(applied["actual_speed"], before["actual_speed"])
        self.assertEqual(before["target_speed_index"], 2)
        self.assertEqual(applied["target_speed_index"], 1)
        self.assertNotEqual(applied["target_lane"], before["target_lane"])
        self.assertEqual(after["target_lane"], applied["target_lane"])
        self.assertEqual(after["target_speed_index"], 1)
        self.assertAlmostEqual(after["simulation_time"], 0.2)
        self.assertEqual(after["physics_steps"] - before["physics_steps"], 3)
        np.testing.assert_array_equal(info["action_mask"], after["action_mask"])

    def test_initial_speed_ties_choose_lower_index(self):
        """等距浮点初速选较低档；只在夹具中设初速，随后推进真实仿真。"""
        for lower in range(4):
            with self.subTest(lower=lower):
                env, _ = self.make_env()
                vehicle = env.unwrapped.vehicle
                vehicle.speed = (config.TARGET_SPEEDS_MPS[lower] + config.TARGET_SPEEDS_MPS[lower + 1]) / 2
                env.controller = JointActionController(vehicle)
                self.assertEqual(vehicle.speed_index, lower)
                env.step(4)
                self.assertAlmostEqual(vehicle.target_speed, config.TARGET_SPEEDS_MPS[lower])

    def test_reset_rejects_incompatible_action_override(self):
        """reset 的配置入口不能把适配层改回另一套动作或速度档位。"""
        env, _ = self.make_env()
        before = self.vehicle_snapshot(env)
        with self.assertRaises(ValueError):
            env.reset(options={"config": {"action": {"type": "DiscreteMetaAction", "target_speeds": [20, 25, 30]}}})
        self.assert_snapshot_equal(before, self.vehicle_snapshot(env))
        env.step(4)

    def test_close_recreate_and_stock_defaults_are_unchanged(self):
        """联合环境可关闭后重建，测试覆盖未污染 stock 的 20 辆或五动作。"""
        first, _ = self.make_env()
        first.step(4)
        first.close()
        second, _ = self.make_env()
        second.step(4)
        stock = make_stock_env(render_mode=None)
        self.addCleanup(stock.close)
        stock.reset(seed=31)
        self.assertEqual(stock.action_space.n, 5)
        self.assertEqual(stock.unwrapped.config["vehicles_count"], 20)
        self.assertEqual(len(stock.unwrapped.road.vehicles), 21)
        stock.step(stock.unwrapped.action_type.actions_indexes["IDLE"])


if __name__ == "__main__":
    unittest.main()
