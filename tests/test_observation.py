"""阶段 1B-2：以真实仿真状态独立核对 34 维观测的物理含义。

测试可显式设置初态，但不替换 reset/step、动力学、观测或奖励。
归一化期望值由测试独立手算；缺少依赖必须报告“未验证”。
"""

import copy
import importlib
import math
import unittest

import config
from environment import make_joint_debug_env, make_observation_debug_env


DEPENDENCY_ERROR = None
try:
    gym = importlib.import_module("gymnasium")
    np = importlib.import_module("numpy")
    importlib.import_module("highway_env")
    from highway_env.vehicle.kinematics import Vehicle
    from observation import ObservationBuilder
except ModuleNotFoundError as exc:
    DEPENDENCY_ERROR = f"未验证：缺少真实仿真依赖 {exc.name}"


@unittest.skipIf(DEPENDENCY_ERROR is not None, DEPENDENCY_ERROR or "")
class ObservationTests(unittest.TestCase):
    """每个实例独立配置无周围车辆，不改变正式演示的 20 辆场景。"""

    def make_env(self, *, seed=91, speed_index=2, factory=None):
        factory = factory or make_observation_debug_env
        env = factory(render_mode=None)
        self.addCleanup(env.close)
        obs, info = env.reset(
            seed=seed,
            options={
                "config": {"vehicles_count": 0, "initial_lane_id": 1},
                "initial_target_speed_index": speed_index,
            },
        )
        self.assertEqual(len(env.unwrapped.road.vehicles), 1)
        return env, obs, info

    def set_ego(self, env, *, speed=20.0, heading=0.0, offset=0.0):
        """仅夹具初始化允许直接设置真实物理状态。"""
        ego = env.unwrapped.vehicle
        ego.position = np.array([200.0, 4.0 + offset])
        ego.speed = speed
        ego.heading = heading
        ego.on_state_update()
        return ego

    def add_vehicle(self, env, *, dx, lane=1, speed=20.0, heading=0.0, y=None):
        ego = env.unwrapped.vehicle
        position = np.array([ego.position[0] + dx, 4.0 * lane if y is None else y])
        other = Vehicle(env.unwrapped.road, position, heading=heading, speed=speed)
        env.unwrapped.road.vehicles.append(other)
        return other

    def assert_observation(self, env, obs):
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (34,))
        self.assertEqual(obs.dtype, np.dtype(np.float32))
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertTrue(env.observation_space.contains(obs))

    def snapshot(self, env):
        """快照覆盖车辆物理/控制状态、历史、时间和两种随机数生成器。"""
        raw = env.unwrapped
        return {
            "time": raw.time,
            "steps": raw.steps,
            "previous_action": env.previous_action,
            "vehicles": [
                {
                    "position": tuple(v.position),
                    "speed": v.speed,
                    "heading": v.heading,
                    "lane_index": v.lane_index,
                    "target_lane_index": getattr(v, "target_lane_index", None),
                    "target_speed": getattr(v, "target_speed", None),
                    "speed_index": getattr(v, "speed_index", None),
                    "action": copy.deepcopy(v.action),
                    "crashed": v.crashed,
                }
                for v in raw.road.vehicles
            ],
            "rng": copy.deepcopy(raw.np_random.bit_generator.state),
            "action_rng": copy.deepcopy(env.action_space.np_random.bit_generator.state),
        }

    def test_reset_and_real_step_return_declared_float32_box(self):
        env, obs, info = self.make_env()
        self.assert_observation(env, obs)
        self.assertIsInstance(env.observation_space, gym.spaces.Box)
        self.assertEqual(env.action_space.n, 9)
        self.assertEqual(env.unwrapped.action_space.n, 5)
        self.assertIs(env.controller, env.env.controller)
        self.assertEqual(env.unwrapped.observation_space.shape, (5, 5))
        expected_low = np.zeros(34, dtype=np.float32)
        expected_low[[2, 5, 8, 11, 14, 17, 18, 19, 20]] = -1
        np.testing.assert_array_equal(env.observation_space.low, expected_low)
        np.testing.assert_array_equal(env.observation_space.high, np.ones(34))
        np.testing.assert_array_equal(info["action_mask"], env.get_action_mask())
        self.assertIn("joint_debug", info)
        self.assertIn("observation_debug", info)
        result = env.step(4)
        self.assertEqual(len(result), 5)
        self.assert_observation(env, result[0])
        self.assertTrue(math.isfinite(result[1]))
        np.testing.assert_array_equal(result[4]["action_mask"], env.get_action_mask())

    def test_known_scene_has_all_traffic_slots_and_seven_ego_features(self):
        env, _, _ = self.make_env()
        self.set_ego(env, offset=0.5)
        cases = [
            (0, 20, 10), (0, -30, 30),
            (1, 40, 15), (1, -50, 25),
            (2, 60, 0), (2, -70, 20),
        ]
        for lane, dx, speed in cases:
            self.add_vehicle(env, dx=dx, lane=lane, speed=speed)
        obs, diag = env.get_observation()
        # 两车都为 5 米长；前后方向均为 other_speed - ego_speed。
        expected_traffic = np.array([
            1, 0.15, -0.25, 1, 0.25, 0.25,
            1, 0.35, -0.125, 1, 0.45, 0.125,
            1, 0.55, -0.5, 1, 0.65, 0,
        ])
        np.testing.assert_allclose(obs[:18], expected_traffic, atol=1e-7)
        # 实际纵速20/40、横速0、偏移0.5/2、车道1/2、目标13.9/27.8。
        np.testing.assert_allclose(obs[18:25], [0.5, 0, 0.25, 0.5, 0.5, 0.5, 1], atol=1e-7)
        np.testing.assert_array_equal(obs[25:], np.zeros(9))
        np.testing.assert_allclose(
            diag["raw_values"][:18],
            [1, 15, -10, 1, 25, 10, 1, 35, -5, 1, 45, 5, 1, 55, -20, 1, 65, 0],
        )
        self.assertEqual(tuple(diag["feature_names"]), config.OBSERVATION_FEATURE_NAMES)
        self.assertEqual(len(set(diag["feature_names"])), 34)
        np.testing.assert_array_equal(diag["normalized_values"], obs)
        env.step(4)  # 这些夹具仍连接真实动力学。

    def test_nearest_visible_front_and_rear_are_selected_per_fixed_lane(self):
        env, _, _ = self.make_env()
        self.set_ego(env)
        for lane in range(3):
            # 先放远车，避免误将列表首车当作最近车。
            self.add_vehicle(env, dx=80, lane=lane, speed=39)
            self.add_vehicle(env, dx=-80, lane=lane, speed=1)
            self.add_vehicle(env, dx=20 + lane, lane=lane, speed=10 + lane)
            self.add_vehicle(env, dx=-30 - lane, lane=lane, speed=30 + lane)
        obs, _ = env.get_observation()
        for lane in range(3):
            with self.subTest(lane=lane):
                start = 6 * lane
                np.testing.assert_allclose(obs[start:start + 6], [
                    1, (15 + lane) / 100, (-10 + lane) / 40,
                    1, (25 + lane) / 100, (10 + lane) / 40,
                ], atol=1e-7)

    def test_visibility_filter_precedes_nearest_selection(self):
        """同槽中纵向更近但欧氏不可见的车，不能挡住可见候选。"""
        env, _, _ = self.make_env()
        self.set_ego(env)
        # 两车按位置都属于车道2；99.9纵向、5.9横向的车在100m外。
        self.add_vehicle(env, dx=99.9, y=9.9, speed=1)
        self.add_vehicle(env, dx=99.95, y=6.1, speed=30)
        obs, diag = env.get_observation()
        np.testing.assert_allclose(obs[12:15], [1, 0.9495, 0.25], atol=1e-7)
        self.assertAlmostEqual(diag["raw_values"][13], 94.95)

    def test_ego_is_excluded_and_missing_encoding_is_explicit(self):
        env, _, _ = self.make_env()
        self.set_ego(env)
        obs, diag = env.get_observation()
        np.testing.assert_array_equal(obs[:18], [0, 1, 0] * 6)
        np.testing.assert_array_equal(diag["raw_values"][:18], [0, 100, 0] * 6)
        env.step(4)

    def test_center_distance_visibility_boundaries_on_same_lane(self):
        for distance, expected_presence in ((99.9, 1), (100.0, 1), (100.1, 0)):
            with self.subTest(distance=distance):
                env, _, _ = self.make_env()
                self.set_ego(env)
                self.add_vehicle(env, dx=distance)
                obs, _ = env.get_observation()
                self.assertEqual(obs[6], expected_presence)
                expected_gap = (distance - 5) / 100 if expected_presence else 1
                self.assertAlmostEqual(float(obs[7]), expected_gap, places=6)

    def test_adjacent_lane_uses_euclidean_distance_not_longitudinal_distance(self):
        for distance, visible in ((99.9, True), (100.0, True), (100.1, False)):
            with self.subTest(distance=distance):
                env, _, _ = self.make_env()
                self.set_ego(env)
                # 邻道横向距离4米，纵向差小于100不代表欧氏距离在100内。
                dx = math.sqrt(distance * distance - 4 * 4)
                self.add_vehicle(env, dx=dx, lane=0)
                obs, _ = env.get_observation()
                self.assertEqual(obs[0], int(visible))
                self.assertAlmostEqual(float(obs[1]), (dx - 5) / 100 if visible else 1, places=6)

    def test_bumper_gap_accounts_for_lengths_and_nonzero_heading_projection(self):
        env, _, _ = self.make_env()
        self.set_ego(env, speed=20, heading=0.2)
        other = self.add_vehicle(env, dx=30, lane=1, speed=15, heading=-0.3)
        other.LENGTH = 8.0
        other.WIDTH = 2.5
        obs, diag = env.get_observation()
        # 独立投影公式：矩形沿道路轴半长=(L|cos θ|+W|sin θ|)/2。
        ego_half = (5 * abs(math.cos(0.2)) + 2 * abs(math.sin(0.2))) / 2
        other_half = (8 * abs(math.cos(-0.3)) + 2.5 * abs(math.sin(-0.3))) / 2
        gap = 30 - ego_half - other_half
        relative_speed = 15 * math.cos(-0.3) - 20 * math.cos(0.2)
        self.assertAlmostEqual(float(diag["raw_values"][7]), gap)
        self.assertAlmostEqual(float(obs[7]), gap / 100, places=6)
        self.assertAlmostEqual(float(obs[8]), relative_speed / 40, places=6)
        self.assertAlmostEqual(float(obs[18]), 20 * math.cos(0.2) / 40, places=6)
        self.assertAlmostEqual(float(obs[19]), 20 * math.sin(0.2) / 10, places=6)
        self.assertNotAlmostEqual(gap, 30)

    def test_missing_and_overlapping_vehicle_have_different_presence(self):
        env, _, _ = self.make_env()
        self.set_ego(env)
        absent, _ = env.get_observation()
        self.add_vehicle(env, dx=3, speed=20)
        overlap, diag = env.get_observation()
        np.testing.assert_array_equal(absent[6:9], [0, 1, 0])
        np.testing.assert_array_equal(overlap[6:9], [1, 0, 0])
        self.assertEqual(diag["raw_values"][7], 0)

    def test_front_slow_and_rear_fast_use_same_relative_speed_sign(self):
        env, _, _ = self.make_env()
        self.set_ego(env, speed=20)
        self.add_vehicle(env, dx=30, speed=12)
        self.add_vehicle(env, dx=-30, speed=28)
        obs, diag = env.get_observation()
        self.assertAlmostEqual(float(obs[8]), -0.2)
        self.assertAlmostEqual(float(obs[11]), 0.2)
        self.assertEqual(diag["raw_values"][8], -8)
        self.assertEqual(diag["raw_values"][11], 8)

    def test_outside_vehicle_cannot_change_observation_without_advancing_time(self):
        env, _, _ = self.make_env()
        ego = self.set_ego(env)
        baseline, _ = env.get_observation()
        other = self.add_vehicle(env, dx=100.1, speed=3)
        outside, _ = env.get_observation()
        np.testing.assert_array_equal(outside, baseline)
        other.position = ego.position + np.array([-200.0, 4.0])
        other.speed = 70
        other.heading = math.pi / 3
        other.LENGTH = 15
        other.on_state_update()
        moved, _ = env.get_observation()
        np.testing.assert_array_equal(moved, baseline)
        self.assertEqual(env.unwrapped.time, 0)
        # 同一车辆从可见区域移出后，只留下既定缺失编码。
        other.position = ego.position + np.array([30.0, 0.0])
        other.on_state_update()
        visible, _ = env.get_observation()
        self.assertEqual(visible[6], 1)
        other.position = ego.position + np.array([101.0, 0.0])
        other.on_state_update()
        removed, _ = env.get_observation()
        np.testing.assert_array_equal(removed, baseline)

    def test_hidden_targets_and_stale_lane_cache_do_not_select_observation_slot(self):
        env, _, _ = self.make_env()
        self.set_ego(env)
        other = self.add_vehicle(env, dx=30, lane=0, speed=10)
        before, _ = env.get_observation()
        other.target_speed = 99
        other.target_lane_index = (*env.unwrapped.vehicle.lane_index[:2], 2)
        # 强行构造过期缓存：观测仍须按当前几何位置归入车道0。
        other.lane_index = (*env.unwrapped.vehicle.lane_index[:2], 2)
        after, _ = env.get_observation()
        np.testing.assert_array_equal(before, after)
        self.assertEqual(after[0], 1)
        self.assertEqual(after[12], 0)
        other.target_speed = -20
        other.target_lane_index = (*env.unwrapped.vehicle.lane_index[:2], 1)
        again, _ = env.get_observation()
        np.testing.assert_array_equal(before, again)

    def test_lane_boundary_tie_uses_lower_lane_and_position_over_intent(self):
        env, _, _ = self.make_env()
        self.set_ego(env)
        first = self.add_vehicle(env, dx=30, y=2.0, speed=10)
        self.add_vehicle(env, dx=-30, y=6.0, speed=30)
        first.target_lane_index = (*env.unwrapped.vehicle.lane_index[:2], 2)
        obs, _ = env.get_observation()
        np.testing.assert_array_equal(obs[[0, 3, 6, 9, 12, 15]], [1, 0, 0, 1, 0, 0])
        first.position[1] = 2.01
        # 故意不调用 on_state_update：归属读取位置而非缓存。
        moved, _ = env.get_observation()
        np.testing.assert_array_equal(moved[[0, 3, 6, 9, 12, 15]], [0, 0, 1, 1, 0, 0])

    def test_longitudinal_ties_are_deterministic_and_zero_is_front(self):
        env, _, _ = self.make_env()
        self.set_ego(env)
        self.add_vehicle(env, dx=20, lane=0, speed=8)
        self.add_vehicle(env, dx=20, lane=0, speed=32)
        self.add_vehicle(env, dx=-20, lane=2, speed=12)
        self.add_vehicle(env, dx=-20, lane=2, speed=28)
        self.add_vehicle(env, dx=0, lane=1, speed=20)
        obs, _ = env.get_observation()
        # 等纵向距离时 road.vehicles 列表先出现者优先。
        self.assertAlmostEqual(float(obs[2]), (8 - 20) / 40)
        self.assertAlmostEqual(float(obs[17]), (12 - 20) / 40)
        np.testing.assert_array_equal(obs[6:12], [1, 0, 0, 0, 1, 0])

    def test_action_history_records_all_nine_project_actions_after_real_step(self):
        for action in range(9):
            with self.subTest(action=action):
                env, initial, _ = self.make_env()
                self.assertIsNone(env.previous_action)
                np.testing.assert_array_equal(initial[25:], np.zeros(9))
                obs, _, _, _, info = env.step(action)
                expected = np.zeros(9)
                expected[action] = 1
                np.testing.assert_array_equal(obs[25:], expected)
                self.assertEqual(env.previous_action, action)
                self.assertEqual(info["joint_debug"]["action_id"], action)
                self.assertEqual(info["joint_debug"]["native_action_id"], 1)
                reread, _ = env.get_observation()
                np.testing.assert_array_equal(reread, obs)

    def test_rejected_action_preserves_history_and_reset_clears_it(self):
        env, _, _ = self.make_env()
        env.step(1)  # 启动换道，此时 Left/Right 受结构掩码禁止。
        before_obs, _ = env.get_observation()
        before_state = self.snapshot(env)
        for action, error in ((9, ValueError), (True, TypeError), (8, ValueError)):
            with self.subTest(action=action):
                with self.assertRaises(error):
                    env.step(action)
                after_obs, _ = env.get_observation()
                np.testing.assert_array_equal(after_obs, before_obs)
                self.assertEqual(self.snapshot(env), before_state)
        reset_obs, info = env.reset(seed=92)
        self.assertIsNone(env.previous_action)
        np.testing.assert_array_equal(reset_obs[25:], np.zeros(9))
        self.assertEqual(reset_obs[24], 0)
        np.testing.assert_array_equal(info["action_mask"], env.get_action_mask())
        obs, *_ = env.step(4)
        self.assertEqual(obs[25 + 4], 1)

    def test_lane_change_flag_matches_existing_controller_and_mask(self):
        env, initial, _ = self.make_env()
        self.assertEqual(initial[24], 0)
        obs, _, _, _, info = env.step(1)
        self.assertEqual(obs[24], 1)
        self.assertTrue(env.controller.is_lane_change_in_progress())
        self.assertTrue(info["joint_debug"]["after"]["lane_change_in_progress"])
        self.assertFalse(np.any(info["action_mask"][[0, 1, 2, 6, 7, 8]]))
        for _ in range(100):
            if not env.controller.is_lane_change_in_progress():
                break
            obs, _, terminated, truncated, info = env.step(4)
            self.assertFalse(terminated or truncated)
            self.assertEqual(bool(obs[24]), env.controller.is_lane_change_in_progress())
        self.assertEqual(obs[24], 0)
        self.assertTrue(env.get_action_mask()[7])

    def test_fixed_normalization_preserves_actual_above_target_and_reports_clips(self):
        env, _, _ = self.make_env(speed_index=4)
        ego = self.set_ego(env, speed=35)
        obs, diag = env.get_observation()
        self.assertAlmostEqual(float(obs[18]), 35 / 40)
        self.assertEqual(obs[23], 1)
        self.assertFalse(diag["clipped_features"])
        ego.speed = 60
        self.add_vehicle(env, dx=30, speed=-50)
        clipped, clip_diag = env.get_observation()
        self.assertEqual(clipped[18], 1)
        self.assertEqual(clipped[8], -1)
        reports = {item["index"]: item for item in clip_diag["clipped_features"]}
        self.assertEqual(set(reports), {8, 18})
        for index, raw, unclipped, normalized in ((18, 60, 1.5, 1), (8, -110, -2.75, -1)):
            item = reports[index]
            self.assertEqual(item["name"], config.OBSERVATION_FEATURE_NAMES[index])
            self.assertAlmostEqual(item["raw_value"], raw)
            self.assertAlmostEqual(item["unclipped_normalized"], unclipped)
            self.assertEqual(item["normalized_value"], normalized)
            self.assertEqual(item["low"], -1)
            self.assertEqual(item["high"], 1)
        self.assert_observation(env, clipped)

    def test_signed_velocity_and_offset_bounds_and_clipping(self):
        # 边界恰好落在[-1,1]不应产生超界诊断；符号由道路坐标决定。
        cases = [
            (-40, 0, 0, 18, -1),
            (40, 0, 0, 18, 1),
            (10, math.pi / 2, 0, 19, 1),
            (10, -math.pi / 2, 0, 19, -1),
            (20, 0, 2, 20, 1),
            (20, 0, -2, 20, 1),  # y=2正好归车道0，距其中心+2米。
        ]
        for speed, heading, offset, index, expected in cases:
            with self.subTest(index=index, speed=speed, heading=heading, offset=offset):
                env, _, _ = self.make_env()
                self.set_ego(env, speed=speed, heading=heading, offset=offset)
                obs, diag = env.get_observation()
                self.assertAlmostEqual(float(obs[index]), expected, places=6)
                self.assertNotIn(index, [item["index"] for item in diag["clipped_features"]])
        env, _, _ = self.make_env()
        ego = self.set_ego(env, speed=20, heading=math.pi / 2)
        obs, diag = env.get_observation()
        self.assertEqual(obs[19], 1)
        self.assertIn(19, [item["index"] for item in diag["clipped_features"]])
        ego.heading = 0
        ego.position[1] = -3
        ego.on_state_update()
        obs, diag = env.get_observation()
        self.assertEqual(obs[20], -1)
        self.assertIn(20, [item["index"] for item in diag["clipped_features"]])

    def test_nonfinite_physical_values_are_errors_not_silently_clipped(self):
        for attribute, value in (("speed", float("nan")), ("heading", float("inf"))):
            with self.subTest(attribute=attribute):
                env, _, _ = self.make_env()
                ego = self.set_ego(env)
                setattr(ego, attribute, value)
                with self.assertRaises(ValueError):
                    env.get_observation()
        env, _, _ = self.make_env()
        self.set_ego(env)
        other = self.add_vehicle(env, dx=30, speed=float("nan"))
        with self.assertRaises(ValueError):
            env.get_observation()
        other.speed = 20
        other.LENGTH = float("inf")
        with self.assertRaises(ValueError):
            env.get_observation()

    def test_repeated_observation_and_independent_builder_have_no_side_effects(self):
        env, _, _ = self.make_env()
        self.set_ego(env)
        self.add_vehicle(env, dx=30, lane=0, speed=10)
        env.action_space.seed(17)
        before = self.snapshot(env)
        first, diag = env.get_observation()
        builder = ObservationBuilder(env.controller)
        for _ in range(5):
            again, again_diag = env.get_observation()
            independent, _ = builder.build(previous_action=env.previous_action)
            np.testing.assert_array_equal(again, first)
            np.testing.assert_array_equal(independent, first)
            np.testing.assert_array_equal(again_diag["raw_values"], diag["raw_values"])
            self.assertEqual(self.snapshot(env), before)
        self.assertTrue(builder.observation_space.contains(first))
        # 改写返回副本不能污染下一次读取或真实状态。
        first[:] = 0
        diag["raw_values"][:] = -999
        fresh, fresh_diag = env.get_observation()
        self.assertEqual(fresh[0], 1)
        self.assertEqual(fresh_diag["raw_values"][1], 25)
        self.assertEqual(self.snapshot(env), before)
        env.step(4)

    def test_same_seed_and_legal_sequence_reproduce_observations(self):
        first, first_obs, _ = self.make_env(seed=117, speed_index=3)
        second, second_obs, _ = self.make_env(seed=117, speed_index=3)
        np.testing.assert_array_equal(first_obs, second_obs)
        sequence = [0] + [4] * 40 + [8] + [4] * 40
        for index, action in enumerate(sequence):
            with self.subTest(decision=index, action=action):
                first_result = first.step(action)
                second_result = second.step(action)
                np.testing.assert_array_equal(first_result[0], second_result[0])
                self.assertEqual(first_result[1:4], second_result[1:4])
                np.testing.assert_array_equal(first_result[4]["action_mask"], second_result[4]["action_mask"])
                np.testing.assert_array_equal(
                    first_result[4]["observation_debug"]["raw_values"],
                    second_result[4]["observation_debug"]["raw_values"],
                )
                self.assertFalse(first_result[2] or first_result[3])

    def test_observation_wrapper_adds_no_physics_steps_or_control_changes(self):
        wrapped, _, _ = self.make_env(seed=127, speed_index=3)
        joint, _, _ = self.make_env(seed=127, speed_index=3, factory=make_joint_debug_env)
        actions = [0] + [4] * 40 + [8]
        for decision, action in enumerate(actions, start=1):
            with self.subTest(decision=decision):
                before_time = wrapped.unwrapped.time
                before_steps = wrapped.unwrapped.steps
                wrapped_result = wrapped.step(action)
                joint_result = joint.step(action)
                self.assertAlmostEqual(wrapped.unwrapped.time - before_time, 0.2)
                self.assertEqual(wrapped.unwrapped.steps - before_steps, 3)
                self.assertEqual(wrapped.unwrapped.steps, joint.unwrapped.steps)
                self.assertEqual(wrapped.unwrapped.time, joint.unwrapped.time)
                np.testing.assert_array_equal(wrapped.unwrapped.vehicle.position, joint.unwrapped.vehicle.position)
                self.assertEqual(wrapped.unwrapped.vehicle.speed, joint.unwrapped.vehicle.speed)
                self.assertEqual(wrapped.unwrapped.vehicle.target_lane_index, joint.unwrapped.vehicle.target_lane_index)
                self.assertEqual(wrapped.unwrapped.vehicle.target_speed, joint.unwrapped.vehicle.target_speed)
                self.assertEqual(wrapped_result[1:4], joint_result[1:4])
                time_after_step = wrapped.unwrapped.time
                wrapped.get_observation()
                self.assertEqual(wrapped.unwrapped.time, time_after_step)


if __name__ == "__main__":
    unittest.main()
