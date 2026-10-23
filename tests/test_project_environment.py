"""正式 Gym 环境：真实动力学、奖励、终止与非法动作策略的独立验收。

允许夹具设置车辆初态；每个行为断言均通过真实 reset/step 验证。
奖励预期从物理位置和独立公式计算，不调用 reward 模块生成期望答案。
"""

import copy
import importlib
import math
import unittest

import config


DEPENDENCY_ERROR = None
try:
    gym = importlib.import_module("gymnasium")
    np = importlib.import_module("numpy")
    importlib.import_module("highway_env")
    from highway_env.vehicle.kinematics import Vehicle
    from environment import HighwayLaneEnv, make_env
except ModuleNotFoundError as exc:
    DEPENDENCY_ERROR = f"未验证：缺少真实仿真依赖 {exc.name}"


@unittest.skipIf(DEPENDENCY_ERROR is not None, DEPENDENCY_ERROR or "")
class ProjectEnvironmentTests(unittest.TestCase):
    """空交通只属于各测试实例，正式默认场景仍保留 20 辆周车。"""

    def make_project(self, *, mode="hold", lane=1, speed_index=2, seed=117,
                     render_mode=None):
        env = make_env(render_mode=render_mode, invalid_action_mode=mode)
        self.addCleanup(env.close)
        obs, info = env.reset(seed=seed, options={
            "config": {"vehicles_count": 0, "initial_lane_id": lane},
            "initial_target_speed_index": speed_index,
        })
        self.assertEqual(len(env.simulator.unwrapped.road.vehicles), 1)
        return env, obs, info

    def assert_observation(self, env, obs):
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (34,))
        self.assertEqual(obs.dtype, np.dtype(np.float32))
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertTrue(env.observation_space.contains(obs))

    def assert_reward_sum(self, reward, info):
        self.assertIsInstance(reward, float)
        self.assertTrue(math.isfinite(reward))
        self.assertEqual(set(info["reward_terms"]), {
            "progress", "headway", "failure", "lane_change", "target_speed",
        })
        self.assertAlmostEqual(reward, sum(info["reward_terms"].values()), places=12)

    def set_ego(self, env, *, speed=20.85, lane=1, heading=0.0, y=None):
        """仅为构造独立物理初态，不替换动力学。"""
        ego = env.simulator.unwrapped.vehicle
        ego.position = np.array([200.0, float(4 * lane if y is None else y)])
        ego.speed = speed
        ego.heading = heading
        ego.on_state_update()
        return ego

    def add_vehicle(self, env, *, dx, lane=1, speed=20.85, y=None):
        raw = env.simulator.unwrapped
        vehicle = Vehicle(raw.road, np.array([
            raw.vehicle.position[0] + dx,
            float(4 * lane if y is None else y),
        ]), speed=speed)
        raw.road.vehicles.append(vehicle)
        return vehicle

    def snapshot(self, env):
        """拒绝动作时也不能改变动作历史、累计奖励、随机数或任一车辆。"""
        raw = env.simulator.unwrapped
        obs, _ = env.simulator.get_observation()
        return {
            "time": raw.time, "steps": raw.steps,
            "observation": obs.tobytes(),
            "mask": env.get_action_mask().tobytes(),
            "episode_reward": env.episode_reward,
            "episode_steps": env.episode_steps,
            "invalid_action_count": env.invalid_action_count,
            "previous_action": env.simulator.previous_action,
            "rng": copy.deepcopy(raw.np_random.bit_generator.state),
            "vehicles": [
                (tuple(v.position), v.speed, v.heading, v.lane_index,
                 getattr(v, "target_lane_index", None),
                 getattr(v, "target_speed", None), getattr(v, "speed_index", None),
                 copy.deepcopy(v.action), v.crashed)
                for v in raw.road.vehicles
            ],
        }

    @staticmethod
    def independent_gap(ego, other):
        """当前三条直车道沿 x 方向；矩形投影用数学公式独立计算。"""
        half_ego = (ego.LENGTH * abs(math.cos(ego.heading))
                    + ego.WIDTH * abs(math.sin(ego.heading))) / 2
        half_other = (other.LENGTH * abs(math.cos(other.heading))
                      + other.WIDTH * abs(math.sin(other.heading))) / 2
        return max(0.0, other.position[0] - ego.position[0] - half_ego - half_other)

    def test_formal_gym_interface_default_scene_and_reset_diagnostics(self):
        env = HighwayLaneEnv(render_mode=None)
        self.addCleanup(env.close)
        self.assertIsInstance(env, gym.Env)
        self.assertIs(env.unwrapped, env, "Gym 检查必须检查正式环境本身")
        with self.assertRaises(gym.error.ResetNeeded):
            env.step(4)
        obs, info = env.reset(seed=0)
        self.assert_observation(env, obs)
        self.assertEqual(env.action_space.n, 9)
        self.assertEqual(env.simulator.unwrapped.action_space.n, 5)
        raw = env.simulator.unwrapped
        for key, expected in {
            "lanes_count": 3, "vehicles_count": 20, "controlled_vehicles": 1,
            "duration": 40, "policy_frequency": 5, "simulation_frequency": 15,
            "offroad_terminal": True,
        }.items():
            self.assertEqual(raw.config[key], expected)
        self.assertEqual(len(raw.road.vehicles), 21)
        self.assertEqual(info["reward_terms"], dict.fromkeys(
            ("progress", "headway", "failure", "lane_change", "target_speed"), 0.0))
        self.assertEqual(info["reward_diagnostics"], {})
        self.assertIsNone(info["requested_action"])
        self.assertIsNone(info["executed_action"])
        self.assertFalse(info["invalid_action"])
        self.assertEqual(info["episode_reward"], 0.0)
        self.assertEqual(info["episode_steps"], 0)
        self.assertEqual(info["invalid_action_count"], 0)
        np.testing.assert_array_equal(obs[25:], np.zeros(9))
        np.testing.assert_array_equal(info["action_mask"], env.get_action_mask())
        self.assertEqual(set(env.metadata["render_modes"]), {"human", "rgb_array"})

    def test_all_nine_legal_actions_match_in_both_modes(self):
        """两个模式对合法动作必须产生相同物理结果、奖励和动作历史。"""
        for action in range(9):
            with self.subTest(action=action):
                hold, _, _ = self.make_project(mode="hold")
                strict, _, _ = self.make_project(mode="raise")
                before = hold.simulator.unwrapped.vehicle.position.copy()
                first = hold.step(action)
                second = strict.step(action)
                np.testing.assert_array_equal(first[0], second[0])
                self.assertEqual(first[1:4], second[1:4])
                self.assertEqual(first[4]["reward_terms"], second[4]["reward_terms"])
                self.assert_observation(hold, first[0])
                self.assert_reward_sum(first[1], first[4])
                self.assertEqual(first[4]["requested_action"], action)
                self.assertEqual(first[4]["executed_action"], action)
                self.assertFalse(first[4]["invalid_action"])
                self.assertAlmostEqual(first[4]["simulation_time"], 0.2)
                self.assertEqual(first[4]["physics_steps"], 3)
                self.assertGreater(hold.simulator.unwrapped.vehicle.position[0], before[0])
                self.assertEqual(hold.simulator.unwrapped.vehicle.speed_index,
                                 2 + action % 3 - 1)
                self.assertEqual(hold.simulator.unwrapped.vehicle.target_lane_index[2],
                                 1 + action // 3 - 1)
                expected_history = np.zeros(9)
                expected_history[action] = 1
                np.testing.assert_array_equal(first[0][25:], expected_history)
                np.testing.assert_array_equal(first[4]["action_mask"], hold.get_action_mask())

    def test_hold_fallback_replaces_entire_action_and_records_executed_history(self):
        cases = [(0, 2, 2), (2, 2, 6), (1, 0, 0), (1, 4, 8)]
        for lane, speed_index, action in cases:
            with self.subTest(lane=lane, speed_index=speed_index, action=action):
                env, _, _ = self.make_project(lane=lane, speed_index=speed_index)
                vehicle = env.simulator.unwrapped.vehicle
                lane_target = vehicle.target_lane_index
                speed_target = vehicle.target_speed
                self.assertFalse(env.get_action_mask()[action])
                obs, reward, _, _, info = env.step(action)
                self.assertEqual(info["requested_action"], action)
                self.assertEqual(info["executed_action"], 4)
                self.assertTrue(info["invalid_action"])
                self.assertEqual(info["invalid_action_count"], 1)
                self.assertEqual(vehicle.target_lane_index, lane_target)
                self.assertEqual(vehicle.target_speed, speed_target)
                self.assertEqual(vehicle.speed_index, speed_index)
                self.assertEqual(info["reward_terms"]["lane_change"], 0.0)
                self.assertEqual(info["reward_terms"]["target_speed"], 0.0)
                self.assert_reward_sum(reward, info)
                np.testing.assert_array_equal(obs[25:], np.eye(9)[4])
                self.assertAlmostEqual(info["simulation_time"], 0.2)

    def test_pending_lane_change_fallback_does_not_cancel_target(self):
        env, _, _ = self.make_project()
        env.step(1)
        target = env.simulator.unwrapped.vehicle.target_lane_index
        self.assertTrue(env.get_debug_state()["lane_change_in_progress"])
        obs, _, _, _, info = env.step(8)
        self.assertTrue(info["invalid_action"])
        self.assertEqual(info["executed_action"], 4)
        self.assertEqual(env.simulator.unwrapped.vehicle.target_lane_index, target)
        self.assertEqual(env.simulator.unwrapped.vehicle.speed_index, 2)
        np.testing.assert_array_equal(obs[25:], np.eye(9)[4])
        self.assertEqual(info["reward_terms"]["lane_change"], 0.0)
        self.assertEqual(info["reward_terms"]["target_speed"], 0.0)

    def test_raise_mask_rejection_is_atomic(self):
        env, _, _ = self.make_project(mode="raise", lane=0)
        env.step(4)
        before = self.snapshot(env)
        with self.assertRaises(ValueError):
            env.step(2)
        self.assertEqual(self.snapshot(env), before)
        env2, _, _ = self.make_project(mode="raise")
        env2.step(1)
        before = self.snapshot(env2)
        with self.assertRaises(ValueError):
            env2.step(8)
        self.assertEqual(self.snapshot(env2), before)

    def test_bad_types_and_out_of_range_always_reject_atomically(self):
        cases = [(-1, ValueError), (9, ValueError), (True, TypeError),
                 (np.bool_(False), TypeError), (4.0, TypeError), ("4", TypeError),
                 (None, TypeError), ([4], TypeError), (np.array(4), TypeError)]
        for mode in ("hold", "raise"):
            env, _, _ = self.make_project(mode=mode)
            for action, error in cases:
                with self.subTest(mode=mode, action=repr(action)):
                    before = self.snapshot(env)
                    with self.assertRaises(error):
                        env.step(action)
                    self.assertEqual(self.snapshot(env), before)
            env.step(np.int64(4))

    def test_no_front_vehicle_known_progress_and_native_reward_is_not_added(self):
        env, _, _ = self.make_project(speed_index=4)
        ego = self.set_ego(env, speed=27.80)
        initial_x = ego.position[0]
        obs, reward, terminated, truncated, info = env.step(4)
        self.assertFalse(terminated or truncated)
        self.assertAlmostEqual(ego.position[0] - initial_x, 27.8 * 0.2)
        self.assertAlmostEqual(reward, 0.2)
        self.assertAlmostEqual(info["reward_terms"]["progress"], 0.2)
        self.assertEqual(info["reward_terms"]["headway"], 0.0)
        self.assertEqual(info["reward_terms"]["failure"], 0.0)
        self.assertNotAlmostEqual(info["native_reward_diagnostic"], 0.0)
        self.assertNotAlmostEqual(reward, 0.2 + info["native_reward_diagnostic"])
        self.assert_reward_sum(reward, info)
        self.assert_observation(env, obs)

    def test_lane_and_speed_costs_apply_once_to_accepted_targets(self):
        env, _, _ = self.make_project(speed_index=3)
        ego = self.set_ego(env, speed=25.0)
        before_x = ego.position[0]
        _, reward, _, _, info = env.step(0)
        p = min(1.0, max(0.0, (ego.position[0] - before_x) / (27.8 * 0.2)))
        self.assertAlmostEqual(info["reward_terms"]["lane_change"], -0.05)
        self.assertAlmostEqual(info["reward_terms"]["target_speed"], -0.02)
        self.assertAlmostEqual(reward, 0.2 * p - 0.05 - 0.02)
        self.assertTrue(env.get_debug_state()["lane_change_in_progress"])
        _, reward2, _, _, next_info = env.step(4)
        self.assertEqual(next_info["reward_terms"]["lane_change"], 0.0)
        self.assertEqual(next_info["reward_terms"]["target_speed"], 0.0)
        self.assert_reward_sum(reward2, next_info)
        self.assertAlmostEqual(next_info["episode_reward"], reward + reward2)
        self.assertEqual(next_info["episode_steps"], 2)

    def test_constant_speed_short_headway_has_independent_expected_reward(self):
        env, _, _ = self.make_project(speed_index=2)
        ego = self.set_ego(env, speed=13.9)
        other = self.add_vehicle(env, dx=15.425, speed=13.9)
        _, reward, terminated, truncated, info = env.step(4)
        self.assertFalse(terminated or truncated)
        # 两车同速直行，5m车长：15.425-5=10.425m；THW=10.425/13.9=0.75s。
        self.assertAlmostEqual(other.position[0] - ego.position[0], 15.425)
        self.assertAlmostEqual(info["reward_diagnostics"]["min_thw"], 0.75)
        self.assertAlmostEqual(info["reward_terms"]["headway"], -0.2)
        self.assertAlmostEqual(info["reward_terms"]["progress"], 0.1)
        self.assertAlmostEqual(reward, -0.1)
        self.assert_reward_sum(reward, info)

    def test_headway_checks_current_lane_then_current_and_target_during_change(self):
        env, _, _ = self.make_project(speed_index=3)
        ego = self.set_ego(env, speed=20.85)
        current = self.add_vehicle(env, dx=60.0, lane=1)
        target = self.add_vehicle(env, dx=15.0, lane=0)
        self.add_vehicle(env, dx=6.0, lane=2)
        _, _, _, _, first_info = env.step(4)
        self.assertEqual(set(first_info["reward_diagnostics"]["checked_lanes"]), {1})
        self.assertEqual(first_info["reward_terms"]["headway"], 0.0)
        _, reward, terminated, truncated, info = env.step(1)
        self.assertFalse(terminated or truncated)
        self.assertEqual(set(info["reward_diagnostics"]["checked_lanes"]), {0, 1})
        v_long = ego.speed * math.cos(ego.heading)
        expected_thw = min(self.independent_gap(ego, current),
                           self.independent_gap(ego, target)) / max(v_long, 0.1)
        expected_h = min(1.0, max(0.0, 1.0 - expected_thw / 1.5))
        self.assertAlmostEqual(info["reward_diagnostics"]["min_thw"], expected_thw)
        self.assertAlmostEqual(info["reward_terms"]["headway"], -0.4 * expected_h)
        self.assert_reward_sum(reward, info)

    def test_front_car_beyond_perception_does_not_enter_headway(self):
        env, _, _ = self.make_project(speed_index=4)
        ego = self.set_ego(env, speed=27.8)
        other = self.add_vehicle(env, dx=101.0, speed=27.8)
        _, reward, _, _, info = env.step(4)
        self.assertGreater(np.linalg.norm(other.position - ego.position), 100)
        self.assertEqual(info["reward_terms"]["headway"], 0.0)
        self.assertIsNone(info["reward_diagnostics"]["min_thw"])
        self.assertAlmostEqual(reward, 0.2)

    def test_reward_uses_unclipped_physical_speed(self):
        env, _, _ = self.make_project(speed_index=4)
        ego = self.set_ego(env, speed=60.0)
        other = self.add_vehicle(env, dx=50.0, speed=60.0)
        obs, reward, terminated, truncated, info = env.step(4)
        self.assertFalse(terminated or truncated)
        v_long = ego.speed * math.cos(ego.heading)
        self.assertGreater(v_long, 40.0)
        self.assertEqual(obs[18], 1.0)
        gap = self.independent_gap(ego, other)
        actual_thw = gap / v_long
        expected_h = min(1.0, max(0.0, 1.0 - actual_thw / 1.5))
        clipped_h = min(1.0, max(0.0, 1.0 - (gap / 40.0) / 1.5))
        self.assertNotAlmostEqual(expected_h, clipped_h)
        self.assertAlmostEqual(info["reward_diagnostics"]["ego_longitudinal_speed"], v_long)
        self.assertAlmostEqual(info["reward_diagnostics"]["min_thw"], actual_thw)
        self.assertAlmostEqual(info["reward_terms"]["headway"], -0.4 * expected_h)
        self.assert_reward_sum(reward, info)

    def test_stopped_vehicle_has_finite_headway_without_target_cost(self):
        env, _, _ = self.make_project(speed_index=0)
        self.set_ego(env, speed=0.0)
        self.add_vehicle(env, dx=6.0, speed=0.0)
        _, reward, terminated, truncated, info = env.step(4)
        self.assertFalse(terminated or truncated)
        self.assertAlmostEqual(info["reward_diagnostics"]["min_thw"], 10.0)
        self.assertAlmostEqual(reward, 0.0)
        self.assertEqual(info["reward_terms"]["target_speed"], 0.0)
        self.assert_reward_sum(reward, info)

    def test_real_collision_and_overlap_are_terminal_with_negative_reward(self):
        env, _, _ = self.make_project(speed_index=0)
        ego = self.set_ego(env, speed=0.0)
        self.add_vehicle(env, dx=4.0, speed=0.0)
        _, reward, terminated, truncated, info = env.step(4)
        self.assertTrue(ego.crashed, "重叠夹具必须经真实碰撞检测产生 crashed")
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["collision"])
        self.assertIn("collision", info["failure_reasons"])
        self.assertAlmostEqual(info["reward_terms"]["failure"], -10.0)
        self.assertAlmostEqual(info["reward_terms"]["headway"], -0.4)
        self.assertAlmostEqual(reward, -10.4)
        self.assert_reward_sum(reward, info)
        with self.assertRaises(gym.error.ResetNeeded):
            env.step(4)

    def test_offroad_is_terminal_and_time_limit_failure_wins(self):
        for with_collision in (False, True):
            with self.subTest(simultaneous_collision=with_collision):
                env, _, _ = self.make_project(speed_index=0)
                ego = self.set_ego(env, speed=0.0, y=-10.0)
                if with_collision:
                    self.add_vehicle(env, dx=0.0, speed=0.0, y=-10.0)
                raw = env.simulator.unwrapped
                raw.time = 39.8
                raw.steps = 597
                _, reward, terminated, truncated, info = env.step(4)
                self.assertAlmostEqual(info["simulation_time"], 40.0)
                self.assertTrue(terminated)
                self.assertFalse(truncated, "失败和40秒同时发生时必须优先报告失败")
                self.assertFalse(ego.on_road)
                self.assertTrue(info["off_road"])
                self.assertIn("off_road", info["failure_reasons"])
                self.assertEqual(info["collision"], with_collision)
                self.assertEqual(ego.crashed, with_collision)
                self.assertAlmostEqual(info["reward_terms"]["failure"], -10.0)
                self.assert_reward_sum(reward, info)

    def test_empty_scene_reaches_true_forty_second_truncation(self):
        env, _, _ = self.make_project(speed_index=4)
        total = 0.0
        for index in range(200):
            _, reward, terminated, truncated, info = env.step(4)
            self.assert_reward_sum(reward, info)
            total += reward
            self.assertFalse(terminated)
            self.assertEqual(truncated, index == 199)
        self.assertAlmostEqual(info["simulation_time"], 40.0)
        self.assertEqual(info["physics_steps"], 600)
        self.assertEqual(info["episode_steps"], 200)
        self.assertAlmostEqual(info["episode_reward"], total)
        self.assertFalse(info["collision"] or info["off_road"])
        self.assertEqual(info["failure_reasons"], [])
        before = self.snapshot(env)
        with self.assertRaises(gym.error.ResetNeeded):
            env.step(4)
        self.assertEqual(self.snapshot(env), before)

    def test_reset_clears_history_pending_targets_and_cumulative_metrics(self):
        env, _, _ = self.make_project()
        env.step(0)
        env.step(8)  # 正在换道，触发一次可观察的 hold 回退。
        self.assertEqual(env.invalid_action_count, 1)
        old = env.simulator.unwrapped.vehicle
        obs, info = env.reset(seed=118, options={"initial_target_speed_index": 0})
        new = env.simulator.unwrapped.vehicle
        self.assertIsNot(new, old)
        self.assertEqual(new.target_lane_index, new.lane_index)
        self.assertEqual(new.target_speed, 0.0)
        self.assertEqual(new.speed_index, 0)
        self.assertFalse(env.get_debug_state()["lane_change_in_progress"])
        np.testing.assert_array_equal(obs[25:], np.zeros(9))
        for name in ("episode_reward", "episode_steps", "invalid_action_count"):
            self.assertEqual(getattr(env, name), 0)
            self.assertEqual(info[name], 0)
        self.assertEqual(info["simulation_time"], 0.0)
        self.assertEqual(info["physics_steps"], 0)
        env.step(5)

    def test_seed_and_legal_sequence_reproduce_full_project_outputs(self):
        first, first_obs, _ = self.make_project(seed=23, speed_index=3)
        second, second_obs, _ = self.make_project(seed=23, speed_index=3)
        np.testing.assert_array_equal(first_obs, second_obs)
        for step, action in enumerate([0] + [4] * 40 + [8] + [4] * 40):
            with self.subTest(step=step):
                self.assertTrue(first.get_action_mask()[action])
                one = first.step(action)
                two = second.step(action)
                np.testing.assert_array_equal(one[0], two[0])
                self.assertEqual(one[1:4], two[1:4])
                self.assertEqual(one[4]["reward_terms"], two[4]["reward_terms"])
                np.testing.assert_array_equal(one[4]["action_mask"], two[4]["action_mask"])
                np.testing.assert_array_equal(first.simulator.unwrapped.vehicle.position,
                                              second.simulator.unwrapped.vehicle.position)
                self.assertEqual(one[4]["episode_reward"], two[4]["episode_reward"])

    def test_returned_observations_masks_and_diagnostics_do_not_alias_state(self):
        env, obs, info = self.make_project()
        obs_before = obs.copy()
        mask_before = info["action_mask"].copy()
        raw_before = info["observation_debug"]["raw_values"].copy()
        next_obs, _, _, _, next_info = env.step(0)
        np.testing.assert_array_equal(obs, obs_before)
        np.testing.assert_array_equal(info["action_mask"], mask_before)
        np.testing.assert_array_equal(info["observation_debug"]["raw_values"], raw_before)
        stable_observation, _ = env.simulator.get_observation()
        expected_mask = env.get_action_mask()
        next_obs[:] = 0
        next_info["action_mask"][:] = False
        next_info["observation_debug"]["raw_values"][:] = -999
        actual_observation, _ = env.simulator.get_observation()
        np.testing.assert_array_equal(actual_observation, stable_observation)
        np.testing.assert_array_equal(env.get_action_mask(), expected_mask)

    def test_reset_seed_does_not_reseed_global_numpy_random(self):
        env, _, _ = self.make_project()
        before = np.random.get_state()
        env.reset(seed=999)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_rgb_render_close_repeat_and_recreate(self):
        env, _, _ = self.make_project(render_mode="rgb_array")
        env.step(4)
        image = env.render()
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.ndim, 3)
        self.assertEqual(image.shape[2], 3)
        self.assertEqual(image.dtype, np.dtype(np.uint8))
        self.assertGreater(image.shape[0], 0)
        self.assertGreater(image.shape[1], 0)
        env.close()
        env.close()
        with self.assertRaises(gym.error.ResetNeeded):
            env.step(4)
        another, _, _ = self.make_project()
        self.assert_observation(another, another.step(4)[0])

    def test_fixed_project_core_cannot_be_changed_by_reset_options(self):
        env, _, _ = self.make_project()
        for key, invalid_value in {
            "lanes_count": 4, "controlled_vehicles": 2, "duration": 80,
            "policy_frequency": 10, "simulation_frequency": 30,
            "offroad_terminal": False,
        }.items():
            with self.subTest(key=key):
                before = self.snapshot(env)
                with self.assertRaises(ValueError):
                    env.reset(options={"config": {key: invalid_value}})
                self.assertEqual(self.snapshot(env), before)

    def test_unknown_action_policy_and_render_mode_are_rejected(self):
        with self.assertRaises(ValueError):
            HighwayLaneEnv(invalid_action_mode="silently_ignore")
        with self.assertRaises(ValueError):
            HighwayLaneEnv(render_mode="invented")


if __name__ == "__main__":
    unittest.main()
