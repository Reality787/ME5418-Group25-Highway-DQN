"""场景分区、真实参数生效和初始化几何检查；不读取最终测试。"""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np

import config
from environment import make_env
from scenarios import ScenarioSampler, is_withheld, reset_scenario


class ScenarioTests(unittest.TestCase):
    def make_scenario(self, split="development", seed=5):
        scenario = ScenarioSampler(split, seed).next()
        scenario["parameters"] = {"density": 1.0, "initial_spacing_m": 25.0,
                                  "traffic_speed_mps": 22.0}
        return scenario

    def make_environment(self):
        env = make_env(invalid_action_mode="raise")
        self.addCleanup(env.close)
        return env

    @staticmethod
    def offsets(info, lane):
        return sorted(row["relative_longitudinal_position_m"]
                      for row in info["scenario"]["initial_summary"]["vehicles"]
                      if not row["is_ego"] and row["lane"] == lane)

    def test_parameters_randomized_and_withheld_excluded(self):
        for split in ("development", "train", "validation"):
            sampler = ScenarioSampler(split, 11)
            scenarios = [sampler.next() for _ in range(500)]
            self.assertTrue(all(not is_withheld(s["parameters"]) for s in scenarios))
            for name, bounds in (("density", (0.8, 1.2)),
                                 ("initial_spacing_m", (20, 35)),
                                 ("traffic_speed_mps", (18, 28))):
                values = [s["parameters"][name] for s in scenarios]
                self.assertGreater(max(values) - min(values), .8 * (bounds[1] - bounds[0]))
                self.assertTrue(all(bounds[0] <= value <= bounds[1] for value in values))
            json.dumps(scenarios)

    def test_namespaces_and_runs_do_not_overlap(self):
        collections = []
        for split in ("development", "train", "validation"):
            for seed in (11, 22, 33):
                sampler = ScenarioSampler(split, seed)
                collections.append({sampler.next()["env_seed"] for _ in range(20)})
        self.assertEqual(len(set.union(*collections)), 9 * 20)
        final_namespace = config.SCENARIO_CONFIG["seed_namespaces"]["final-test"]
        stride = config.SCENARIO_CONFIG["namespace_stride"]
        self.assertTrue(all(seed // stride != final_namespace
                            for collection in collections for seed in collection))

    def test_sampler_snapshot_restores_exact_sequence_and_is_copy(self):
        original = ScenarioSampler("train", 11)
        original.next()
        state = original.state_dict()
        resumed = ScenarioSampler("train", 11)
        resumed.load_state_dict(state)
        self.assertEqual([original.next() for _ in range(5)],
                         [resumed.next() for _ in range(5)])
        before = original.state_dict()
        state["rng_state"]["state"]["state"] = 0
        self.assertEqual(before, original.state_dict())
        with self.assertRaises(ValueError):
            ScenarioSampler("train", 22).load_state_dict(before)

    def test_seed_limits_and_final_test_are_rejected(self):
        for seed in (-1, 1_000_000):
            with self.assertRaises(ValueError):
                ScenarioSampler("train", seed)
        with self.assertRaises(TypeError):
            ScenarioSampler("train", True)
        with self.assertRaises(ValueError):
            ScenarioSampler("final-test", 0)
        sampler = ScenarioSampler("train", 0)
        sampler.cursor = 1_000_000
        with self.assertRaises(ValueError):
            sampler.next()

    def test_fixed_validation_is_independent_from_training(self):
        train = ScenarioSampler("train", 11)
        state = train.state_dict()
        first = ScenarioSampler("validation", 0)
        first_set = [first.next() for _ in range(8)]
        for scenario in first_set:
            env = self.make_environment()
            reset_scenario(env, scenario)
            env.step(4)
        second = ScenarioSampler("validation", 0)
        self.assertEqual(first_set, [second.next() for _ in range(8)])
        self.assertEqual(state, train.state_dict())

    def test_reset_returns_real_observation_mask_and_no_history(self):
        env = self.make_environment()
        scenario = self.make_scenario()
        with patch.object(env.simulator, "step", side_effect=AssertionError("reset不能step")):
            obs, info = reset_scenario(env, scenario)
        self.assertEqual(obs.shape, (34,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(env.action_space.n, 9)
        self.assertTrue(np.isfinite(obs).all())
        np.testing.assert_array_equal(obs, env.get_observation()[0])
        np.testing.assert_array_equal(info["action_mask"], env.get_action_mask())
        np.testing.assert_array_equal(obs[25:], np.zeros(9))
        self.assertEqual(info["simulation_time"], 0)
        self.assertEqual(info["physics_steps"], 0)
        self.assertEqual(info["reward_terms"], dict.fromkeys(
            ["progress", "headway", "failure", "lane_change", "target_speed"], 0))
        _, reward, _, _, next_info = env.step(4)
        self.assertAlmostEqual(reward, sum(next_info["reward_terms"].values()))
        self.assertAlmostEqual(next_info["simulation_time"], .2)
        self.assertEqual(next_info["executed_action"], 4)
        self.assertFalse(next_info["invalid_action"])

    def test_does_not_modify_initial_ego_controller(self):
        scenario = self.make_scenario()
        before_env, after_env = self.make_environment(), self.make_environment()
        before_env.reset(seed=scenario["env_seed"], options={"config": {
            "vehicles_count": 20, "vehicles_density": 1.0, "initial_lane_id": None}})
        before = before_env.simulator.unwrapped.vehicle
        reset_scenario(after_env, scenario)
        after = after_env.simulator.unwrapped.vehicle
        np.testing.assert_array_equal(before.position, after.position)
        self.assertEqual(before.speed, after.speed)
        self.assertEqual(before.target_speed, after.target_speed)
        self.assertEqual(before.speed_index, after.speed_index)
        self.assertEqual(before.target_lane_index, after.target_lane_index)
        self.assertEqual(before_env.controller.get_low_level_parameters(),
                         after_env.controller.get_low_level_parameters())

    def test_density_changes_platoon_spacing_not_first_offset_or_speeds(self):
        env = self.make_environment()
        low, high = self.make_scenario(), self.make_scenario()
        low["parameters"]["density"] = .8
        high["parameters"]["density"] = 1.2
        _, a = reset_scenario(env, low)
        _, b = reset_scenario(env, high)
        for lane in range(3):
            front_a = [x for x in self.offsets(a, lane) if x > 0]
            front_b = [x for x in self.offsets(b, lane) if x > 0]
            self.assertAlmostEqual(front_a[0], 25)
            self.assertAlmostEqual(front_b[0], 25)
            # 独立比例：同jitter下后续中心间隔与density成反比。
            np.testing.assert_allclose(np.diff(front_a) / np.diff(front_b), 1.2 / .8)
        for x, y in zip(a["scenario"]["initial_summary"]["vehicles"],
                        b["scenario"]["initial_summary"]["vehicles"]):
            self.assertEqual(x["speed_mps"], y["speed_mps"])

    def test_initial_spacing_changes_nearest_offsets_not_platoon_intervals(self):
        env = self.make_environment()
        near, far = self.make_scenario(), self.make_scenario()
        near["parameters"]["initial_spacing_m"] = 20.
        far["parameters"]["initial_spacing_m"] = 35.
        _, a = reset_scenario(env, near)
        _, b = reset_scenario(env, far)
        for lane in range(3):
            front_a = [x for x in self.offsets(a, lane) if x > 0]
            front_b = [x for x in self.offsets(b, lane) if x > 0]
            self.assertAlmostEqual(front_a[0], 20)
            self.assertAlmostEqual(front_b[0], 35)
            np.testing.assert_allclose(np.diff(front_a), np.diff(front_b))
            np.testing.assert_allclose(np.asarray(front_b) - front_a, 15)

    def test_traffic_speed_changes_real_speed_and_target(self):
        env = self.make_environment()
        slow, fast = self.make_scenario(), self.make_scenario()
        slow["parameters"]["traffic_speed_mps"] = 18
        fast["parameters"]["traffic_speed_mps"] = 28
        _, a = reset_scenario(env, slow)
        _, b = reset_scenario(env, fast)
        for x, y in zip(a["scenario"]["initial_summary"]["vehicles"],
                        b["scenario"]["initial_summary"]["vehicles"]):
            np.testing.assert_array_equal(x["position_m"], y["position_m"])
            if not x["is_ego"]:
                self.assertAlmostEqual(y["speed_mps"] - x["speed_mps"], 10)
                self.assertEqual(y["speed_mps"], y["target_speed_mps"])
                self.assertTrue(26 <= y["speed_mps"] <= 30)
        self.assertTrue(all(v.target_lane_index == v.lane_index
                            for v in env.simulator.unwrapped.road.vehicles))

    def test_initial_geometry_count_and_config_for_varied_scenes(self):
        env = self.make_environment()
        sampler = ScenarioSampler("development", 9)
        for _ in range(15):
            _, info = reset_scenario(env, sampler.next())
            summary = info["scenario"]["initial_summary"]
            self.assertEqual(summary["vehicle_count"], 21)
            self.assertEqual(summary["surrounding_vehicle_count"], 20)
            self.assertGreater(summary["minimum_same_lane_bumper_gap_m"], 0)
            rows = summary["vehicles"]
            # 独立轴对齐矩形判断：航向为零，不能只相信summary里写的有效性结论。
            for i, a in enumerate(rows):
                for b in rows[i + 1:]:
                    dx, dy = np.abs(np.asarray(a["position_m"]) - b["position_m"])
                    self.assertTrue(dx >= 5 or dy >= 2)
            raw = env.simulator.unwrapped
            self.assertEqual(raw.config["lanes_count"], 3)
            self.assertEqual(raw.config["controlled_vehicles"], 1)
            self.assertEqual(raw.config["vehicles_count"], 20)
            self.assertEqual(raw.config["duration"], 40)
            env.step(4)

    def test_known_raw_observation_matches_real_initial_geometry(self):
        env = self.make_environment()
        _, info = reset_scenario(env, self.make_scenario())
        rows = info["scenario"]["initial_summary"]["vehicles"]
        raw_values = info["observation_debug"]["raw_values"]
        for lane in range(3):
            for side in range(2):
                candidates = [row for row in rows if not row["is_ego"] and row["lane"] == lane
                              and ((row["relative_longitudinal_position_m"] >= 0) == (side == 0))]
                nearest = min(candidates, key=lambda row: abs(row["relative_longitudinal_position_m"]))
                offset = lane * 6 + side * 3
                self.assertEqual(raw_values[offset], 1)
                self.assertAlmostEqual(raw_values[offset + 1], 25 - 2.5 - 2.5)
                self.assertAlmostEqual(raw_values[offset + 2], nearest["speed_mps"] - 25)

    def test_invalid_or_withheld_scenario_rejected_before_reset(self):
        env = self.make_environment()
        scenario = self.make_scenario()
        reset_scenario(env, scenario)
        env.step(4)
        before = env.get_observation()[0]
        before_time = env.simulator.unwrapped.time
        for key, value in (("parameters", {"density": 1.1, "initial_spacing_m": 23,
                                            "traffic_speed_mps": 26}),
                           ("env_seed", 0), ("split", "final-test")):
            bad = deepcopy(scenario)
            bad[key] = value
            with self.assertRaises(ValueError):
                reset_scenario(env, bad)
            np.testing.assert_array_equal(before, env.get_observation()[0])
            self.assertEqual(before_time, env.simulator.unwrapped.time)

    def test_repeat_scenario_and_action_sequence_is_reproducible(self):
        first, second = self.make_environment(), self.make_environment()
        scenario = ScenarioSampler("train", 22).next()
        one, info_one = reset_scenario(first, scenario)
        two, info_two = reset_scenario(second, scenario)
        np.testing.assert_array_equal(one, two)
        self.assertEqual(info_one["scenario"], info_two["scenario"])
        rng = np.random.default_rng(33)
        for _ in range(10):
            action = int(rng.choice(np.flatnonzero(first.get_action_mask())))
            one, reward_a, term_a, trunc_a, _ = first.step(action)
            two, reward_b, term_b, trunc_b, _ = second.step(action)
            np.testing.assert_array_equal(one, two)
            self.assertEqual((reward_a, term_a, trunc_a), (reward_b, term_b, trunc_b))
            if term_a or trunc_a:
                break

    def test_global_rng_and_input_scenario_unchanged(self):
        env = self.make_environment()
        scenario = self.make_scenario()
        original = deepcopy(scenario)
        np.random.seed(789)
        before = np.random.get_state()
        reset_scenario(env, scenario)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual(scenario, original)


if __name__ == "__main__":
    unittest.main()
