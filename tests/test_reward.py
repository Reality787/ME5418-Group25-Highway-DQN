"""奖励纯函数测试：预期值来自用户给定公式的独立手算。"""

import math
import unittest

from reward import compute_reward


class RewardTests(unittest.TestCase):
    def calculate(self, **overrides):
        values = dict(
            dt=0.2,
            longitudinal_displacement=5.56,
            ego_longitudinal_speed=20.0,
            front_bumper_gaps=(),
            failure=False,
            lane_change_started=False,
            target_speed_changed=False,
        )
        values.update(overrides)
        return compute_reward(**values)

    def assert_reward(self, expected, **overrides):
        reward, terms, diagnostics = self.calculate(**overrides)
        self.assertAlmostEqual(reward, expected, places=12)
        self.assertEqual(set(terms), {"progress", "headway", "failure", "lane_change", "target_speed"})
        self.assertEqual(sum(terms.values()), reward)
        self.assertTrue(all(math.isfinite(value) for value in terms.values()))
        return terms, diagnostics

    def test_full_progress_without_front_vehicle_is_point_two(self):
        terms, diagnostics = self.assert_reward(0.2)
        self.assertAlmostEqual(terms["progress"], 0.2)
        self.assertAlmostEqual(diagnostics["progress_fraction"], 1.0)
        self.assertEqual(diagnostics["headway_penalty"], 0.0)
        self.assertIsNone(diagnostics["min_thw"])

    def test_both_target_events_cost_point_zero_seven(self):
        terms, _ = self.assert_reward(0.13, lane_change_started=True, target_speed_changed=True)
        self.assertEqual(terms["lane_change"], -0.05)
        self.assertEqual(terms["target_speed"], -0.02)

    def test_half_threshold_headway_cancels_full_progress(self):
        # 15 m / 20 m/s = 0.75 s；h = 1 - 0.75 / 1.5 = 0.5。
        terms, diagnostics = self.assert_reward(0.0, front_bumper_gaps=(15.0,))
        self.assertEqual(diagnostics["min_thw"], 0.75)
        self.assertEqual(diagnostics["headway_penalty"], 0.5)
        self.assertAlmostEqual(terms["headway"], -0.2)

    def test_failure_is_one_cost_and_total_can_be_negative(self):
        terms, _ = self.assert_reward(-10.27, longitudinal_displacement=0.0,
                                     front_bumper_gaps=(15.0,), failure=True,
                                     lane_change_started=True, target_speed_changed=True)
        self.assertEqual(terms["failure"], -10.0)

    def test_no_implicit_survival_or_native_reward(self):
        terms, _ = self.assert_reward(0.0, longitudinal_displacement=0.0)
        self.assertTrue(all(value == 0 for value in terms.values()))

    def test_progress_clips_negative_and_above_desired_speed(self):
        for displacement, expected_fraction, expected_reward in ((-3.0, 0.0, 0.0),
                                                                 (2.78, 0.5, 0.1),
                                                                 (11.12, 1.0, 0.2)):
            with self.subTest(displacement=displacement):
                _, diagnostics = self.assert_reward(expected_reward, longitudinal_displacement=displacement)
                self.assertAlmostEqual(diagnostics["progress_fraction"], expected_fraction)

    def test_headway_threshold_and_distant_front_clip_to_zero(self):
        for gap in (30.0, 100.0):
            with self.subTest(gap=gap):
                _, diagnostics = self.assert_reward(0.2, front_bumper_gaps=(gap,))
                self.assertEqual(diagnostics["headway_penalty"], 0.0)

    def test_overlap_is_maximum_headway_penalty(self):
        _, diagnostics = self.assert_reward(-0.2, front_bumper_gaps=(0.0,))
        self.assertEqual(diagnostics["min_thw"], 0.0)
        self.assertEqual(diagnostics["headway_penalty"], 1.0)

    def test_zero_and_reverse_speed_use_point_one_denominator(self):
        for speed in (0.0, -3.0):
            with self.subTest(speed=speed):
                # 0.075 m / 0.1 m/s = 0.75 s；零速也不能出现除零。
                _, diagnostics = self.assert_reward(-0.2, longitudinal_displacement=0.0,
                                                     ego_longitudinal_speed=speed,
                                                     front_bumper_gaps=(0.075,))
                self.assertAlmostEqual(diagnostics["min_thw"], 0.75)
                self.assertAlmostEqual(diagnostics["headway_penalty"], 0.5)
        self.assert_reward(0.0, longitudinal_displacement=0.0, ego_longitudinal_speed=0.0)

    def test_minimum_thw_across_relevant_front_vehicles(self):
        _, diagnostics = self.assert_reward(0.0, front_bumper_gaps=(50.0, 15.0, 25.0))
        self.assertEqual(diagnostics["min_thw"], 0.75)

    def test_actual_dt_scales_continuous_terms_not_event_costs(self):
        # p=1，h=.5，连续项互相抵消；事件成本不乘dt。
        self.assert_reward(-0.07, dt=0.1, longitudinal_displacement=2.78,
                           front_bumper_gaps=(15.0,), lane_change_started=True,
                           target_speed_changed=True)
        self.assert_reward(0.1, dt=0.1, longitudinal_displacement=2.78)

    def test_invalid_time_is_not_replaced(self):
        for dt in (0.0, -0.1, math.nan, math.inf, -math.inf):
            with self.subTest(dt=dt):
                with self.assertRaises(ValueError):
                    self.calculate(dt=dt)

    def test_nonfinite_physical_values_and_negative_gap_rejected(self):
        for field in ("longitudinal_displacement", "ego_longitudinal_speed"):
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        self.calculate(**{field: value})
        for value in (-0.1, math.nan, math.inf, -math.inf):
            with self.subTest(gap=value):
                with self.assertRaises(ValueError):
                    self.calculate(front_bumper_gaps=(value,))

    def test_invalid_numeric_and_event_types_rejected(self):
        for field in ("dt", "longitudinal_displacement", "ego_longitudinal_speed"):
            for value in (True, "1", None):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(TypeError):
                        self.calculate(**{field: value})
        for field in ("failure", "lane_change_started", "target_speed_changed"):
            with self.subTest(field=field):
                with self.assertRaises(TypeError):
                    self.calculate(**{field: 1})
        with self.assertRaises(TypeError):
            self.calculate(front_bumper_gaps=("10",))

    def test_repeatable_without_input_mutation_or_shared_diagnostics(self):
        gaps = [50.0, 15.0, 25.0]
        first = self.calculate(front_bumper_gaps=gaps)
        second = self.calculate(front_bumper_gaps=gaps)
        self.assertEqual(first, second)
        self.assertEqual(gaps, [50.0, 15.0, 25.0])
        self.assertIsNot(first[1], second[1])
        self.assertIsNot(first[2], second[2])


if __name__ == "__main__":
    unittest.main()
