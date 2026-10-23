"""物理指标独立手算和真实正式环境检查，不以奖励代替驾驶进度。"""

import math
import unittest

from environment import make_env
from metrics import EpisodeMetrics, aggregate_metrics, is_better


class MetricsTests(unittest.TestCase):
    def create(self, *, speed_index=4):
        env = make_env(invalid_action_mode="raise")
        self.addCleanup(env.close)
        env.reset(seed=710, options={
            "config": {"vehicles_count": 0, "initial_lane_id": 1},
            "initial_target_speed_index": speed_index,
        })
        return env

    @staticmethod
    def record(*, distance, collision=False, off_road=False, completion=False,
               reward=0.0, stopped_fraction=0.0):
        return {
            "distance": distance, "normalized_progress": distance / 1112.0,
            "collision": collision, "off_road": off_road,
            "failure": collision or off_road, "completion": completion,
            "reward": reward, "stopped_fraction": stopped_fraction,
            "elapsed_seconds": 40.0 if completion else 20.0,
            "lane_change_count": 0, "speed_change_count": 0,
        }

    def test_aggregation_keeps_failed_episode_and_fixed_distance_denominator(self):
        records = [
            self.record(distance=1112.0, completion=True, reward=20.0),
            self.record(distance=278.0, collision=True, reward=-5.0),
        ]
        result = aggregate_metrics(records)
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["mean_distance"], 695.0)
        self.assertEqual(result["mean_normalized_progress"], 0.625)
        self.assertEqual(result["failure_rate"], 0.5)
        self.assertEqual(result["collision_rate"], 0.5)
        self.assertEqual(result["off_road_rate"], 0.0)
        self.assertEqual(result["completion_rate"], 0.5)
        self.assertEqual(result["mean_reward"], 7.5)
        self.assertEqual(result["validation_score"], -0.375)
        self.assertEqual(result["score"], result["validation_score"])
        self.assertEqual(records[1]["normalized_progress"], 0.25)

    def test_progress_above_one_and_negative_are_not_clipped(self):
        result = aggregate_metrics([
            self.record(distance=1668.0), self.record(distance=-278.0),
        ])
        self.assertEqual(result["mean_normalized_progress"], 0.625)
        self.assertEqual(aggregate_metrics([self.record(distance=1668.0)])["score"], 1.5)
        self.assertEqual(aggregate_metrics([self.record(distance=-278.0)])["score"], -0.25)

    def test_selection_rule_score_collision_then_earlier_checkpoint(self):
        best = {"validation_score": 0.4, "collision_rate": 0.2, "env_steps": 1000}
        self.assertTrue(is_better(best, None))
        self.assertTrue(is_better({**best, "validation_score": 0.5}, best))
        self.assertFalse(is_better({**best, "validation_score": 0.3, "collision_rate": 0}, best))
        self.assertTrue(is_better({**best, "collision_rate": 0.1, "env_steps": 2000}, best))
        self.assertFalse(is_better({**best, "collision_rate": 0.3, "env_steps": 500}, best))
        self.assertTrue(is_better({**best, "env_steps": 500}, best))
        self.assertFalse(is_better({**best, "env_steps": 2000}, best))
        self.assertFalse(is_better(best.copy(), best))

    def test_empty_and_nonfinite_records_rejected(self):
        with self.assertRaises(ValueError):
            aggregate_metrics([])
        with self.assertRaises(ValueError):
            aggregate_metrics([self.record(distance=math.nan)])

    def test_real_displacement_counts_and_no_premature_completion(self):
        env = self.create(speed_index=4)
        raw = env.simulator.unwrapped
        raw.vehicle.speed = 27.8  # 测试初态，之后每步都使用真实动力学。
        start_x = float(raw.vehicle.position[0])
        metrics = EpisodeMetrics(env)
        self.assertFalse(metrics.finish()["completion"])
        for _ in range(10):
            _, reward, terminated, truncated, info = env.step(4)
            metrics.update(env, reward, terminated, truncated, info)
        result = metrics.finish()
        self.assertAlmostEqual(result["distance"], 27.8 * 2, places=10)
        self.assertAlmostEqual(result["distance"], raw.vehicle.position[0] - start_x)
        self.assertAlmostEqual(result["normalized_progress"], 0.05)
        self.assertEqual(result["steps"], 10)
        self.assertEqual(result["physics_frames"], 30)
        self.assertEqual(result["elapsed_seconds"], 2.0)
        self.assertFalse(result["completion"])
        self.assertFalse(result["ended"])
        with self.assertRaises(ValueError):
            metrics.update(env, reward, terminated, truncated, info)

    def test_real_time_limit_completion(self):
        env = self.create(speed_index=4)
        env.simulator.unwrapped.vehicle.speed = 27.8
        metrics = EpisodeMetrics(env)
        for _ in range(201):
            _, reward, terminated, truncated, info = env.step(4)
            metrics.update(env, reward, terminated, truncated, info)
            if terminated or truncated:
                break
        result = metrics.finish()
        self.assertTrue(result["completion"])
        self.assertTrue(result["truncated"])
        self.assertFalse(result["terminated"])
        self.assertFalse(result["failure"])
        self.assertAlmostEqual(result["distance"], 1112.0, places=8)
        self.assertEqual(result["steps"], 200)
        with self.assertRaises(ValueError):
            metrics.update(env, reward, terminated, truncated, info)

    def test_failure_is_retained_and_not_completion(self):
        env = self.create()
        raw = env.simulator.unwrapped
        metrics = EpisodeMetrics(env)
        raw.vehicle.crashed = True  # 真实终止夹具：仍推进动力学并读取返回标记。
        _, reward, terminated, truncated, info = env.step(4)
        metrics.update(env, reward, terminated, truncated, info)
        result = metrics.finish()
        self.assertTrue(result["collision"])
        self.assertTrue(result["failure"])
        self.assertFalse(result["completion"])
        self.assertTrue(result["terminated"])
        self.assertGreater(result["distance"], 0.0)
        self.assertAlmostEqual(result["normalized_progress"], result["distance"] / 1112.0)

    def test_stop_fraction_and_actual_target_change_counts(self):
        env = self.create(speed_index=0)
        env.simulator.unwrapped.vehicle.speed = 0.0
        metrics = EpisodeMetrics(env)
        _, reward, terminated, truncated, info = env.step(4)
        metrics.update(env, reward, terminated, truncated, info)
        self.assertEqual(metrics.finish()["stopped_fraction"], 1.0)
        self.assertEqual(metrics.finish()["speed_change_count"], 0)
        _, reward, terminated, truncated, info = env.step(2)
        metrics.update(env, reward, terminated, truncated, info)
        _, reward, terminated, truncated, info = env.step(4)
        metrics.update(env, reward, terminated, truncated, info)
        result = metrics.finish()
        self.assertEqual(result["speed_change_count"], 1)
        self.assertEqual(result["lane_change_count"], 1)
        self.assertGreater(result["stopped_fraction"], 0.0)
        self.assertLess(result["stopped_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
