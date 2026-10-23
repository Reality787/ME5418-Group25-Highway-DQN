"""阶段 1A：直接检查真实 HighwayEnv，不构造虚假的观测或奖励。

依赖缺失时以“未验证”原因跳过；其他导入、API 或仿真错误正常报错。
所有场景均无窗口运行；原生动作编号不使用项目的九动作映射。
"""

import importlib
import math
import unittest

import config
from environment import make_stock_env


DEPENDENCY_ERROR = None
try:
    gym = importlib.import_module("gymnasium")
    np = importlib.import_module("numpy")
    importlib.import_module("highway_env")
    from highway_env.envs.common.action import DiscreteMetaAction
except ModuleNotFoundError as exc:
    DEPENDENCY_ERROR = f"未验证：缺少真实仿真依赖 {exc.name}"


@unittest.skipIf(DEPENDENCY_ERROR is not None, DEPENDENCY_ERROR or "")
class StockEnvironmentTests(unittest.TestCase):
    """保留原生五动作、观测和奖励，仅验证阶段 1A 的仿真连接。"""

    def make_env(self):
        """每个真实环境都登记清理，即使断言失败也释放资源。"""
        env = make_stock_env(render_mode=None)
        self.addCleanup(env.close)
        return env

    def test_create_and_reset(self):
        """真实创建并检查 Gymnasium 的二项 reset 返回值。"""
        env = self.make_env()
        result = env.reset(seed=0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        obs, info = result
        self.assertTrue(env.observation_space.contains(obs))
        self.assertIsInstance(info, dict)

    def test_step_observation_in_native_space(self):
        """从原生动作空间采样，后继观测仍属于实际原生空间。"""
        env = self.make_env()
        env.reset(seed=1)
        # reset 可能重建动作空间，因此在 reset 之后设置采样种子。
        env.action_space.seed(1)
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        self.assertTrue(env.observation_space.contains(obs))

    def test_native_action_space_matches_meta_actions(self):
        """核对实际 DiscreteMetaAction 接口，避免借用项目九动作编号。"""
        env = self.make_env()
        env.reset(seed=2)
        action_type = env.unwrapped.action_type
        self.assertIsInstance(action_type, DiscreteMetaAction)
        self.assertIsInstance(env.action_space, gym.spaces.Discrete)
        self.assertEqual(env.action_space.n, 5)
        self.assertEqual(env.action_space.n, len(action_type.actions))
        self.assertEqual(
            set(action_type.actions.values()),
            {"LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"},
        )
        self.assertEqual(set(action_type.actions), set(range(env.action_space.n)))
        for action_id, name in action_type.actions.items():
            self.assertTrue(env.action_space.contains(action_id))
            self.assertEqual(action_type.actions_indexes[name], action_id)

    def test_step_signature_reward_and_info(self):
        """真实 step 返回五项，奖励有限，终止标记和 info 类型正确。"""
        env = self.make_env()
        env.reset(seed=3)
        idle_action = env.unwrapped.action_type.actions_indexes["IDLE"]
        result = env.step(idle_action)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5)
        _, reward, terminated, truncated, info = result
        self.assertTrue(math.isfinite(reward))
        self.assertIsInstance(terminated, (bool, np.bool_))
        self.assertIsInstance(truncated, (bool, np.bool_))
        self.assertIsInstance(info, dict)

    def test_reset_seed_reproducible(self):
        """两次独立创建的环境用同一 seed 得到一致的初始观测。"""
        first = self.make_env()
        second = self.make_env()
        first_obs, _ = first.reset(seed=17)
        second_obs, _ = second.reset(seed=17)
        np.testing.assert_array_equal(first_obs, second_obs)

    def test_scene_configuration_is_effective(self):
        """同时检查配置、真实道路与车辆，以及一个决策间隔的推进。"""
        env = self.make_env()
        env.reset(seed=4)
        raw_env = env.unwrapped
        expected = {
            "lanes_count": config.LANE_COUNT,
            "vehicles_count": config.INITIAL_SURROUNDING_VEHICLE_COUNT,
            "controlled_vehicles": config.CONTROLLED_VEHICLE_COUNT,
            "duration": config.EPISODE_DURATION_SECONDS,
            "policy_frequency": config.POLICY_FREQUENCY_HZ,
            "simulation_frequency": config.SIMULATION_FREQUENCY_HZ,
            "offroad_terminal": True,
        }
        for key, value in expected.items():
            with self.subTest(parameter=key):
                self.assertEqual(raw_env.config[key], value)
        # 原生 highway-v0 由单段直路组成；直接计算实际创建的车道。
        lane_count = sum(
            len(lanes)
            for destinations in raw_env.road.network.graph.values()
            for lanes in destinations.values()
        )
        self.assertEqual(lane_count, config.LANE_COUNT)
        self.assertEqual(
            len(raw_env.controlled_vehicles), config.CONTROLLED_VEHICLE_COUNT
        )
        self.assertEqual(
            len(raw_env.road.vehicles) - len(raw_env.controlled_vehicles),
            config.INITIAL_SURROUNDING_VEHICLE_COUNT,
        )
        initial_time = raw_env.time
        initial_steps = raw_env.steps
        idle_action = raw_env.action_type.actions_indexes["IDLE"]
        env.step(idle_action)
        self.assertAlmostEqual(
            raw_env.time - initial_time, 1 / config.POLICY_FREQUENCY_HZ
        )
        self.assertEqual(
            raw_env.steps - initial_steps,
            config.SIMULATION_FREQUENCY_HZ // config.POLICY_FREQUENCY_HZ,
        )

    def test_empty_road_reaches_time_limit(self):
        """只覆盖独立测试实例的周围车辆数，真实运行到时间上限截断。"""
        env = self.make_env()
        env.unwrapped.configure({"vehicles_count": 0})
        env.reset(seed=5)
        self.assertEqual(len(env.unwrapped.road.vehicles), 1)
        idle_action = env.unwrapped.action_type.actions_indexes["IDLE"]
        duration = env.unwrapped.config["duration"]
        frequency = env.unwrapped.config["policy_frequency"]
        expected_decisions = math.ceil(duration * frequency)
        # 累加浮点时间可能晚一个决策间隔达到上限；循环始终有界。
        max_decisions = expected_decisions + 2
        terminated = truncated = False
        for decisions in range(1, max_decisions + 1):
            _, _, terminated, truncated, _ = env.step(idle_action)
            if terminated or truncated:
                break
        self.assertFalse(terminated, "无周围车辆且使用 IDLE 时不应碰撞或离路")
        self.assertTrue(truncated, "真实仿真未按持续时间返回 truncated")
        self.assertGreaterEqual(decisions, expected_decisions)
        self.assertLessEqual(decisions, expected_decisions + 1)
        self.assertGreaterEqual(env.unwrapped.time, duration)
        self.assertLess(env.unwrapped.time, duration + 1 / frequency + 1e-9)

        # 再建正式环境，确认局部测试覆盖没有污染默认 20 辆周围车辆。
        default_env = self.make_env()
        default_env.reset(seed=5)
        self.assertEqual(config.INITIAL_SURROUNDING_VEHICLE_COUNT, 20)
        self.assertEqual(default_env.unwrapped.config["vehicles_count"], 20)
        self.assertEqual(
            len(default_env.unwrapped.road.vehicles)
            - len(default_env.unwrapped.controlled_vehicles),
            20,
        )

    def test_close_and_recreate(self):
        """关闭一个实际环境后，另一个环境仍可创建、reset 和 step。"""
        first = self.make_env()
        first.reset(seed=6)
        first.step(first.unwrapped.action_type.actions_indexes["IDLE"])
        first.close()
        second = self.make_env()
        obs, _ = second.reset(seed=6)
        self.assertTrue(second.observation_space.contains(obs))
        result = second.step(second.unwrapped.action_type.actions_indexes["IDLE"])
        self.assertEqual(len(result), 5)
        self.assertTrue(second.observation_space.contains(result[0]))


if __name__ == "__main__":
    unittest.main()
