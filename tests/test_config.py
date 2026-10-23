"""使用标准库 unittest 核对用户给定的固定规格，不测试虚假的仿真结果。"""

import unittest

import config


class ConfigTests(unittest.TestCase):
    """这些检查只保证静态配置一致，不代表环境或 DQN 已经实现。"""

    def test_observation_dimensions(self):
        self.assertEqual(config.OBSERVATION_DIM, 34)
        self.assertEqual(config.TRAFFIC_FEATURE_DIM, 18)
        self.assertEqual(config.EGO_CONTROL_FEATURE_DIM, 7)
        self.assertEqual(config.PREVIOUS_ACTION_FEATURE_DIM, 9)
        self.assertEqual(
            config.OBSERVATION_DIM,
            config.TRAFFIC_FEATURE_DIM
            + config.EGO_CONTROL_FEATURE_DIM
            + config.PREVIOUS_ACTION_FEATURE_DIM,
        )

    def test_action_count(self):
        self.assertEqual(config.ACTION_COUNT, 9)
        self.assertEqual(len(config.ACTION_NAMES), 9)

    def test_action_mapping(self):
        expected = {
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
        self.assertEqual(config.ACTION_NAMES, expected)

    def test_target_speeds(self):
        expected = (0.00, 6.95, 13.90, 20.85, 27.80)
        self.assertEqual(len(config.TARGET_SPEEDS_MPS), 5)
        for actual, target in zip(config.TARGET_SPEEDS_MPS, expected):
            self.assertAlmostEqual(actual, target)
        self.assertEqual(
            config.SPEED_INDEX_DELTAS, {"slower": -1, "hold": 0, "faster": 1}
        )

    def test_frequencies(self):
        self.assertEqual(config.POLICY_FREQUENCY_HZ, 5)
        self.assertEqual(config.SIMULATION_FREQUENCY_HZ, 15)

    def test_network_and_gamma(self):
        self.assertEqual(config.NETWORK_INPUT_DIM, 34)
        self.assertEqual(config.HIDDEN_LAYER_SIZES, (128, 128))
        self.assertEqual(config.NETWORK_OUTPUT_DIM, 9)
        self.assertEqual(config.HIDDEN_ACTIVATION, "ReLU")
        self.assertAlmostEqual(config.GAMMA, 0.99)

    def test_scene_parameters(self):
        self.assertEqual(config.ROAD_LAYOUT, "straight")
        self.assertEqual(config.TRAFFIC_DIRECTION, "same_direction")
        self.assertEqual(config.LANE_COUNT, 3)
        self.assertEqual(config.CONTROLLED_VEHICLE_COUNT, 1)
        self.assertEqual(config.INITIAL_SURROUNDING_VEHICLE_COUNT, 20)
        self.assertEqual(config.EPISODE_DURATION_SECONDS, 40)
        self.assertAlmostEqual(config.DESIRED_SPEED_MPS, 27.80)
        self.assertEqual(config.PERCEPTION_RANGE_METERS, 100)
        self.assertEqual(config.TERMINATION_EVENTS, ("collision", "off_road"))
        self.assertEqual(config.TRUNCATION_EVENTS, ("time_limit",))


if __name__ == "__main__":
    unittest.main()
