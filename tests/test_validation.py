"""独立验证使用真实正式场景，不能污染训练参数、回放、场景或随机流。"""

import copy
import unittest

import numpy as np
import torch

from agent import DQNAgent
from environment import make_env
from scenarios import ScenarioSampler, reset_scenario
from validation import validate_agent


class ValidationTests(unittest.TestCase):
    def assert_nested_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (tuple, list)):
            self.assertEqual(len(first), len(second))
            for one, two in zip(first, second):
                self.assert_nested_equal(one, two)
        else:
            self.assertEqual(first, second)

    def agent(self):
        # 小批量仅用于隔离测试；正式训练参数由统一配置冻结。
        return DQNAgent(seed=17, settings={
            "learning_rate": 1e-4, "batch_size": 2, "replay_capacity": 16,
            "learning_starts": 0, "train_every": 4, "target_update_every": 1000,
            "gamma": 0.99, "gradient_norm_clip": 10.0,
            "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay_fraction": 0.6,
        })

    def test_validation_preserves_complete_agent_training_env_and_random_streams(self):
        agent = self.agent()
        env = make_env(invalid_action_mode="raise")
        self.addCleanup(env.close)
        sampler = ScenarioSampler(split="train", seed=41)
        scenario = sampler.next()
        obs, info = reset_scenario(env, scenario)
        for _ in range(2):
            action = agent.select_action(obs, info["action_mask"], epsilon=1.0)
            nxt, reward, terminated, truncated, following = env.step(action)
            agent.replay.add(obs, action, reward, nxt, terminated, truncated, following["action_mask"])
            obs, info = nxt, following
            self.assertFalse(terminated or truncated)
        agent.learn()
        agent.online.train(True)
        before_state = copy.deepcopy(agent.state_dict())
        before_modes = (agent.online.training, agent.target.training)
        before_np = copy.deepcopy(np.random.get_state())
        control_sampler = copy.deepcopy(sampler)
        before_world = {
            "time": env.simulator.unwrapped.time,
            "frames": env.simulator.unwrapped.steps,
            "obs": env.get_observation()[0].copy(),
            "mask": env.get_action_mask(),
            "rng": copy.deepcopy(env.simulator.unwrapped.np_random.bit_generator.state),
        }
        result = validate_agent(agent, episodes=2, seed=0)
        self.assert_nested_equal(before_state, agent.state_dict())
        self.assertEqual(before_modes, (agent.online.training, agent.target.training))
        self.assert_nested_equal(before_np, np.random.get_state())
        self.assertEqual(sampler.next(), control_sampler.next())
        self.assertEqual(env.simulator.unwrapped.time, before_world["time"])
        self.assertEqual(env.simulator.unwrapped.steps, before_world["frames"])
        np.testing.assert_array_equal(env.get_observation()[0], before_world["obs"])
        np.testing.assert_array_equal(env.get_action_mask(), before_world["mask"])
        self.assertEqual(env.simulator.unwrapped.np_random.bit_generator.state, before_world["rng"])
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(len(result["episode_records"]), 2)
        self.assertAlmostEqual(result["score"], result["mean_normalized_progress"] - 2 * result["failure_rate"])
        for record in result["episode_records"]:
            self.assertTrue(record["ended"])
            self.assertLessEqual(record["steps"], 201)
            self.assertAlmostEqual(record["normalized_progress"], record["distance"] / 1112.0)

    def test_repeated_validation_uses_identical_scenarios_and_results(self):
        agent = self.agent()
        first = validate_agent(agent, episodes=2, seed=19)
        # 消费训练探索随机流不能改变固定贪心验证；此操作不更新网络。
        obs = np.zeros(34, dtype=np.float32)
        mask = np.ones(9, dtype=np.bool_)
        for _ in range(11):
            agent.select_action(obs, mask, epsilon=1.0)
        second = validate_agent(agent, episodes=2, seed=19)
        self.assert_nested_equal(first, second)

    def test_invalid_validation_options_rejected(self):
        agent = self.agent()
        for invalid in (0, -1, 1.2, True):
            with self.subTest(episodes=invalid), self.assertRaises(ValueError):
                validate_agent(agent, episodes=invalid)
        for invalid in (-1, 1.2, True):
            with self.subTest(seed=invalid), self.assertRaises(ValueError):
                validate_agent(agent, seed=invalid)


if __name__ == "__main__":
    unittest.main()
