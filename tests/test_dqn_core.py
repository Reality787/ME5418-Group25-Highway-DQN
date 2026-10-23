"""普通 DQN 数值/状态测试；不以随机驾驶回报替代学习正确性。"""

from copy import deepcopy
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

from agent import DQNAgent
from replay_buffer import ReplayBuffer


# 测试夹具缩小容量和同步周期；正式默认值集中在 config.DQN_CONFIG。
TEST_SETTINGS = {
    "learning_rate": 1e-4, "batch_size": 4, "replay_capacity": 16,
    "learning_starts": 0, "train_every": 4, "target_update_every": 2,
    "gamma": 0.99, "gradient_norm_clip": 10.0,
    "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay_fraction": 0.6,
}


def make_batch(size=4):
    generator = np.random.default_rng(98)
    return {
        "obs": generator.normal(size=(size, 34)).astype(np.float32),
        "action": np.arange(size, dtype=np.int64) % 9,
        "reward": np.arange(size, dtype=np.float32) + 0.5,
        "next_obs": generator.normal(size=(size, 34)).astype(np.float32),
        "terminated": np.zeros(size, dtype=np.bool_),
        "truncated": np.zeros(size, dtype=np.bool_),
        "next_action_mask": np.ones((size, 9), dtype=np.bool_),
    }


def add_transition(buffer, index=0):
    buffer.add(np.full(34, index, dtype=np.float32), index % 9, float(index),
               np.full(34, index + 1, dtype=np.float32), False, False, np.ones(9, dtype=np.bool_))


class ReplayBufferTests(unittest.TestCase):
    def test_capacity_ring_retains_newest(self):
        buffer = ReplayBuffer(3, seed=7)
        for index in range(5):
            add_transition(buffer, index)
        self.assertEqual(len(buffer), 3)
        batch = buffer.sample(3)
        self.assertEqual(set(batch["reward"].tolist()), {2.0, 3.0, 4.0})
        self.assertEqual(batch["obs"].shape, (3, 34))
        self.assertEqual(batch["next_action_mask"].dtype, np.bool_)

    def test_input_and_sample_are_independent_copies(self):
        buffer = ReplayBuffer(3)
        observation = np.zeros(34, dtype=np.float32)
        mask = np.ones(9, dtype=np.bool_)
        buffer.add(observation, 4, 1.0, observation, False, True, mask)
        observation[:] = 77
        mask[:] = False
        sample = buffer.sample(1)
        np.testing.assert_array_equal(sample["obs"], np.zeros((1, 34)))
        self.assertTrue(sample["next_action_mask"].all())
        sample["obs"][:] = 90
        sample["next_action_mask"][:] = False
        self.assertEqual(float(buffer.sample(1)["obs"][0, 0]), 0)
        self.assertTrue(buffer.sample(1)["next_action_mask"].all())

    def test_reproducible_independent_sampling(self):
        left, right = ReplayBuffer(12, 4), ReplayBuffer(12, 4)
        for index in range(12):
            add_transition(left, index)
            add_transition(right, index)
        global_state = deepcopy(np.random.get_state())
        for _ in range(3):
            a, b = left.sample(5), right.sample(5)
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
        for old, new in zip(global_state, np.random.get_state()):
            np.testing.assert_array_equal(old, new)

    def test_insufficient_sampling_rejected_without_rng_change(self):
        buffer = ReplayBuffer(3)
        before = deepcopy(buffer.rng.bit_generator.state)
        for value in (0, -1, 1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                buffer.sample(value)
        self.assertEqual(before, buffer.rng.bit_generator.state)

    def test_termination_and_truncation_are_preserved(self):
        buffer = ReplayBuffer(3)
        buffer.add(np.zeros(34), 4, -10, np.ones(34), True, False, np.zeros(9, dtype=bool))
        buffer.add(np.zeros(34), 4, 0.2, np.ones(34), False, True, np.ones(9, dtype=bool))
        batch = buffer.sample(2)
        rows = np.argsort(batch["reward"])
        np.testing.assert_array_equal(batch["terminated"][rows], [True, False])
        np.testing.assert_array_equal(batch["truncated"][rows], [False, True])

    def test_invalid_add_is_atomic(self):
        buffer = ReplayBuffer(3)
        args = [np.zeros(34), 4, 1, np.zeros(34), False, False, np.ones(9, dtype=bool)]
        for index, bad_value in ((0, np.zeros(33)), (1, 9), (1, True), (2, np.inf),
                                 (4, 1), (6, np.zeros(9, dtype=bool)), (6, np.ones(9))):
            invalid = args.copy()
            invalid[index] = bad_value
            with self.subTest(index=index, bad=str(bad_value)), self.assertRaises((ValueError, TypeError)):
                buffer.add(*invalid)
            self.assertEqual(len(buffer), 0)
            self.assertEqual(buffer._position, 0)

    def test_safe_serialization_and_next_sample_resume(self):
        source = ReplayBuffer(5, seed=9)
        for index in range(8):
            add_transition(source, index)
        source.sample(2)
        payload = io.BytesIO()
        torch.save(source.state_dict(), payload)
        payload.seek(0)
        restored = ReplayBuffer(5, seed=800)
        restored.load_state_dict(torch.load(payload, weights_only=True, map_location="cpu"))
        for _ in range(3):
            a, b = source.sample(3), restored.sample(3)
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
        add_transition(source, 99)
        add_transition(restored, 99)
        np.testing.assert_array_equal(source.sample(5)["obs"], restored.sample(5)["obs"])

    def test_loading_incompatible_capacity_rejected(self):
        with self.assertRaisesRegex(ValueError, "容量"):
            ReplayBuffer(4).load_state_dict(ReplayBuffer(5).state_dict())


class DQNAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def make_agent(self, seed=3):
        return DQNAgent(seed=seed, device="cpu", settings=TEST_SETTINGS)

    def constant_outputs(self, network, values):
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
            network.layers[-1].bias.copy_(torch.tensor(values, dtype=torch.float32))

    def test_initial_target_independent_equal_frozen(self):
        before = torch.get_rng_state().clone()
        agent = self.make_agent()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for online, target in zip(agent.online.parameters(), agent.target.parameters()):
            self.assertTrue(torch.equal(online, target))
            self.assertNotEqual(online.data_ptr(), target.data_ptr())
            self.assertFalse(target.requires_grad)
        self.assertFalse(agent.target.training)
        second = self.make_agent()
        for left, right in zip(agent.online.parameters(), second.online.parameters()):
            self.assertTrue(torch.equal(left, right))

    def test_masked_exploration_only_legal_reproducible(self):
        left, right = self.make_agent(), self.make_agent()
        mask = np.array([False, True, False, False, True, False, False, False, True])
        a = [left.select_action(np.zeros(34), mask, 1.0) for _ in range(200)]
        b = [right.select_action(np.zeros(34), mask, 1.0) for _ in range(200)]
        self.assertEqual(a, b)
        self.assertEqual(set(a), {1, 4, 8})

    def test_greedy_ignores_illegal_high_q_and_preserves_rng(self):
        agent = self.make_agent()
        self.constant_outputs(agent.online, [1e6, 2, 5, 4, 3, 1e6, 0, 1e6, 0])
        mask = np.array([False, True, True, True, True, False, False, False, False])
        before = deepcopy(agent.rng.bit_generator.state)
        self.assertEqual(agent.select_action(np.zeros(34), mask, 0), 2)
        self.assertEqual(before, agent.rng.bit_generator.state)

    def test_action_empty_mask_and_bad_inputs_rejected(self):
        agent = self.make_agent()
        with self.assertRaisesRegex(ValueError, "不能为空"):
            agent.select_action(np.zeros(34), np.zeros(9, dtype=bool), 1)
        with self.assertRaises(ValueError):
            agent.select_action(np.zeros(34), np.ones(9), 0)
        with self.assertRaises(ValueError):
            agent.select_action(np.zeros(33), np.ones(9, dtype=bool), 0)
        with self.assertRaises(ValueError):
            agent.select_action(np.zeros(34), np.ones(9, dtype=bool), np.nan)

    def test_hand_calculated_td_targets_terminal_truncation_masks(self):
        agent = self.make_agent()
        # 非法动作 Q=1000 不能进入 max；截断样本使用目标网络的4。
        self.constant_outputs(agent.target, [1000, 2, 4, 0, 0, 0, 0, 0, 0])
        batch = make_batch(3)
        batch["reward"][:] = [7, 2, 3]
        batch["terminated"][:] = [True, False, False]
        batch["truncated"][:] = [False, True, False]
        batch["next_action_mask"][:] = False
        batch["next_action_mask"][1, [1, 2]] = True
        batch["next_action_mask"][2, 1] = True
        targets = agent.td_targets(batch)
        np.testing.assert_allclose(targets.cpu().numpy(), [7, 2 + .99 * 4, 3 + .99 * 2], rtol=1e-6)
        self.assertEqual(targets.shape, (3,))
        self.assertFalse(targets.requires_grad)

    def test_all_terminal_empty_masks_skip_target_and_stay_finite(self):
        agent = self.make_agent()
        batch = make_batch()
        batch["terminated"][:] = True
        batch["next_action_mask"][:] = False
        with patch.object(agent.target, "forward", side_effect=AssertionError("不得计算终止后 Q")):
            targets = agent.td_targets(batch)
            result = agent.learn(batch)
        np.testing.assert_array_equal(targets.cpu().numpy(), batch["reward"])
        self.assertTrue(np.isfinite(result["loss"]))

    def test_nonterminal_empty_mask_rejected(self):
        agent = self.make_agent()
        batch = make_batch()
        batch["next_action_mask"][1] = False
        batch["truncated"][1] = True
        with self.assertRaisesRegex(ValueError, "空掩码"):
            agent.td_targets(batch)

    def test_batch_shape_checks_prevent_broadcasting(self):
        agent = self.make_agent()
        for key in ("reward", "action", "terminated", "truncated"):
            batch = make_batch()
            batch[key] = batch[key][:, None]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "形状"):
                agent.learn(batch)

    def test_update_parameters_finite_gradients_and_sync_timing(self):
        agent = self.make_agent()
        before = [value.detach().clone() for value in agent.online.parameters()]
        old_target = [value.detach().clone() for value in agent.target.parameters()]
        first = agent.learn(make_batch())
        self.assertEqual(first["gradient_steps"], 1)
        self.assertFalse(first["target_synced"])
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, agent.online.parameters())))
        for old, target in zip(old_target, agent.target.parameters()):
            self.assertTrue(torch.equal(old, target))
            self.assertIsNone(target.grad)
        for parameter in agent.online.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        second = agent.learn(make_batch())
        self.assertEqual(second["gradient_steps"], 2)
        self.assertTrue(second["target_synced"])
        for online, target in zip(agent.online.parameters(), agent.target.parameters()):
            self.assertTrue(torch.equal(online, target))
            self.assertIsNone(target.grad)
        for key in ("loss", "q_mean", "target_mean", "grad_norm"):
            self.assertTrue(np.isfinite(second[key]))

    def test_insufficient_replay_does_not_update(self):
        agent = self.make_agent()
        for index in range(3):
            add_transition(agent.replay, index)
        self.assertIsNone(agent.learn())
        self.assertEqual(agent.gradient_steps, 0)
        add_transition(agent.replay, 3)
        self.assertEqual(agent.learn()["gradient_steps"], 1)

    def test_fixed_terminal_regression_error_decreases(self):
        agent = self.make_agent()
        batch = make_batch()
        batch["terminated"][:] = True
        losses = [agent.learn(batch)["loss"] for _ in range(100)]
        self.assertLess(losses[-1], losses[0] * 0.5)

    def test_full_safe_checkpoint_resume_action_sample_update(self):
        source = self.make_agent(32)
        for index in range(12):
            add_transition(source.replay, index)
        source.learn()
        source.select_action(np.zeros(34), np.ones(9, dtype=bool), 1)
        payload = io.BytesIO()
        torch.save(source.state_dict(), payload)
        payload.seek(0)
        restored = self.make_agent(999)
        restored.load_state_dict(torch.load(payload, weights_only=True, map_location="cpu"))
        self.assertEqual(source.gradient_steps, restored.gradient_steps)
        for _ in range(5):
            self.assertEqual(source.select_action(np.zeros(34), np.ones(9, dtype=bool), .8),
                             restored.select_action(np.zeros(34), np.ones(9, dtype=bool), .8))
        left, right = source.replay.sample(4), restored.replay.sample(4)
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
        self.assertEqual(source.learn(), restored.learn())
        for a, b in zip(source.online.parameters(), restored.online.parameters()):
            self.assertTrue(torch.equal(a, b))
        for a, b in zip(source.target.parameters(), restored.target.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_checkpoint_config_mismatch_rejected(self):
        agent = self.make_agent()
        state = agent.state_dict()
        state["settings"]["gamma"] = .5
        with self.assertRaisesRegex(ValueError, "不兼容"):
            agent.load_state_dict(state)

    def test_action_and_learning_do_not_mutate_input_arrays(self):
        agent = self.make_agent()
        observation = np.zeros(34, dtype=np.float32)
        mask = np.ones(9, dtype=bool)
        agent.select_action(observation, mask, .2)
        np.testing.assert_array_equal(observation, np.zeros(34))
        self.assertTrue(mask.all())
        batch = make_batch()
        before = {key: value.copy() for key, value in batch.items()}
        agent.learn(batch)
        for key in batch:
            np.testing.assert_array_equal(batch[key], before[key])


if __name__ == "__main__":
    unittest.main()
