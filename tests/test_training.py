"""训练协议与检查点：真实短回合验证，不用mock环境冒充恢复成功。"""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import config
from agent import DQNAgent
from checkpoints import (atomic_save, compatibility_record, inference_payload,
                         load_checked, load_inference, signature)
from train import epsilon_at, run_training


class TrainingCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.compatibility = compatibility_record()
        cls.settings = {
            "learning_starts": 16, "batch_size": 8, "replay_capacity": 512,
            "target_update_every": 5,
        }

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertIsInstance(right, torch.Tensor)
            self.assertEqual(left.dtype, right.dtype)
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                with self.subTest(field=key):
                    self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_epsilon_uses_fixed_total_environment_budget(self):
        budget = 100000
        self.assertEqual(epsilon_at(0, budget), 1.0)
        self.assertAlmostEqual(epsilon_at(30000, budget), .525)
        self.assertAlmostEqual(epsilon_at(60000, budget), .05)
        self.assertAlmostEqual(epsilon_at(budget, budget), .05)
        self.assertAlmostEqual(epsilon_at(budget + 1, budget), .05)

    def test_compatibility_rejects_feature_reward_dynamics_and_actions(self):
        agent = DQNAgent(seed=15, settings=self.settings)
        payload = inference_payload(agent, self.compatibility, {"seed": 15}, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inference.pt"
            atomic_save(path, payload)
            for mutation in ("feature_order", "normalization", "reward", "control", "actions"):
                expected = deepcopy(self.compatibility)
                if mutation == "feature_order":
                    expected["config"]["OBSERVATION_FEATURE_NAMES"][0:2] = reversed(
                        expected["config"]["OBSERVATION_FEATURE_NAMES"][0:2])
                elif mutation == "normalization":
                    expected["config"]["OBSERVATION_SCALES"][18] = 50.0
                elif mutation == "reward":
                    expected["config"]["REWARD_FAILURE_COST"] = 20.0
                elif mutation == "control":
                    expected["control"]["TAU_ACC"] = .7
                else:
                    expected["config"]["ACTION_NAMES"]["0"] = "Keep + hold"
                with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "不兼容"):
                    load_checked(path, expected, kind="dqn_inference")

    def test_safe_inference_cpu_load_preserves_outputs(self):
        source = DQNAgent(seed=21, settings=self.settings)
        restored = DQNAgent(seed=88, settings=self.settings)
        observation = torch.arange(34, dtype=torch.float32) / 34
        expected = source.online(observation).detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inference.pt"
            atomic_save(path, inference_payload(source, self.compatibility, {"seed": 21}, 17))
            raw = torch.load(path, weights_only=True, map_location="cpu")
            self.assertEqual(raw["kind"], "dqn_inference")
            self.assertEqual(raw["signature"], signature(self.compatibility))
            load_inference(path, restored, self.compatibility)
        self.assertTrue(torch.equal(expected, restored.online(observation)))
        self.assertEqual(next(restored.online.parameters()).device.type, "cpu")
        self.assertFalse(restored.online.training)

    def test_checkpoint_kind_is_not_interchangeable(self):
        agent = DQNAgent(settings=self.settings)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inference.pt"
            atomic_save(path, inference_payload(agent, self.compatibility, {}, 0))
            with self.assertRaisesRegex(ValueError, "类型"):
                load_checked(path, self.compatibility, kind="dqn_resume_boundary")

    def test_real_boundary_resume_matches_continuous_training(self):
        """同seed、同400步预算；边界停止不改变epsilon分母或场景随机序列。"""
        # 仅缩小独立验证集并增加检查频率，不伪造仿真或改变学习公式。
        # 每个回合边界保存，确保连续/额外停止两种执行比较的是同一完整边界。
        short_config = {"validation_episodes": 1, "validation_every": 100,
                        "checkpoint_every": 1}
        with tempfile.TemporaryDirectory(prefix="me5418-boundary-test-") as directory, \
                patch.dict(config.TRAINING_CONFIG, short_config), redirect_stdout(io.StringIO()):
            continuous_dir = Path(directory) / "continuous"
            resumed_dir = Path(directory) / "resumed"
            arguments = dict(mode="smoke", seed=303, steps=400, settings_override=self.settings)
            continuous = run_training(**arguments, run_dir=continuous_dir)
            stopped = run_training(**arguments, run_dir=resumed_dir, stop_after_boundary=30)
            self.assertEqual(stopped["status"], "stopped_at_boundary")
            self.assertGreaterEqual(stopped["env_steps"], 30)
            self.assertLess(stopped["env_steps"], 400)
            self.assertEqual(stopped["env_steps"], stopped["last_resume_env_steps"])
            boundary_payload = torch.load(resumed_dir / "resume.pt", weights_only=True, map_location="cpu")
            self.assertTrue(boundary_payload["episode_boundary"])
            self.assertEqual(boundary_payload["training_state"]["env_steps"], stopped["env_steps"])
            self.assertEqual(boundary_payload["agent"]["gradient_steps"], stopped["gradient_steps"])
            self.assertGreater(boundary_payload["agent"]["replay"]["size"], 0)
            self.assertGreater(len(boundary_payload["agent"]["optimizer"]["state"]), 0)
            self.assertIn("rng_state", boundary_payload["agent"])
            self.assertIn("torch_rng_state", boundary_payload["agent"])
            self.assertEqual(boundary_payload["sampler"]["cursor"], stopped["episodes"])
            resumed = run_training(**arguments, run_dir=resumed_dir, resume=resumed_dir / "resume.pt")
            for key in ("env_steps", "gradient_steps", "episodes", "physics_frames", "last_resume_env_steps"):
                self.assertEqual(continuous[key], resumed[key])
            self.assertEqual(continuous["env_steps"], 400)
            self.assertEqual(continuous["gradient_steps"], 97)
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(resumed["segment"], 1)
            self.assertFalse(resumed["final_test_executed"])
            # 最终推理权重逐位相同；最近完整边界的全部学习状态也逐位相同。
            final_a = torch.load(continuous_dir / "last.pt", weights_only=True, map_location="cpu")
            final_b = torch.load(resumed_dir / "last.pt", weights_only=True, map_location="cpu")
            self.assert_nested_equal(final_a["online"], final_b["online"])
            resume_a = torch.load(continuous_dir / "resume.pt", weights_only=True, map_location="cpu")
            resume_b = torch.load(resumed_dir / "resume.pt", weights_only=True, map_location="cpu")
            self.assert_nested_equal(resume_a["agent"], resume_b["agent"])
            self.assert_nested_equal(resume_a["sampler"], resume_b["sampler"])
            self.assertEqual(final_a["validation"]["score"], final_b["validation"]["score"])
            self.assertEqual(final_a["validation"]["failure_rate"], final_b["validation"]["failure_rate"])
            scenario_a = [json.loads(line) for line in (continuous_dir / "scenarios.jsonl").read_text().splitlines()]
            scenario_b = [json.loads(line) for line in (resumed_dir / "scenarios.jsonl").read_text().splitlines()]
            for sequence in (scenario_a, scenario_b):
                for item in sequence:
                    item.pop("segment")
            self.assert_nested_equal(scenario_a, scenario_b)
            # 禁止恢复时更改预算，否则epsilon衰减会悄悄改变。
            with self.assertRaisesRegex(ValueError, "不匹配"):
                run_training(**{**arguments, "steps": 401}, run_dir=resumed_dir,
                             resume=resumed_dir / "resume.pt")


if __name__ == "__main__":
    unittest.main()
