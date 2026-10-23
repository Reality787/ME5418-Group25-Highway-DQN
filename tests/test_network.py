"""阶段 2 网络验收：数值验证不是驾驶训练，不推进环境来选择动作。

网络依赖缺失会作为导入错误报告；不会通过跳过网络测试伪装验证成功。
仅真实环境依赖缺失时，环境相关检查明确标注“未验证”。
"""

import copy
import importlib
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import textwrap
import unittest

import numpy as np
import torch
from torch import nn

import config
import network
from network import DQNNetwork


PROJECT_ROOT = Path(network.__file__).resolve().parent
ENVIRONMENT_DEPENDENCY_ERROR = None
try:
    importlib.import_module("gymnasium")
    importlib.import_module("highway_env")
    from environment import HighwayLaneEnv, make_env
except ModuleNotFoundError as exc:
    ENVIRONMENT_DEPENDENCY_ERROR = f"未验证：缺少真实环境依赖 {exc.name}"


class DQNNetworkTests(unittest.TestCase):
    """固定随机种子和短小合成任务；恢复线程数与 RNG，不影响其余测试。"""

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.enterContext(torch.random.fork_rng(devices=[]))
        torch.manual_seed(5418)
        self.model = DQNNetwork()

    @staticmethod
    def environment_snapshot(env):
        """覆盖物理状态、目标、历史、掩码、累计奖励和实例/全局随机数。"""
        raw = env.simulator.unwrapped
        observation, _ = env.get_observation()
        return pickle.dumps({
            "time": raw.time,
            "physics_steps": raw.steps,
            "episode_steps": env.episode_steps,
            "episode_reward": env.episode_reward,
            "invalid_action_count": env.invalid_action_count,
            "needs_reset": env._needs_reset,
            "previous_action": env.simulator.previous_action,
            "observation": observation.tobytes(),
            "mask": env.get_action_mask().tobytes(),
            "rng": copy.deepcopy(raw.np_random.bit_generator.state),
            "action_rng": copy.deepcopy(env.action_space.np_random.bit_generator.state),
            "global_numpy_rng": np.random.get_state(),
            "vehicles": [
                (tuple(v.position), v.speed, v.heading, v.lane_index,
                 getattr(v, "target_lane_index", None),
                 getattr(v, "target_speed", None), getattr(v, "speed_index", None),
                 copy.deepcopy(v.action), v.crashed)
                for v in raw.road.vehicles
            ],
        }, protocol=5)

    def test_exact_architecture_bias_and_independent_parameter_count(self):
        self.assertIsInstance(self.model, nn.Module)
        self.assertIsInstance(self.model.layers, nn.Sequential)
        layers = list(self.model.layers)
        self.assertEqual([type(layer) for layer in layers],
                         [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear])
        expected_dimensions = [(34, 128), (128, 128), (128, 9)]
        for layer, dimensions in zip((layers[0], layers[2], layers[4]), expected_dimensions):
            self.assertEqual((layer.in_features, layer.out_features), dimensions)
            self.assertIsNotNone(layer.bias)
        self.assertEqual(config.NETWORK_INPUT_DIM, 34)
        self.assertEqual(config.HIDDEN_LAYER_SIZES, (128, 128))
        self.assertEqual(config.NETWORK_OUTPUT_DIM, 9)
        # 用独立的每层“权重 + 偏置”公式核对，不能仅复用生产代码的计数值。
        expected_count = (34 * 128 + 128) + (128 * 128 + 128) + (128 * 9 + 9)
        self.assertEqual(expected_count, 22153)
        self.assertEqual(sum(parameter.numel() for parameter in self.model.parameters()),
                         expected_count)
        self.assertEqual(len(list(self.model.parameters())), 6)
        self.assertEqual(len(list(self.model.modules())), 7)
        self.assertTrue(all(parameter.device.type == "cpu" and parameter.dtype == torch.float32
                            for parameter in self.model.parameters()))

    def test_single_observation_returns_nine_finite_q_values(self):
        observation = torch.linspace(-1.0, 1.0, 34, dtype=torch.float32)
        output = self.model(observation)
        self.assertEqual(tuple(output.shape), (9,))
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output.device, observation.device)
        self.assertTrue(torch.isfinite(output).all().item())

    def test_batch_shapes_and_single_batch_agree(self):
        for batch_size in (1, 4, 17):
            with self.subTest(batch_size=batch_size):
                observations = torch.randn(batch_size, 34)
                output = self.model(observations)
                self.assertEqual(tuple(output.shape), (batch_size, 9))
                self.assertTrue(torch.isfinite(output).all().item())
                torch.testing.assert_close(output[0], self.model(observations[0]),
                                           rtol=1e-5, atol=1e-7)

    def test_output_is_raw_q_without_softmax_or_sigmoid(self):
        # 末层权重归零后，输出必须原样等于含负数和大于 1 数的偏置。
        expected = torch.arange(-4.0, 5.0)
        with torch.no_grad():
            self.model.layers[-1].weight.zero_()
            self.model.layers[-1].bias.copy_(expected)
        actual = self.model(torch.randn(34))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_backward_preserves_graph_and_produces_finite_gradients(self):
        observations = torch.randn(6, 34, requires_grad=True)
        target = torch.randn(6, 9)
        output = self.model(observations)
        self.assertTrue(output.requires_grad)
        self.assertIsNotNone(output.grad_fn)
        nn.functional.mse_loss(output, target).backward()
        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
                self.assertGreater(parameter.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(observations.grad)
        self.assertTrue(torch.isfinite(observations.grad).all().item())
        self.assertGreater(observations.grad.abs().sum().item(), 0.0)

    def test_one_optimizer_step_changes_parameters(self):
        observations = torch.randn(8, 34)
        target = torch.randn(8, 9)
        before = {name: parameter.detach().clone()
                  for name, parameter in self.model.named_parameters()}
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.02)
        optimizer.zero_grad(set_to_none=True)
        nn.functional.mse_loss(self.model(observations), target).backward()
        optimizer.step()
        differences = [torch.sum(torch.abs(parameter.detach() - before[name])).item()
                       for name, parameter in self.model.named_parameters()]
        self.assertTrue(all(np.isfinite(differences)))
        self.assertGreater(sum(differences), 0.0)

    def test_fixed_small_regression_task_reduces_error(self):
        """固定合成监督任务，仅测试数值可优化性，不是驾驶训练。"""
        observations = torch.randn(16, 34)
        target = 0.3 * observations[:, :9] + 0.2
        with torch.no_grad():
            initial_loss = nn.functional.mse_loss(self.model(observations), target).item()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(self.model(observations), target)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            final_loss = nn.functional.mse_loss(self.model(observations), target).item()
        self.assertTrue(np.isfinite(final_loss))
        self.assertLess(final_loss, 0.1 * initial_loss)

    def test_state_dict_round_trip_loads_on_cpu_with_identical_outputs(self):
        observations = torch.randn(5, 34)
        self.model.eval()
        with torch.no_grad():
            expected = self.model(observations).clone()
        with tempfile.TemporaryDirectory(prefix="me5418-network-test-") as directory:
            path = Path(directory) / "network_state_dict.pt"
            torch.save(self.model.state_dict(), path)
            self.assertGreater(path.stat().st_size, 0)
            restored = DQNNetwork()
            state = torch.load(path, map_location="cpu", weights_only=True)
            restored.load_state_dict(state, strict=True)
            restored.eval()
            self.assertTrue(all(parameter.device.type == "cpu"
                                for parameter in restored.parameters()))
            with torch.no_grad():
                actual = restored(observations)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for key, value in self.model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)

    def test_non_tensor_inputs_are_rejected_explicitly(self):
        for value in (np.zeros(34, dtype=np.float32), [0.0] * 34, None, 4):
            with self.subTest(input_type=type(value).__name__):
                with self.assertRaises(TypeError) as error:
                    self.model(value)
                self.assertTrue(str(error.exception).strip())

    def test_incorrect_dimensions_and_empty_batch_are_rejected(self):
        for shape in ((), (33,), (35,), (2, 33), (2, 3, 34), (0, 34)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError) as error:
                    self.model(torch.zeros(shape, dtype=torch.float32))
                self.assertTrue(str(error.exception).strip())

    def test_dtype_mismatch_is_rejected_without_silent_conversion(self):
        for dtype in (torch.float64, torch.float16, torch.int64, torch.bool):
            with self.subTest(dtype=dtype):
                with self.assertRaises(TypeError) as error:
                    self.model(torch.zeros(34, dtype=dtype))
                self.assertTrue(str(error.exception).strip())

    def test_explicit_double_model_accepts_double_and_preserves_gradients(self):
        model = self.model.double()
        observations = torch.randn(3, 34, dtype=torch.float64, requires_grad=True)
        output = model(observations)
        self.assertEqual(output.dtype, torch.float64)
        output.square().sum().backward()
        self.assertEqual(observations.grad.dtype, torch.float64)
        self.assertTrue(torch.isfinite(observations.grad).all().item())
        with self.assertRaises(TypeError):
            model(torch.zeros(34, dtype=torch.float32))

    def test_device_mismatch_is_rejected_without_requiring_cuda(self):
        # meta 张量不分配 GPU；这里仅验证入口的设备一致性检查。
        observations = torch.empty(34, device="meta", dtype=torch.float32)
        with self.assertRaises(ValueError) as error:
            self.model(observations)
        self.assertTrue(str(error.exception).strip())

    def test_numpy_backed_input_array_is_not_modified(self):
        original = np.linspace(-1, 1, 34, dtype=np.float32)
        before = original.copy()
        observations = torch.from_numpy(original)
        for _ in range(3):
            self.model(observations)
        np.testing.assert_array_equal(original, before)
        np.testing.assert_array_equal(observations.numpy(), before)

    def test_noncontiguous_tensor_is_supported_without_input_mutation(self):
        storage = torch.randn(4, 68)
        observations = storage[:, ::2]
        self.assertFalse(observations.is_contiguous())
        before = storage.clone()
        output = self.model(observations)
        self.assertEqual(tuple(output.shape), (4, 9))
        self.assertTrue(torch.isfinite(output).all().item())
        torch.testing.assert_close(storage, before, rtol=0, atol=0)

    def test_repeated_forward_does_not_change_parameters_or_consume_rng(self):
        observations = torch.randn(2, 34)
        state_before = {key: value.clone() for key, value in self.model.state_dict().items()}
        rng_before = torch.get_rng_state().clone()
        first = self.model(observations)
        second = self.model(observations)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
        for name, parameter in self.model.state_dict().items():
            torch.testing.assert_close(parameter, state_before[name], rtol=0, atol=0)

    @unittest.skipIf(ENVIRONMENT_DEPENDENCY_ERROR is not None,
                     ENVIRONMENT_DEPENDENCY_ERROR or "")
    def test_formal_reset_observation_enters_network_without_changing_environment(self):
        env = make_env(render_mode=None)
        self.addCleanup(env.close)
        observation, info = env.reset(seed=5418)
        self.assertIsInstance(env, HighwayLaneEnv)
        self.assertIs(env.unwrapped, env)
        self.assertEqual(observation.shape, (34,))
        self.assertEqual(observation.dtype, np.dtype(np.float32))
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(env.action_space.n, 9)
        self.assertEqual(info["action_mask"].shape, (9,))
        self.assertIn("reward_terms", info)
        self.assertIsNone(info["native_reward_diagnostic"])
        observation_before = observation.copy()
        before = self.environment_snapshot(env)
        output = self.model(torch.from_numpy(observation))
        self.assertEqual(tuple(output.shape), (9,))
        self.assertTrue(torch.isfinite(output).all().item())
        np.testing.assert_array_equal(observation, observation_before)
        self.assertEqual(self.environment_snapshot(env), before)

    @unittest.skipIf(ENVIRONMENT_DEPENDENCY_ERROR is not None,
                     ENVIRONMENT_DEPENDENCY_ERROR or "")
    def test_real_step_observation_enters_network_without_extra_simulation(self):
        env = make_env(render_mode=None)
        self.addCleanup(env.close)
        env.reset(seed=5418)
        observation, reward, terminated, truncated, info = env.step(4)
        self.assertAlmostEqual(info["simulation_time"], 0.2)
        self.assertEqual(info["physics_steps"], 3)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertAlmostEqual(reward, sum(info["reward_terms"].values()), places=12)
        self.assertTrue(np.isfinite(reward))
        self.assertNotAlmostEqual(reward, info["native_reward_diagnostic"], places=6)
        self.assertEqual(observation[25 + 4], 1)
        before = self.environment_snapshot(env)
        self.model(torch.from_numpy(observation))
        self.assertEqual(self.environment_snapshot(env), before)

    def test_importing_network_does_not_construct_model_or_environment(self):
        script = textwrap.dedent("""
            import sys
            import torch
            from unittest.mock import patch
            with patch.object(torch.nn.Linear, '__init__',
                              side_effect=AssertionError('导入时不得构造网络')):
                import network
            forbidden = {'environment', 'project_environment', 'joint_actions',
                         'observation', 'highway_env', 'gymnasium', 'agent',
                         'replay_buffer', 'baseline', 'train'}
            assert not (forbidden & set(sys.modules)), forbidden & set(sys.modules)
            print('network import has no model or simulation side effects')
        """)
        result = subprocess.run([sys.executable, "-c", script], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, timeout=60, check=False,
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("no model or simulation side effects", result.stdout)

    @unittest.skipIf(ENVIRONMENT_DEPENDENCY_ERROR is not None,
                     ENVIRONMENT_DEPENDENCY_ERROR or "")
    def test_formal_environment_runs_when_training_module_imports_are_blocked(self):
        script = textwrap.dedent("""
            import importlib.abc
            import sys
            blocked = {'torch', 'network', 'agent', 'replay_buffer', 'baseline', 'train'}
            class BlockTrainingImports(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in blocked:
                        raise ImportError('环境不应导入学习模块: ' + fullname)
            sys.meta_path.insert(0, BlockTrainingImports())
            from environment import make_env
            env = make_env(render_mode=None)
            try:
                obs, info = env.reset(seed=5418)
                assert obs.shape == (34,) and str(obs.dtype) == 'float32'
                assert env.action_space.n == 9
                obs, reward, terminated, truncated, info = env.step(4)
                assert abs(reward - sum(info['reward_terms'].values())) < 1e-12
                assert not (blocked & set(sys.modules))
            finally:
                env.close()
            print('formal environment runs independently of torch and training modules')
        """)
        result = subprocess.run([sys.executable, "-c", script], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, timeout=60, check=False,
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("independently of torch", result.stdout)


if __name__ == "__main__":
    unittest.main()
