"""真实环境与网络整合回归；不修改旧测试，不用合成观测冒充驾驶交互。"""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import config
from agent import DQNAgent
from checkpoints import (atomic_save, compatibility_record, inference_payload,
                         load_inference, signature)
from environment import make_env
from highway_m1 import HighwayDecisionEnv, SPEED_TARGETS
from highway_m1.environment import BoundedEgoVehicle
from highway_m2 import QNetwork, load_checkpoint, save_checkpoint
from legacy_checkpoint import legacy_compatibility_record, load_legacy_inference
from network import DQNNetwork
from teammate_demo import project_demo_info, render_project_frame
from teammates.m1.demo import DemonstrationPolicy, overlay

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "network_before_integration", ROOT / "tests/fixtures/network_before_integration.py"
)
legacy_network = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy_network)


class TeammateIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def m1_env(self, index=2):
        env = HighwayDecisionEnv(config={"vehicles_count": 0, "initial_speed_index": index})
        self.addCleanup(env.close)
        return env

    def project_env(self, render_mode=None):
        env = make_env(render_mode=render_mode, invalid_action_mode="raise")
        self.addCleanup(env.close)
        return env

    def test_dqn_actually_inherits_teammate_network_without_duplicate_layers(self):
        model = DQNNetwork()
        self.assertIsInstance(model, QNetwork)
        self.assertEqual(len(list(model.modules())), 7)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 22153)
        self.assertEqual(list(model.state_dict()),
                         ['layers.0.weight', 'layers.0.bias', 'layers.2.weight',
                          'layers.2.bias', 'layers.4.weight', 'layers.4.bias'])

    def test_initialization_matches_original_network_for_multiple_seeds(self):
        with torch.random.fork_rng():
            for seed in (0, 11, 22, 33):
                torch.manual_seed(seed)
                previous = legacy_network.DQNNetwork()
                torch.manual_seed(seed)
                current = DQNNetwork()
                for name, value in previous.state_dict().items():
                    self.assertTrue(torch.equal(value, current.state_dict()[name]), (seed, name))

    def test_explicit_cpu_float32_ignores_global_default_dtype(self):
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            self.assertTrue(all(p.dtype == torch.float32 and p.device.type == "cpu"
                                for p in DQNNetwork().parameters()))
        finally:
            torch.set_default_dtype(previous)

    def test_real_observation_outputs_and_gradients_equal_original(self):
        env = self.project_env()
        obs, info = env.reset(seed=91)
        observations = [obs]
        for _ in range(5):
            obs, _, terminated, truncated, _ = env.step(4)
            observations.append(obs)
            if terminated or truncated:
                break
        previous, current = legacy_network.DQNNetwork(), DQNNetwork()
        current.load_state_dict(previous.state_dict(), strict=True)
        x = torch.from_numpy(np.stack(observations))
        self.assertTrue(torch.equal(previous(x), current(x)))
        previous(x).square().mean().backward()
        current(x).square().mean().backward()
        for a, b in zip(previous.parameters(), current.parameters()):
            self.assertTrue(torch.equal(a.grad, b.grad))

    def test_one_optimizer_step_matches_original_network(self):
        previous, current = legacy_network.DQNNetwork(), DQNNetwork()
        current.load_state_dict(previous.state_dict())
        x = torch.linspace(-1, 1, 34).repeat(4, 1)
        for model in (previous, current):
            optimizer = torch.optim.Adam(model.parameters(), lr=config.DQN_CONFIG['learning_rate'])
            optimizer.zero_grad()
            torch.nn.functional.smooth_l1_loss(model(x), torch.ones(4, 9)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.DQN_CONFIG['gradient_norm_clip'])
            optimizer.step()
        for a, b in zip(previous.parameters(), current.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_m1_real_gym_interface_and_timing(self):
        env = self.m1_env()
        observation, info = env.reset(seed=25)
        self.assertEqual(observation.shape, (34,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(env.action_space.n, 9)
        self.assertEqual(info['action_mask'].dtype, np.bool_)
        next_observation, reward, terminated, truncated, info = env.step(4)
        self.assertTrue(env.observation_space.contains(next_observation))
        self.assertTrue(np.isfinite(reward))
        self.assertEqual(info['simulation_steps'], 3)
        self.assertAlmostEqual(info['elapsed_time'], .2)
        self.assertFalse(terminated or truncated)

    def test_m1_all_nine_actions_keep_original_targets(self):
        env = self.m1_env()
        for action in range(9):
            with self.subTest(action=action):
                env.reset(seed=25)
                observation, _, _, _, info = env.step(action)
                self.assertEqual(info['executed_action'], action)
                self.assertEqual(info['target_lane'], action // 3)
                self.assertEqual(info['target_speed_index'], 2 + action % 3 - 1)
                self.assertAlmostEqual(info['target_speed'], SPEED_TARGETS[2 + action % 3 - 1])
                self.assertEqual(observation[25:].argmax(), action)
                self.assertAlmostEqual(info['simulation_time_s'], .2)

    def test_m1_structural_masks_keep_zero_speed_and_lane_lock(self):
        env = self.m1_env(index=0)
        env.reset(seed=25)
        mask = env.action_mask()
        self.assertFalse(mask[[0, 3, 6]].any())
        self.assertTrue(mask[4])
        env.step(1)
        self.assertTrue(env.manoeuvre_in_progress())
        self.assertFalse(env.action_mask()[[0, 1, 2, 6, 7, 8]].any())
        self.assertTrue(env.action_mask()[5])
        self.assertEqual(env.vehicle.target_speed, 0)
        env.step(5)
        self.assertGreater(env.vehicle.speed, 0)

    def test_m1_horizon_truncates_with_real_physics(self):
        env = self.m1_env()
        env.reset(seed=25)
        for count in range(1, 202):
            _, _, terminated, truncated, info = env.step(4)
            if terminated or truncated:
                break
        self.assertEqual(count, 200)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info['simulation_steps'], 600)

    def test_m1_sensing_filters_euclidean_distance(self):
        from highway_env.vehicle.kinematics import Vehicle
        env = self.m1_env()
        env.reset(seed=25)
        ego = env.vehicle
        # 随机原点 +100 后相减可能得到100.00000000000003；边界夹具使用精确坐标。
        ego.position = np.array([100.0, 4.0])
        ego.on_state_update()
        other = Vehicle(env.road, ego.position + [100.0, 4.0], heading=0, speed=10)
        env.road.vehicles.append(other)
        self.assertFalse(any(lane['front']['presence'] for lane in env._sense()))
        other.position = ego.position + [100.0, 0.0]
        other.on_state_update()
        front = env._sense()[1]['front']
        self.assertTrue(front['presence'])
        self.assertAlmostEqual(front['gap'], 95)
        self.assertLess(front['relative_vx'], 0)

    def test_m1_different_normalization_is_not_silently_adopted(self):
        m1, project = self.m1_env(index=3), self.project_env()
        first, _ = m1.reset(seed=25)
        second, _ = project.reset(seed=25, options={'config': {'vehicles_count': 0}})
        self.assertAlmostEqual(float(first[18]), 20.85 / 35, places=6)
        self.assertAlmostEqual(float(second[18]), 25 / 40, places=6)
        self.assertEqual(config.OBS_LATERAL_SPEED_SCALE_MPS, 10)
        self.assertNotEqual(tuple(first), tuple(second))

    def test_m1_reward_and_acceleration_remain_separate_from_formal(self):
        m1, project = self.m1_env(index=3), self.project_env()
        m1.reset(seed=25)
        project.reset(seed=25, options={'config': {'vehicles_count': 0}})
        _, reward_m1, _, _, info_m1 = m1.step(3)
        _, reward_project, _, _, info_project = project.step(3)
        self.assertAlmostEqual(info_m1['reward_components']['speed_change'], -.01)
        self.assertAlmostEqual(info_project['reward_terms']['target_speed'], -.02)
        self.assertGreaterEqual(m1.vehicle.action['acceleration'], -5)
        self.assertLessEqual(m1.vehicle.action['acceleration'], 3)
        self.assertIsNone(project.controller.get_low_level_parameters()['fixed_acceleration_limit'])
        self.assertNotAlmostEqual(reward_m1, reward_project)

    def test_m1_seed_and_action_sequence_is_reproducible(self):
        a, b = self.m1_env(), self.m1_env()
        obs_a, _ = a.reset(seed=59)
        obs_b, _ = b.reset(seed=59)
        np.testing.assert_array_equal(obs_a, obs_b)
        for action in [5, 4, 3, 4, 1, 4, 4]:
            result_a, result_b = a.step(action), b.step(action)
            np.testing.assert_array_equal(result_a[0], result_b[0])
            self.assertEqual(result_a[1:4], result_b[1:4])

    def test_project_info_adapter_and_reused_policy_have_no_environment_side_effect(self):
        env = self.project_env()
        _, info = env.reset(seed=19)
        before = pickle.dumps(info)
        adapted = project_demo_info(info)
        policy = DemonstrationPolicy('random', 42)
        other = DemonstrationPolicy('random', 42)
        for _ in range(12):
            action = policy(adapted)
            self.assertTrue(info['action_mask'][action])
            self.assertEqual(action, other(adapted))
        self.assertEqual(pickle.dumps(info), before)
        self.assertEqual(env.get_debug_state()['physics_steps'], 0)
        self.assertEqual(adapted['target_speed'], env.controller.vehicle.target_speed)

    def test_original_overlay_renders_real_project_frame(self):
        env = self.project_env(render_mode='rgb_array')
        _, info = env.reset(seed=19)
        _, reward, _, _, info = env.step(4)
        frame = render_project_frame(env)
        self.assertGreater(np.std(frame), 1)
        self.assertGreaterEqual(frame.shape[1], 1000)
        result = overlay(frame, project_demo_info(info), reward, 'hold', 19, SPEED_TARGETS)
        self.assertEqual(result.shape, (frame.shape[0] + 248, frame.shape[1], 3))
        np.testing.assert_array_equal(result[:frame.shape[0]], frame)
        self.assertGreater(np.std(result), 1)

    def test_teammate_checkpoint_roundtrip_stays_distinct_from_project_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'm2.pt'
            model = QNetwork()
            save_checkpoint(model, path)
            restored = load_checkpoint(path)
            x = torch.ones(34)
            self.assertTrue(torch.equal(model(x), restored(x)))
            save_checkpoint(DQNNetwork(), path)
            record = torch.load(path, weights_only=True)
            self.assertEqual(record['observation_names'], list(config.OBSERVATION_FEATURE_NAMES))
            with self.assertRaisesRegex(ValueError, 'differ'):
                load_checkpoint(path)

    def test_legacy_inference_requires_explicit_bridge_and_preserves_outputs(self):
        expected = legacy_compatibility_record()
        self.assertEqual(signature(expected),
                         'c98a0123f46fa245cbaf1a9a6140dd2ae99b6ca916f8141047bc4cc746657989')
        source, destination = DQNAgent(seed=11), DQNAgent(seed=22)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.pt'
            atomic_save(path, inference_payload(source, expected, {'seed': 11}, 15))
            with self.assertRaisesRegex(ValueError, '不兼容'):
                load_inference(path, destination, compatibility_record())
            load_legacy_inference(path, destination)
            self.assertTrue(torch.equal(source.online(torch.ones(34)), destination.online(torch.ones(34))))

    def test_legacy_bridge_rejects_modified_feature_reward_or_source(self):
        current = compatibility_record()
        for section, name in [('source_hashes', 'network.py'), ('config', 'REWARD_FAILURE_COST')]:
            modified = deepcopy(current)
            modified[section][name] = 'changed'
            with self.subTest(name=name), patch('legacy_checkpoint.compatibility_record', return_value=modified):
                with self.assertRaises(ValueError):
                    legacy_compatibility_record()

    def test_legacy_bridge_rejects_resume_and_unknown_weight_keys(self):
        agent = DQNAgent(seed=1)
        payload = inference_payload(agent, legacy_compatibility_record(), {}, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.pt'
            altered = deepcopy(payload)
            altered['kind'] = 'dqn_resume_boundary'
            atomic_save(path, altered)
            with self.assertRaisesRegex(ValueError, '类型'):
                load_legacy_inference(path, agent)
            altered = deepcopy(payload)
            altered['online']['unexpected'] = torch.zeros(1)
            atomic_save(path, altered)
            with self.assertRaisesRegex(ValueError, '权重键'):
                load_legacy_inference(path, agent)

    def test_frozen_configuration_and_learning_core_unchanged(self):
        import hashlib
        profile = json.loads((ROOT / 'integration_compatibility.json').read_text())
        for name in ['config.py', 'agent.py', 'replay_buffer.py', 'train.py', 'environment.py',
                     'joint_actions.py', 'observation.py', 'reward.py', 'project_environment.py']:
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
                             profile['base_source_hashes'][name], name)


if __name__ == '__main__':
    unittest.main()
