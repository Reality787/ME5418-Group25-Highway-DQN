"""独立的标准库验收：不替代或改写队友原 pytest 测试。"""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest

import numpy as np
import torch
from gymnasium.error import ResetNeeded
from gymnasium.utils.env_checker import check_env
from highway_env.vehicle.kinematics import Vehicle

from environment import make_env
from highway_m1 import HighwayDecisionEnv, SPEED_TARGETS
from highway_m1.environment import BoundedEgoVehicle
from highway_m2 import QNetwork as OriginalQNetwork
from network import QNetwork, greedy_action, load_checkpoint, save_checkpoint


class ActiveInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def env(self, **config):
        env = make_env(config={'vehicles_count': 0, **config})
        self.addCleanup(env.close)
        env.reset(seed=25)
        return env

    def place(self, env, lane=1, index=2, speed=None, offset=0, heading=0):
        vehicle = env.vehicle
        vehicle.position = np.array([100.0, lane * 4.0 + offset])
        vehicle.heading = float(heading)
        vehicle.speed_index = index
        vehicle.target_speed = SPEED_TARGETS[index]
        vehicle.speed = vehicle.target_speed if speed is None else speed
        vehicle.target_lane_index = ('0', '1', lane)
        vehicle.crashed = False
        vehicle.on_state_update()
        return vehicle

    def other(self, env, lane, dx, speed):
        vehicle = Vehicle(env.road, [env.vehicle.position[0] + dx, lane * 4.0], speed=speed)
        env.road.vehicles.append(vehicle)
        return vehicle

    def test_adopted_files_match_source_and_documented_patch_hashes(self):
        root = Path(__file__).resolve().parents[1]
        record = json.loads((root / 'teammates/SOURCE_MANIFEST.json').read_text())
        for name in ('highway_m1/environment.py', 'highway_m1/__init__.py',
                     'highway_m2/network.py', 'highway_m2/__init__.py'):
            self.assertFalse(record['adopted_files'][name]['modified'])
        for name, item in record['adopted_files'].items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(),
                                 item['sha256'])

    def test_public_entrypoints_are_original_classes(self):
        self.assertIs(QNetwork, OriginalQNetwork)
        env = self.env()
        self.assertIs(type(env), HighwayDecisionEnv)
        self.assertIs(type(env.vehicle), BoundedEgoVehicle)

    def test_default_scene_and_gym_contract(self):
        env = make_env()
        self.addCleanup(env.close)
        check_env(env, skip_render_check=True)
        obs, info = env.reset(seed=25)
        self.assertEqual(len(env.road.vehicles), 21)
        self.assertEqual(env.config['duration'], 40)
        self.assertEqual(env.config['policy_frequency'], 5)
        self.assertEqual(env.config['simulation_frequency'], 15)
        self.assertEqual(obs.shape, (34,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertTrue(np.isfinite(obs).all())
        self.assertEqual(env.action_space.n, 9)
        self.assertEqual(info['target_speed_index'], 3)
        self.assertAlmostEqual(env.vehicle.speed, 20.85)
        np.testing.assert_array_equal(obs[25:], np.zeros(9))

    def test_observation_physical_values_use_m1_scales_and_slots(self):
        env = self.env()
        ego = self.place(env, speed=14, offset=.5, heading=.2)
        ego.target_lane_index = ('0', '1', 2)
        for lane in range(3):
            self.other(env, lane, 20, 10)
            self.other(env, lane, -30, 20)
        self.assertEqual(ego.LENGTH, 5)
        obs = env.observation_type.observe()
        vx = 14 * np.cos(.2)
        for lane in range(3):
            np.testing.assert_allclose(obs[6*lane:6*lane+6],
                [1, .15, (10-vx)/35, 1, .25, (20-vx)/35], atol=1e-7)
        np.testing.assert_allclose(obs[18:25],
            [vx/35, 14*np.sin(.2)/35, .25, .5, 1, 13.9/27.8, 1], atol=1e-7)

    def test_nearest_visible_physical_vehicle_and_missing_encoding(self):
        env = self.env(); ego = self.place(env)
        far = self.other(env, 1, 100.1, 20)
        np.testing.assert_array_equal(env.observation_type.observe()[:18], np.tile([0,1,0], 6))
        far.position = np.array([200.0, 8.0]); far.on_state_update()
        self.assertFalse(env._sense()[2]['front']['presence'])  # 欧氏距离大于100
        far.position = np.array([200.0, 4.0]); far.on_state_update()
        self.assertAlmostEqual(env._sense()[1]['front']['gap'], 95)
        near = self.other(env, 1, 25, 10)
        self.other(env, 1, 50, 30)
        self.assertAlmostEqual(env._sense()[1]['front']['gap'], 20)
        before = env.observation_type.observe().copy()
        near.target_speed = 1000; near.target_lane_index = ('0', '1', 0)
        np.testing.assert_array_equal(before, env.observation_type.observe())

    def test_all_nine_actions_and_single_interval(self):
        env = self.env()
        for action in range(9):
            with self.subTest(action=action):
                env.reset(seed=25); self.place(env)
                obs, _, terminated, truncated, info = env.step(action)
                self.assertEqual(info['target_lane'], action//3)
                self.assertEqual(info['target_speed_index'], 2 + action%3 - 1)
                self.assertAlmostEqual(info['elapsed_time'], .2)
                self.assertEqual(info['simulation_steps'], 3)
                self.assertFalse(terminated or truncated)
                np.testing.assert_array_equal(obs[25:], np.eye(9)[action])

    def test_speed_changes_use_target_not_actual_speed(self):
        env = self.env(); self.place(env, index=3, speed=25)
        env.step(3)
        self.assertAlmostEqual(env.vehicle.target_speed, 13.9)
        # M1限制减速度为-5，0.2秒后为24，而非旧环境的21.696。
        self.assertAlmostEqual(env.vehicle.speed, 24)
        env.step(4)
        self.assertAlmostEqual(env.vehicle.target_speed, 13.9)

    def test_masks_and_original_atomic_hold_fallback(self):
        env = self.env(); self.place(env, lane=0, index=0)
        mask = env.action_mask()
        self.assertFalse(mask[:3].any())
        self.assertFalse(mask[[0,3,6]].any())
        self.assertTrue(mask[4])
        obs, _, _, _, info = env.step(2)
        self.assertTrue(info['invalid_action'])
        self.assertEqual(info['requested_action'], 2)
        self.assertEqual(info['executed_action'], 4)
        self.assertEqual(env.vehicle.speed_index, 0)
        self.assertEqual(env.vehicle.target_lane_index[2], 0)
        self.assertEqual(obs[25:].argmax(), 4)
        self.place(env, lane=2, index=4)
        self.assertFalse(env.action_mask()[6:].any())
        self.assertFalse(env.action_mask()[[2,5,8]].any())

    def test_lane_lock_and_keep_preserve_target(self):
        env = self.env(); self.place(env)
        env.step(0)
        self.assertTrue(env.manoeuvre_in_progress())
        self.assertFalse(env.action_mask()[[0,1,2,6,7,8]].any())
        env.step(5)
        self.assertEqual(env.vehicle.target_lane_index[2], 0)
        self.assertAlmostEqual(env.vehicle.target_speed, 13.9)
        for _ in range(100):
            env.step(4)
            if not env.manoeuvre_in_progress(): break
        self.assertFalse(env.manoeuvre_in_progress())
        self.assertTrue(env.action_mask()[7])

    def test_out_of_space_action_rejected_and_reset_clears_history(self):
        env = self.env(); env.step(5)
        old_time = env.time; old_position = env.vehicle.position.copy()
        with self.assertRaises(ValueError): env.step(9)
        self.assertEqual(env.time, old_time)
        np.testing.assert_array_equal(env.vehicle.position, old_position)
        obs, info = env.reset(seed=25)
        np.testing.assert_array_equal(obs[25:], np.zeros(9))
        self.assertIsNone(info['executed_action'])

    def test_original_progress_reward_is_not_multiplied_by_dt(self):
        env = self.env()
        _, reward, _, _, info = env.step(4)
        self.assertAlmostEqual(reward, 20.85/27.8)
        self.assertAlmostEqual(info['reward_components']['progress'], .75)

    def test_original_headway_and_joint_action_costs(self):
        env = self.env(); self.place(env, index=3)
        other = self.other(env, 1, 12, 20.85)
        _, reward, _, _, info = env.step(4)
        gap = other.position[0] - env.vehicle.position[0] - 5
        expected = .75 - .5 * max(0, 1-gap/(1.5*20.85))
        self.assertAlmostEqual(reward, expected)
        env.reset(seed=25); self.place(env)
        before = env.vehicle.position[0]
        _, reward, _, _, info = env.step(0)
        progress = (env.vehicle.position[0]-before)/(27.8*.2)
        self.assertAlmostEqual(reward, progress-.03-.01)

    def test_acceleration_bounds_apply_to_controller_and_collision(self):
        env = self.env(); vehicle = self.place(env)
        self.assertEqual(vehicle.speed_control(100), 3)
        self.assertEqual(vehicle.speed_control(-100), -5)
        for action in (2, 3, 5, 4):
            _, _, _, _, info = env.step(action)
            self.assertGreaterEqual(info['control']['acceleration'], -5)
            self.assertLessEqual(info['control']['acceleration'], 3)
        vehicle.crashed = True
        vehicle.action['acceleration'] = -100
        vehicle.clip_actions()
        self.assertGreaterEqual(vehicle.action['acceleration'], -5)
        self.assertEqual((vehicle.MIN_SPEED, vehicle.MAX_SPEED), (0,35))

    def test_zero_target_stop_and_restart(self):
        env = self.env(initial_speed_index=0)
        self.assertEqual(env.vehicle.target_speed, 0)
        self.place(env, index=0, speed=6.95)
        for _ in range(40): env.step(4)
        self.assertLess(env.vehicle.speed, .05)
        env.step(5)
        self.assertGreater(env.vehicle.speed, .1)
        self.assertEqual(env.vehicle.target_speed, 6.95)

    def test_termination_and_original_failure_costs(self):
        for crashed, offroad in ((True,False),(False,True),(True,True)):
            with self.subTest(crashed=crashed,offroad=offroad):
                env = self.env(); vehicle = self.place(env)
                if offroad: vehicle.position[1] = -12; vehicle.on_state_update()
                vehicle.crashed = crashed
                _, _, terminated, truncated, info = env.step(4)
                self.assertTrue(terminated); self.assertFalse(truncated)
                self.assertEqual(info['reward_components']['collision'], -20*crashed)
                self.assertEqual(info['reward_components']['offroad'], -20*offroad)
                with self.assertRaises(ResetNeeded): env.step(4)

    def test_truncation_after_200_decisions(self):
        env = self.env()
        for step in range(1,201):
            _, _, terminated, truncated, info = env.step(4)
            self.assertFalse(terminated)
            self.assertEqual(truncated, step==200)
        self.assertEqual(info['simulation_steps'], 600)

    def test_original_render_and_close_recreate(self):
        env = make_env(config={'vehicles_count':0}, render_mode='rgb_array')
        try:
            env.reset(seed=25); env.step(4); frame = env.render()
            self.assertEqual(frame.shape, (300,1000,3))
            self.assertEqual(frame.dtype, np.uint8)
            self.assertGreater(frame.std(), 1)
        finally: env.close()
        fresh = self.env(); fresh.step(4)

    def test_seed_and_action_sequence_reproducible(self):
        a, b = self.env(), self.env()
        for action in (4,0,4,5,3,4):
            xa,ra,ta,ua,ia = a.step(action)
            xb,rb,tb,ub,ib = b.step(action)
            np.testing.assert_array_equal(xa,xb)
            self.assertEqual((ra,ta,ua),(rb,tb,ub))
            np.testing.assert_array_equal(ia['action_mask'], ib['action_mask'])

    def test_observation_reading_does_not_change_state(self):
        env = self.env(); a = env.observation_type.observe()
        time = env.time; position = env.vehicle.position.copy()
        b = env.observation_type.observe()
        np.testing.assert_array_equal(a,b); self.assertEqual(env.time,time)
        np.testing.assert_array_equal(env.vehicle.position,position)

    def test_network_structure_count_and_dtype_contract(self):
        model = QNetwork()
        linear = [m for m in model.layers if isinstance(m,torch.nn.Linear)]
        self.assertEqual([(m.in_features,m.out_features) for m in linear],[(34,128),(128,128),(128,9)])
        self.assertTrue(all(m.bias is not None for m in linear))
        self.assertEqual(sum(p.numel() for p in model.parameters()),22153)
        self.assertEqual(model(torch.ones(34,dtype=torch.float64)).shape,(9,))
        self.assertEqual(model(torch.ones(4,34)).shape,(4,9))
        self.assertEqual(model(torch.ones(34,dtype=torch.float64)).dtype,torch.float32)
        with self.assertRaises(ValueError):model(torch.ones(33))

    def test_actual_observation_network_mask_and_no_side_effect(self):
        env = self.env(); obs,info = env.reset(seed=25)
        model = QNetwork(); before = obs.copy(); time = env.time
        q = model(torch.from_numpy(obs))
        self.assertTrue(torch.isfinite(q).all())
        action = int(greedy_action(q,torch.from_numpy(info['action_mask'])))
        self.assertTrue(info['action_mask'][action])
        np.testing.assert_array_equal(obs,before); self.assertEqual(env.time,time)
        result = env.step(action)
        self.assertEqual(result[4]['executed_action'],action)

    def test_backward_optimizer_and_saved_outputs(self):
        with torch.random.fork_rng():
            torch.manual_seed(25)
            model=QNetwork(); x=torch.linspace(-1,1,34).repeat(8,1); y=torch.ones(8,9)
            optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            before={k:p.detach().clone() for k,p in model.named_parameters()}
            initial=torch.nn.functional.mse_loss(model(x),y).item()
            for _ in range(10):
                optimizer.zero_grad();loss=torch.nn.functional.mse_loss(model(x),y)
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                optimizer.step()
            self.assertLess(torch.nn.functional.mse_loss(model(x),y).item(),initial)
            self.assertTrue(all(not torch.equal(before[k],p) for k,p in model.named_parameters()))
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'q.pt';save_checkpoint(model,path)
                restored=load_checkpoint(path,map_location='cpu')
                self.assertTrue(torch.equal(model(x),restored(x)))


if __name__ == '__main__':
    unittest.main()
