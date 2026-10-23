"""Task routing, walking phases and simultaneous body/EE velocity objectives."""
import unittest
from unittest.mock import patch
import importlib.util
from pathlib import Path
import numpy as np

from whole_body_mppi.control.tasks.definitions import get_task
from whole_body_mppi.control.tasks.ee_velocity import EEVelocityController


class EEVelocityTaskTests(unittest.TestCase):
    def test_cli_accepts_both_new_names_and_rejects_old_name(self):
        path = Path(__file__).resolve().parents[1]/'scripts/simulate_mppi.py'
        spec = importlib.util.spec_from_file_location('velocity_task_cli', path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        for name in ('ee_reach_cmd_vel', 'ee_walk_cmd_vel'):
            with patch('sys.argv', ['simulate_mppi.py', '--robot', 'go1_dm', '--task', name, '--sync']):
                args = cli.parse_args()
                self.assertEqual(args.task, name)
                self.assertTrue(args.sync)
        with patch('sys.argv', ['simulate_mppi.py', '--robot', 'go1_dm', '--task', 'ee_cmd_vel']), patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                cli.parse_args()
        with self.assertRaises(ValueError):
            get_task('ee_cmd_vel')

    def test_walk_tracks_body_and_ee_velocity_with_stand_walk_stop_phases(self):
        a = EEVelocityController('ee_walk_cmd_vel')
        self.addCleanup(a.shutdown)
        a.set_params(5, .1, max(6, 2*a.n_iterations))
        self.assertEqual(a.task_data['config_path'], 'configs/mppi_ee_walk_cmd_vel.yml')
        self.assertEqual(a.gait_scheduler.type, 'in_place')
        a.set_command_time(1.)
        self.assertEqual(a.command_phase_index, 1)
        self.assertEqual(a.gait_scheduler.type, 'walk_fast')
        np.testing.assert_allclose(a.body_ref[7:9], [.2, 0])
        self.assertEqual(a.Q[0, 0], 0.)
        self.assertGreater(a.Q[25, 25], 0.)
        self.assertGreater(a.Q[26, 26], 0.)
        np.testing.assert_allclose(a.ee_command, [.2, 0, 0])
        a.joints_ref = a.gait_scheduler.gait[:, a.gait_scheduler.indices[:a.horizon]]
        home = np.r_[0., a.model.key('home').qpos, np.zeros(a.model.nv)]
        states = np.broadcast_to(home, (1, a.horizon, len(home))).copy()
        actions = np.broadcast_to(a.sampling_init, (1, a.horizon, 18)).copy()
        body_mismatch = a.cost_func(states[..., 1:], actions, a.joints_ref, a.body_ref)
        states[..., 26] = .2
        body_match = a.cost_func(states[..., 1:], actions, a.joints_ref, a.body_ref)
        np.testing.assert_allclose(body_mismatch-body_match, a.horizon*a.Q[25, 25]*.2**2)
        states[..., 26] = 0.
        weighted = a.task_cost(states, actions)
        a.velocity_weight = 0.
        unweighted = a.task_cost(states, actions)
        np.testing.assert_allclose(weighted-unweighted, 2000.*a.horizon*.2**2)
        a.set_command_time(5.)
        self.assertEqual(a.command_phase_index, 2)
        self.assertEqual(a.gait_scheduler.type, 'in_place')
        np.testing.assert_allclose(a.body_ref[7:9], [0, 0])
        np.testing.assert_array_equal(a.act_min[16:18], [0, 0])
        np.testing.assert_array_equal(a.act_max[16:18], [0, 0])

    def test_future_commands_change_at_phase_boundary(self):
        a = EEVelocityController('ee_walk_cmd_vel')
        self.addCleanup(a.shutdown)
        a.set_command_time(.98)
        np.testing.assert_array_equal(a.phase_indices([.98, .99, 1., 4.99, 5., 100.]), [0, 0, 1, 1, 2, 2])
        a.joints_ref = a.gait_scheduler.gait[:, a.gait_scheduler.indices[:a.horizon]]
        home = np.r_[0., a.model.key('home').qpos, np.zeros(a.model.nv)]
        states = np.broadcast_to(home, (1, 4, len(home))).copy()
        actions = np.broadcast_to(a.sampling_init, (1, 4, 18)).copy()
        a.velocity_weight = 2000.
        weighted = a.task_cost(states, actions)
        a.velocity_weight = 0.
        baseline = a.task_cost(states, actions)
        np.testing.assert_allclose(weighted-baseline, 2*2000.*.2**2)
