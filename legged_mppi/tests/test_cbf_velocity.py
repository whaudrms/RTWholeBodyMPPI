"""Velocity CBF frames, command transitions, solver integration and reports."""
import csv
import json
import tempfile
import unittest

import mujoco
import numpy as np

from whole_body_mppi.control.safety.cbf import configure_cbf
from whole_body_mppi.control.solvers.options import apply_mppi_mode
from whole_body_mppi.control.tasks.ee_velocity import EEVelocityController
from whole_body_mppi.control.tasks.base import PACKAGE_ROOT
from whole_body_mppi.interface.simulator import Simulator


class VelocityCBFTests(unittest.TestCase):
    def agent(self, task='ee_walk_cmd_vel'):
        a = EEVelocityController(task)
        self.addCleanup(a.shutdown)
        a.set_params(5, .1, max(6, 2*a.n_iterations))
        return a

    @staticmethod
    def observation(a):
        return np.r_[a.model.key('home').qpos, np.zeros(a.model.nv)]

    def test_speed_and_error_norms_use_correct_frames(self):
        a = self.agent()
        a.set_command_time(1.)
        configure_cbf(a, 'on')
        obs = self.observation(a)
        obs[3:7] = [np.sqrt(.5), 0., 0., np.sqrt(.5)]  # yaw +90deg
        obs[a.model.nq:a.model.nq+3] = [0., .3, 0.]
        before = obs.copy()
        body, ee, bc, ec = a.cbf.velocity_values(obs)
        np.testing.assert_allclose(body, [.3, 0.], atol=1e-12)
        np.testing.assert_allclose(ee, [0., .3, 0.], atol=1e-12)
        np.testing.assert_allclose(bc, [.2, 0.])
        np.testing.assert_allclose(ec, [.2, 0., 0.])
        margins = a.cbf.velocity_barriers(obs)
        np.testing.assert_allclose(margins, [.75, .75, 1-.1**2/.25**2, 1-.13/.25**2])
        np.testing.assert_array_equal(obs, before)

    def test_ee_speed_includes_arm_motion_with_stationary_body(self):
        a = self.agent()
        configure_cbf(a, 'on')
        obs = self.observation(a)
        d = mujoco.MjData(a.model)
        d.qpos[:] = obs[:a.model.nq]
        mujoco.mj_forward(a.model, d)
        jp, jr = np.zeros((3, a.model.nv)), np.zeros((3, a.model.nv))
        mujoco.mj_jacSite(a.model, d, jp, jr, a.model.site('dm_tcp').id)
        arm_dofs = np.arange(18, 24)
        axis = arm_dofs[np.argmax(np.linalg.norm(jp[:, arm_dofs], axis=0))]
        obs[a.model.nq+axis] = 1./np.linalg.norm(jp[:, axis])
        body, ee, _, _ = a.cbf.velocity_values(obs)
        np.testing.assert_allclose(body, 0.)
        self.assertAlmostEqual(np.linalg.norm(ee), 1.)
        margins = a.cbf.velocity_barriers(obs)
        self.assertEqual(margins[0], 1.)
        self.assertLess(margins[1], 0.)

    def test_first_transition_and_future_command_phase_are_checked(self):
        a = self.agent()
        configure_cbf(a, 'on')
        a.set_command_time(.99)
        obs = self.observation(a)
        states = np.tile(np.r_[0., obs], (2, 3, 1))
        before = states.copy()
        c = a.cbf.constraints(states, obs)
        width = c.shape[-1]//3
        # Future t=1.0 uses the new .2m/s command, although current command is 0.
        np.testing.assert_allclose(c[:, 0, width-2:width], 1-.2**2/.25**2)
        self.assertTrue(np.all(c[:, 0, 3*width-2:] < 0.))
        np.testing.assert_array_equal(states, before)
        # An accelerating body can violate the barrier rate before its cap.
        a.set_command_time(0.)
        states[:, 0, 1+a.model.nq] = .3
        c = a.cbf.constraints(states, obs)
        self.assertGreater(c[0, 0, width-4], 0.)
        self.assertLess(c[0, 0, 3*width-4], 0.)

    def test_on_all_modes_and_off_keeps_seeded_actions(self):
        for task in ('ee_reach_cmd_vel', 'ee_walk_cmd_vel'):
            plain, off = self.agent(task), self.agent(task)
            configure_cbf(off, 'off')
            obs = self.observation(plain)
            np.testing.assert_array_equal(plain.update(obs.copy()), off.update(obs.copy()))
            for mode in ('mppi', 'dial-mpc', 'cma-mppi', 'mpopi', 'mbd'):
                with self.subTest(task=task, mode=mode):
                    a = self.agent(task)
                    configure_cbf(a, 'on')
                    apply_mppi_mode(a, mode)
                    a.set_command_time(1.02)
                    action = a.update(self.observation(a))
                    self.assertTrue(np.isfinite(action).all())
                    self.assertTrue(np.isfinite(a.eval_best_trajectory()))
                    self.assertFalse(hasattr(a, 'ee_objective'))
                    self.assertGreaterEqual(a.cbf.last['verified_min_residual'],
                                            a.cbf.last['nominal_min_residual']-1e-12)
                    np.testing.assert_array_equal(action[a.fixed_arm_actions], a.act_min[a.fixed_arm_actions])

    def test_on_off_sync_and_report_preserve_velocity_task_semantics(self):
        for mode, sync in (('off', False), ('on', False), ('on', True)):
            with self.subTest(mode=mode, sync=sync):
                a = self.agent()
                # Exercise start, move, stop transitions in this short fixture.
                a.phase_ends = np.array([.01, .02, .03])
                configure_cbf(a, mode)
                sim = Simulator(agent=a, viewer=False, T=4, robot='go1_dm', initial_key='home',
                                model_path=PACKAGE_ROOT/a.task_data['sim_path'], sync=sync)
                sim.run()
                np.testing.assert_array_equal(sim.command_phase_indices, [0, 1, 2, 2])
                self.assertTrue(np.isfinite(sim.cbf_velocity_margins).all())
                self.assertTrue(np.isnan(sim.cbf_ee_radius).all())
                self.assertIsNone(sim.ee_errors)
                self.assertTrue(np.isnan(sim.body_position_errors).all())
                with tempfile.TemporaryDirectory() as root:
                    folder = sim.save_results('mppi', root)
                    summary = json.loads((folder/'summary.json').read_text())
                    config = json.loads((folder/'config.json').read_text())
                    saved = np.load(folder/'trajectory.npz')
                    self.assertEqual(summary['cbf']['enabled'], mode == 'on')
                    self.assertIsNotNone(summary['cbf']['velocity']['ee_speed']['margin'])
                    self.assertFalse(config['execution']['cbf']['ee']['active'])
                    self.assertNotIn('ee_targets', saved.files)
                    self.assertNotIn('body_targets', saved.files)
                    self.assertEqual(saved['cbf_velocity_margins'].shape, (4, 4))
                    with (folder/'metrics.csv').open() as stream:
                        row = next(csv.DictReader(stream))
                    self.assertIn('cbf_ee_speed_margin', row)
                    self.assertIn('cbf_body_velocity_error_margin', row)

    def test_invalid_velocity_limits_rejected(self):
        a = self.agent()
        for key in ('body_speed_max', 'ee_speed_max', 'body_velocity_error_max', 'ee_velocity_error_max'):
            for value in (0., -1., float('nan')):
                a.params['cbf'] = {key: value}
                with self.assertRaises(ValueError):
                    configure_cbf(a, 'on')


if __name__ == '__main__':
    unittest.main()
