"""DCBF state layout, first transition, filtering and baseline compatibility."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from whole_body_mppi.control.safety.cbf import configure_cbf
from whole_body_mppi.control.solvers.options import apply_mppi_mode
from whole_body_mppi.control.tasks.locomotion import LocomotionController
from whole_body_mppi.control.tasks.whole_body import WholeBodyController
from whole_body_mppi.control.tasks.ee_velocity import EEVelocityController
from whole_body_mppi.control.tasks.base import PACKAGE_ROOT
from whole_body_mppi.interface.simulator import Simulator


class CBFTests(unittest.TestCase):
    def agent(self, robot='go1', whole_body=False, variant='full'):
        a = WholeBodyController() if whole_body else LocomotionController('stand', robot, variant)
        self.addCleanup(a.shutdown)
        a.set_params(5, .1, 6)
        return a

    @staticmethod
    def observation(a):
        d = mujoco.MjData(a.model)
        mujoco.mj_resetDataKeyframe(a.model, d, 0)
        return np.r_[d.qpos, d.qvel]

    def test_off_preserves_seeded_actions_and_plan_exactly(self):
        plain, off = self.agent(), self.agent()
        configure_cbf(off, 'off')
        obs = self.observation(plain)
        np.testing.assert_array_equal(plain.update(obs.copy()), off.update(obs.copy()))
        np.testing.assert_array_equal(plain.selected_trajectory, off.selected_trajectory)
        self.assertEqual(off.cbf.extra_rollouts, 0)

    def test_tilt_derivative_matches_mujoco_quaternion_integration(self):
        a = self.agent()
        configure_cbf(a, 'on')
        obs = self.observation(a)
        obs[3:7] = np.array([.93, .2, -.3, .04])/np.linalg.norm([.93, .2, -.3, .04])
        obs[a.model.nq+3:a.model.nq+6] = [.4, -.7, .2]
        h, b = a.cbf.barriers(obs)
        future = obs.copy()
        eps = 1e-6
        mujoco.mj_integratePos(a.model, future[:a.model.nq], obs[a.model.nq:], eps)
        h1, _ = a.cbf.barriers(future)
        self.assertAlmostEqual((b[0]-h[0])/a.cbf.options.lookahead_seconds,
                               (h1[0]-h[0])/eps, places=5)

    def test_ee_funnel_contains_initial_error_and_shrinks_without_reanchoring(self):
        a = self.agent(whole_body=True)
        configure_cbf(a, 'on')
        obs = self.observation(a)
        a.cbf.observe(obs, 0.)
        error = np.linalg.norm(a.ee_objective.positions(obs[:a.model.nq])-a.ee_objective.target)
        r0, _ = a.cbf.ee_envelope()
        self.assertGreater(r0, error)
        self.assertEqual(a.cbf.ee_final_radius, a.task_data['ee_position_tolerance'])
        later = obs.copy()
        later[0] += .2  # measured error changes; the clock/funnel must not reset
        a.cbf.observe(later, 2.)
        r2, _ = a.cbf.ee_envelope()
        self.assertAlmostEqual(r2, a.cbf.ee_final_radius+(r0-a.cbf.ee_final_radius)*np.exp(-1.))
        a.cbf.observe(later, 100.)
        self.assertAlmostEqual(a.cbf.ee_envelope()[0], a.cbf.ee_final_radius)
        h, b = a.cbf.barriers(later)
        self.assertLess(h[-1], 0.)
        self.assertLess(b[-1], 0.)

    def test_ee_derivative_includes_velocity_and_funnel_motion_in_both_frames(self):
        for frame in ('world', 'base'):
            with self.subTest(frame=frame):
                a = self.agent(whole_body=True)
                a.ee_objective.frame = frame
                configure_cbf(a, 'on')
                obs = self.observation(a)
                obs[a.model.nq:] = np.linspace(-.1, .1, a.model.nv)
                a.cbf.observe(obs, 0.)
                h, b = a.cbf.barriers(obs)
                future = obs.copy()
                eps = 1e-6
                mujoco.mj_integratePos(a.model, future[:a.model.nq], obs[a.model.nq:], eps)
                h1, _ = a.cbf.barriers(future, eps)
                self.assertAlmostEqual((b[-1]-h[-1])/a.cbf.options.ee_lookahead_seconds,
                                       (h1[-1]-h[-1])/eps, places=4)

    def test_ee_prediction_uses_future_radius_and_initial_transition(self):
        a = self.agent(whole_body=True)
        configure_cbf(a, 'on')
        obs = self.observation(a)
        a.cbf.observe(obs, 0.)
        states = np.tile(np.r_[0., obs], (2, 3, 1))
        before = states.copy()
        c = a.cbf.constraints(states, obs)
        width = c.shape[-1]//3
        h0, _ = a.cbf.barriers(obs)
        self.assertLess(c[0, 0, width-1], h0[-1])
        self.assertLess(c[0, 2, width-1], c[0, 0, width-1])
        np.testing.assert_array_equal(states, before)
        self.assertTrue(np.isfinite(a.cbf.penalty(states, obs)).all())

    def test_ee_waypoint_switch_resets_from_measured_error(self):
        a = self.agent(whole_body=True)
        configure_cbf(a, 'on')
        obs = self.observation(a)
        a.cbf.observe(obs, 5.)
        a.cbf.observe(obs, 8.)
        a.waypoint_index += 1  # duplicate coordinates are still a new waypoint
        a.cbf.prepare_reference(obs)
        self.assertEqual(a.cbf._ee_start_time, 8.)
        r, _ = a.cbf.ee_envelope()
        self.assertEqual(r, a.cbf._ee_start_radius)
        a.ee_objective.target += np.array([.1, 0., 0.])
        a.cbf.observe(obs, 9.)
        error = np.linalg.norm(a.ee_objective.positions(obs[:a.model.nq])-a.ee_objective.target)
        self.assertAlmostEqual(a.cbf.ee_envelope()[0], max(a.cbf.ee_final_radius,
                                                       error+a.cbf.options.ee_initial_margin))

    def test_ee_off_keeps_actions_identical_and_velocity_tasks_have_no_ee_barrier(self):
        plain, off = self.agent(whole_body=True), self.agent(whole_body=True)
        configure_cbf(off, 'off')
        obs = self.observation(plain)
        np.testing.assert_array_equal(plain.update(obs.copy()), off.update(obs.copy()))
        a = EEVelocityController()
        self.addCleanup(a.shutdown)
        configure_cbf(a, 'off')
        self.assertFalse(a.cbf.ee_enabled)

    def test_ee_funnel_uses_physics_time_and_exports_monitor_for_on_and_off(self):
        import json
        for mode in ('on', 'off'):
            with self.subTest(mode=mode):
                a = self.agent(whole_body=True)
                configure_cbf(a, mode)
                sim = Simulator(agent=a, viewer=False, T=3, robot='go1_dm', initial_key='home',
                                model_path=PACKAGE_ROOT/a.task_data['sim_path'])
                sim.run()
                np.testing.assert_allclose(sim.time, [0., .01, .02])
                self.assertTrue(np.all(np.diff(sim.cbf_ee_radius) < 0))
                self.assertTrue(np.isfinite(sim.cbf_ee_margin).all())
                with tempfile.TemporaryDirectory() as root:
                    folder = sim.save_results('mppi', root)
                    saved = np.load(folder/'trajectory.npz')
                    np.testing.assert_array_equal(saved['cbf_ee_radius'], sim.cbf_ee_radius)
                    summary = json.loads((folder/'summary.json').read_text())
                    self.assertIsNotNone(summary['cbf']['ee_margin'])
                    self.assertEqual(summary['cbf']['enabled'], mode == 'on')

    def test_initial_transition_is_checked_without_mutating_states(self):
        a = self.agent()
        configure_cbf(a, 'on')
        obs = self.observation(a)
        states = np.tile(np.r_[0., obs], (2, 3, 1))
        states[:, 0, 3] -= .05  # world z: FULLPHYSICS includes time
        before = states.copy()
        constraints = a.cbf.constraints(states, obs)
        width = constraints.shape[-1]//3
        self.assertLess(constraints[0, 0, 2*width+1], 0.)
        self.assertTrue(np.all(np.isfinite(a.cbf.penalty(states, obs))))
        np.testing.assert_array_equal(states, before)

    def test_all_solver_modes_and_robot_layouts(self):
        cases = [('go1', False, 'full'), ('go1_dm', False, 'simple'),
                 ('go1_dm', False, 'full'), ('go1_dm', True, 'full')]
        for robot, whole_body, variant in cases:
            for mode in ('mppi', 'dial-mpc', 'cma-mppi', 'mpopi', 'mbd'):
                with self.subTest(robot=robot, whole_body=whole_body, variant=variant, mode=mode):
                    a = self.agent(robot, whole_body, variant)
                    configure_cbf(a, 'on')
                    apply_mppi_mode(a, mode)
                    action = a.update(self.observation(a))
                    self.assertTrue(np.all(np.isfinite(action)))
                    self.assertTrue(np.isfinite(a.eval_best_trajectory()))
                    self.assertGreaterEqual(a.cbf.last['verified_min_residual'],
                                            a.cbf.last['nominal_min_residual']-1e-12)
                    self.assertTrue(np.all(a.selected_trajectory >= a.act_min))
                    self.assertTrue(np.all(a.selected_trajectory <= a.act_max))
                    if whole_body:
                        np.testing.assert_array_equal(a.selected_trajectory[:, a.fixed_arm_actions],
                                                      np.broadcast_to(a.act_min[a.fixed_arm_actions],
                                                                      (a.horizon, len(a.fixed_arm_actions))))
                    elif robot == 'go1_dm' and variant == 'full':
                        np.testing.assert_array_equal(action[12:], a.model.key('home').ctrl[12:])

    def test_infeasible_state_is_reported_even_when_qp_returns(self):
        a = self.agent()
        configure_cbf(a, 'on')
        obs = self.observation(a)
        obs[2] = -.5
        plans = np.tile(a.model.key(0).ctrl, (a.horizon, 1))
        filtered = a.cbf.filter_plan(plans, obs)
        self.assertTrue(a.cbf.last['infeasible'])
        self.assertLess(a.cbf.last['verified_min_residual'], 0.)
        self.assertTrue(np.all(np.isfinite(filtered)))

    def test_velocity_task_accepts_on_and_off_without_position_barrier(self):
        a = EEVelocityController()
        self.addCleanup(a.shutdown)
        configure_cbf(a, 'on')
        self.assertTrue(a.cbf.enabled)
        self.assertTrue(a.cbf.velocity_enabled)
        self.assertFalse(a.cbf.ee_enabled)
        self.assertFalse(hasattr(a, 'ee_objective'))
        configure_cbf(a, 'off')
        self.assertFalse(a.cbf.enabled)

    def test_cli_default_and_velocity_support(self):
        path = Path(__file__).resolve().parents[1]/'scripts/simulate_mppi.py'
        spec = importlib.util.spec_from_file_location('simulate_cbf_cli', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch('sys.argv', ['simulate_mppi.py', '--task', 'walk_straight']):
            self.assertEqual(module.parse_args().cbf, 'off')
        with patch('sys.argv', ['simulate_mppi.py', '--task', 'ee_walk', '--robot', 'go1_dm', '--cbf', 'on']):
            self.assertEqual(module.parse_args().cbf, 'on')
        with patch('sys.argv', ['simulate_mppi.py', '--task', 'ee_walk_cmd_vel', '--robot', 'go1_dm', '--cbf', 'on']):
            self.assertEqual(module.parse_args().cbf, 'on')

    def test_result_export_records_cbf_and_counts_extra_rollouts(self):
        import json
        a = self.agent()
        configure_cbf(a, 'on')
        sim = Simulator(agent=a, viewer=False, T=3, model_path=PACKAGE_ROOT/a.task_data['sim_path'])
        sim.run()
        with tempfile.TemporaryDirectory() as root:
            folder = sim.save_results('mppi', root)
            summary = json.loads((folder/'summary.json').read_text())
            config = json.loads((folder/'config.json').read_text())
            saved = np.load(folder/'trajectory.npz')
            self.assertTrue(summary['cbf']['enabled'])
            self.assertGreater(summary['cbf']['extra_rollouts_total'], 0)
            self.assertTrue(config['execution']['cbf']['enabled'])
            self.assertEqual(saved['cbf_margin'].shape, saved['time'].shape)
            self.assertEqual(np.isfinite(saved['cbf_verified_residual']).sum(), 2)


if __name__ == '__main__':
    unittest.main()
