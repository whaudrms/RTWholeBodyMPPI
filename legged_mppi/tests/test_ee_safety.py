"""Reduced control mapping and independent body/EE CLFs."""
import unittest
import numpy as np
from whole_body_mppi.control.tasks.whole_body import WholeBodyController
from whole_body_mppi.control.solvers.options import apply_mppi_mode


class EESafetyTests(unittest.TestCase):
    def test_filters_keep_wrist_fixed_and_use_16_controls(self):
        for task in ('ee_reach_cmd_vel', 'ee_walk_cmd_vel'):
            for safety in ('shield-mppi', 'clf-cbf-qp'):
                with self.subTest(task=task, safety=safety):
                    a = WholeBodyController(task)
                    try:
                        apply_mppi_mode(a, 'mpopi', safety)
                        a.set_params(5, .1, max(6, 2*a.n_iterations))
                        obs = np.r_[a.model.key('home').qpos, np.zeros(a.model.nv)]
                        action = a.update(obs)
                        self.assertEqual(a.selected_trajectory.shape, (5,16))
                        self.assertEqual(action.shape, (18,))
                        np.testing.assert_array_equal(action[16:], a.model.key('home').ctrl[16:])
                        expanded = a.rollout_func.expand_actions(a.selected_trajectory)
                        np.testing.assert_array_equal(expanded[:,16:], np.tile(action[16:], (5,1)))
                        # The wrapper and explicitly expanded backend must predict identically.
                        states = np.empty((1,5,a.state_rollouts.shape[-1])); direct = np.empty_like(states)
                        initial = np.r_[0., obs][None]
                        a.rollout_func(states, a.selected_trajectory[None], initial, num_workers=1, nstep=5)
                        a.rollout_backend.threaded_rollout(direct, expanded[None], initial, num_workers=1, nstep=5)
                        np.testing.assert_allclose(states, direct, atol=1e-10)
                        if a.qp_filter is not None:
                            self.assertEqual(a.qp_filter.previous_solution.shape, (18,))
                            self.assertIn('ee_qp_slack', a.qp_filter.history[-1])
                            self.assertEqual(a.qp_filter.n_clf, 2)
                    finally:
                        a.shutdown()

    def test_clf_body_indices_and_ee_relative_frame(self):
        a = WholeBodyController('ee_walk_cmd_vel')
        try:
            apply_mppi_mode(a, 'mppi', 'clf-cbf-qp')
            obs = np.r_[a.model.key('home').qpos, np.zeros(a.model.nv)]
            obs[a.model.nq] = .2  # world x base velocity, not an arm coordinate
            values = a.qp_filter.values(np.r_[0., obs], np.zeros(6))[0]
            self.assertAlmostEqual(values[0], .5)
            self.assertAlmostEqual(values[1], 0.)  # fixed arm has zero relative velocity
            obs[a.model.nq+18] = .3
            changed = a.qp_filter.values(np.r_[0., obs], np.zeros(6))[0]
            self.assertAlmostEqual(changed[0], values[0])
            self.assertGreater(changed[1], 0.)
        finally:
            a.shutdown()


if __name__ == '__main__':
    unittest.main()
