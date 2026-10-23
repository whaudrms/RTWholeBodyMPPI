"""DCBF initial transition, bounded repair and failure reporting."""
import unittest
from types import SimpleNamespace
import numpy as np
from whole_body_mppi.control.safety.shield import SafetyShield
from whole_body_mppi.control.solvers.options import apply_mppi_mode


class ShieldTests(unittest.TestCase):
    def make_shield(self):
        def rollout(states, controls, initial, **kwargs):
            states[:] = initial[:, None, :]
            states[..., 3] = controls[..., 0]
        agent = SimpleNamespace(horizon=1, act_dim=1, act_min=np.array([.1]),
                                act_max=np.array([.5]), state_rollouts=np.empty((1, 1, 8)),
                                rollout_func=rollout)
        config = dict(alpha=.9, cost_weight=10., min_height=.18, max_tilt_deg=30.,
                      repair_steps=1, max_iterations=5, max_evaluations=30,
                      tolerance=1e-6, fd_step=1e-5)
        return SafetyShield(agent, config)

    def test_initial_transition_cost(self):
        shield = self.make_shield()
        obs = np.array([0, 0, .3, 1, 0, 0, 0])
        states = np.r_[0, obs][None, None].copy()
        states[..., 3] = .2
        self.assertGreater(shield.cost(obs, states)[0], 0)  # Both heights safe, contraction violated.
        states[..., 3] = .3
        self.assertEqual(shield.cost(obs, states)[0], 0)

    def test_bounded_repair_improves_and_keeps_nominal_unchanged(self):
        shield = self.make_shield()
        obs = np.array([0, 0, .3, 1, 0, 0, 0])
        nominal = np.array([[.2]])
        result = shield.repair(obs, nominal)
        self.assertTrue(shield.history[-1]['feasible'])
        self.assertLess(shield.history[-1]['after'], shield.history[-1]['before'])
        self.assertTrue(np.all(result <= .5))
        np.testing.assert_array_equal(nominal, [[.2]])

    def test_exhausted_budget_reports_violation(self):
        shield = self.make_shield()
        shield.cfg['max_evaluations'] = 1
        shield.repair(np.array([0, 0, .3, 1, 0, 0, 0]), np.array([[.2]]))
        self.assertFalse(shield.history[-1]['feasible'])
        self.assertEqual(shield.history[-1]['evaluations'], 1)

    def test_unsupported_task_rejected(self):
        with self.assertRaises(ValueError):
            apply_mppi_mode(SimpleNamespace(task='stairs', robot='go1'), 'shield-mppi')


if __name__ == '__main__':
    unittest.main()
