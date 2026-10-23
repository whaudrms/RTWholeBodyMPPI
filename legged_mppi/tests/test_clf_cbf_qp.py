"""OSQP filter acceptance, unchanged future plan, and fail-closed behavior."""
import unittest
from types import SimpleNamespace
import numpy as np
from whole_body_mppi.control.safety.clf_cbf_qp import CLFCBFQP, FilterFailure


def make_filter():
    agent = SimpleNamespace(act_dim=12, horizon=40, h=.01,
                            act_min=np.full(12, -1.), act_max=np.ones(12), command_velocity=np.zeros(3))
    config = dict(velocity_scale=[.2, .2, .5], max_correction=.05, decay_rate=1.,
                  slack_weight=100., fd_step=.0001, tolerance=.0001, solver_tolerance=1e-6,
                  max_iterations=4000)
    barrier = dict(alpha=.9, cost_weight=10000., min_height=.18, max_tilt_deg=10.,
                   repair_steps=3, max_iterations=3, max_evaluations=150, tolerance=1e-6, fd_step=1e-4)
    filt = CLFCBFQP(agent, config, barrier)
    # Controlled affine residual fixture; exercises actual OSQP assembly/solve.
    filt.values = lambda states, reference: np.array([[1., 1., 1.]])
    filt.predict = lambda obs, controls, reference: np.column_stack(
        (1.-controls[:, 0], np.ones(len(controls)), np.ones(len(controls))))
    return filt


class QPTests(unittest.TestCase):
    def test_accepts_and_only_changes_first_command(self):
        filt = make_filter()
        nominal = np.zeros((40, 12))
        output = filt.repair(np.zeros(37), nominal)
        self.assertTrue(filt.history[-1]['accepted'])
        self.assertGreater(output[0, 0], 0)
        self.assertLessEqual(abs(output[0, 0]), .05)
        np.testing.assert_array_equal(output[1:], nominal[1:])
        np.testing.assert_array_equal(nominal, np.zeros_like(nominal))
        self.assertGreaterEqual(filt.history[-1]['slack'], 0)

    def test_hard_cbf_infeasible_stops(self):
        filt = make_filter()
        filt.predict = lambda obs, controls, reference: np.tile([1., -1., 1.], (len(controls), 1))
        with self.assertRaises(FilterFailure):
            filt.repair(np.zeros(37), np.zeros((40, 12)))
        self.assertIsNotNone(filt.failure)
        self.assertFalse(filt.history[-1]['accepted'])
        self.assertIn('infeasible', filt.history[-1]['status'])

    def test_nonlinear_recheck_failure_stops(self):
        filt = make_filter()
        predict = filt.predict
        def nonlinear(obs, controls, reference):
            values = predict(obs, controls, reference)
            if len(controls) == 1:  # Recheck rejects an optimistic local model.
                values[:, 1] = -1
            return values
        filt.predict = nonlinear
        with self.assertRaisesRegex(FilterFailure, 'Nonlinear'):
            filt.repair(np.zeros(37), np.zeros((40, 12)))
        self.assertFalse(filt.history[-1]['accepted'])

    def test_soft_clf_mismatch_is_logged_not_rejected(self):
        filt = make_filter()
        predict = filt.predict
        def nonlinear(obs, controls, reference):
            values = predict(obs, controls, reference)
            if len(controls) == 1:
                values[:, 0] += .1
            return values
        filt.predict = nonlinear
        filt.repair(np.zeros(37), np.zeros((40, 12)))
        row = filt.history[-1]
        self.assertTrue(row['accepted'])
        self.assertGreater(row['actual_required_slack'], row['qp_slack'])
        self.assertGreater(row['clf_slack_excess'], .09)
        self.assertFalse(row['clf_decay_satisfied'])
        self.assertFalse(row['clf_decreased'])

    def test_nonfinite_nonlinear_prediction_still_stops(self):
        filt = make_filter()
        predict = filt.predict
        def nonlinear(obs, controls, reference):
            values = predict(obs, controls, reference)
            if len(controls) == 1:
                values[:, 0] = np.nan
            return values
        filt.predict = nonlinear
        with self.assertRaisesRegex(FilterFailure, 'Non-finite nonlinear'):
            filt.repair(np.zeros(37), np.zeros((40, 12)))

    def test_velocity_clf_is_sum_of_normalized_components(self):
        filt = make_filter()
        states = np.zeros((1, 38))
        states[:, 3] = .27
        states[:, 4] = 1
        states[:, 20:22] = [.2, -.2]
        states[:, 25] = .5
        values = CLFCBFQP.values(filt, states, np.zeros(3))
        self.assertAlmostEqual(values[0, 0], 1.5)

    def test_nonfinite_command_stops(self):
        filt = make_filter()
        nominal = np.zeros((40, 12)); nominal[0, 0] = np.nan
        with self.assertRaises(FilterFailure):
            filt.repair(np.zeros(37), nominal)


if __name__ == '__main__':
    unittest.main()
