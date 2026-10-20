"""Numerical diffusion checks and bounded solver integration."""
import unittest
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import CubicSpline

from whole_body_mppi.control.solvers.mbd import (
    diffusion_schedule, forward_process, estimate_score, reverse_step, optimize_mbd_actions)


class MBDTests(unittest.TestCase):
    def make_solver(self, sampling='normal', iterations=3):
        return SimpleNamespace(
            sample_type=sampling, horizon=5, n_knots=3, act_dim=2,
            n_samples=6*iterations, n_iterations=iterations, noise_sigma=np.array([0.2, 0.3]),
            act_min=np.array([-1., -2.]), act_max=np.array([1., 1.]), temperature=0.1,
            num_workers=1, state_rollouts=np.zeros((6*iterations, 5, 1)),
            random_generator=np.random.default_rng(7), trajectory=np.zeros((5, 2)),
            rollout_func=lambda *args, **kwargs: None,
            mbd_params=dict(alpha_bar_first=0.98, alpha_bar_last=0.2))

    def test_forward_schedule_and_reverse_equations(self):
        bars, alphas = diffusion_schedule(3, 0.98, 0.2)
        np.testing.assert_allclose(np.cumprod(alphas), bars[1:])
        clean = np.array([[1., 2.]])
        noise = np.arange(6).reshape(3, 1, 2) / 10
        trace = forward_process(clean, alphas, noise)
        expected = clean.copy()
        np.testing.assert_array_equal(trace[0], clean)
        for index, alpha in enumerate(alphas):
            expected = np.sqrt(alpha)*expected + np.sqrt(1-alpha)*noise[index]
            np.testing.assert_allclose(trace[index+1], expected)
        weighted_clean = np.array([[3., 4.]])
        score = estimate_score(trace[-1], weighted_clean, bars[-1])
        latent = reverse_step(trace[-1], score, alphas[-1], bars[-1])
        np.testing.assert_allclose(latent, np.sqrt(bars[-2])*weighted_clean)
        np.testing.assert_allclose(diffusion_schedule(1, 0.98, 0.2)[0], [1., 0.2])
        with self.assertRaises(ValueError):
            forward_process(clean, alphas, np.zeros((2, 1, 2)))

    def test_trace_rollout_budget_and_final_plan(self):
        for sampling in ('normal', 'cubic'):
            for iterations in (1, 3):
                with self.subTest(sampling=sampling, iterations=iterations):
                    solver = self.make_solver(sampling, iterations)
                    batches = []
                    def cost(actions):
                        batches.append(actions.copy())
                        return np.sum(actions**2, axis=(1, 2))
                    plan = optimize_mbd_actions(solver, np.zeros(2), cost)
                    points = 3 if sampling == 'cubic' else 5
                    self.assertEqual(solver.mbd_forward_trace.shape, (iterations+1, points, 2))
                    self.assertEqual(solver.mbd_backward_trace.shape, (iterations+1, points, 2))
                    self.assertEqual(solver.mbd_score_trace.shape, (iterations, points, 2))
                    self.assertEqual(sum(len(a) for a in batches), 6*iterations)
                    np.testing.assert_array_equal(solver.mbd_backward_trace[0], solver.mbd_forward_trace[-1])
                    indices = np.array([0, 2, 4]) if sampling == 'cubic' else np.arange(5)
                    weighted = np.einsum('n,nkj->kj', solver.exp_weights, batches[-1][:, indices])
                    expected = CubicSpline(indices, weighted, axis=0)(np.arange(5)) if sampling == 'cubic' else weighted
                    np.testing.assert_allclose(plan, np.clip(expected, solver.act_min, solver.act_max), atol=1e-13)
                    for actions in batches:
                        self.assertTrue((actions >= solver.act_min).all())
                        self.assertTrue((actions <= solver.act_max).all())
                    # The exact same seed and warm start reproduce the plan.
                    twin = self.make_solver(sampling, iterations)
                    twin_plan = optimize_mbd_actions(twin, np.zeros(2), lambda a: np.sum(a**2, axis=(1, 2)))
                    np.testing.assert_array_equal(twin_plan, plan)

    def test_invalid_parameters_and_candidates(self):
        for first, last in ((1., 0.2), (0.2, 0.3), (0.98, 0.), (np.nan, 0.2)):
            with self.assertRaises(ValueError):
                diffusion_schedule(3, first, last)
        solver = self.make_solver()
        plan = optimize_mbd_actions(solver, np.zeros(2),
                                   lambda a: np.array([0., np.nan, np.inf, np.inf, np.inf, np.inf]))
        self.assertTrue(np.isfinite(plan).all())
        np.testing.assert_array_equal(solver.exp_weights[1:], 0.)
        with self.assertRaises(FloatingPointError):
            optimize_mbd_actions(solver, np.zeros(2), lambda a: np.full(6, np.inf))
        solver.noise_sigma[0] = 0.
        with self.assertRaises(ValueError):
            optimize_mbd_actions(solver, np.zeros(2), lambda a: np.zeros(6))


if __name__ == '__main__':
    unittest.main()
