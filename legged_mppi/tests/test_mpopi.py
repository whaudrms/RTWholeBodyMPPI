"""MPOPI proposal moments, last-cycle output, and bounded sampling checks."""
import unittest
from types import SimpleNamespace
import numpy as np

from whole_body_mppi.control.solvers.mpopi import (
    adapt_proposal, exponential_weights, normalized_rank_weights, optimize_mpopi_actions)


class MPOPITests(unittest.TestCase):
    def make_solver(self, sampling='normal'):
        return SimpleNamespace(
            sample_type=sampling, horizon=5, n_knots=3, act_dim=2, n_samples=8,
            n_iterations=2, noise_sigma=np.array([0.2, 0.3]),
            act_min=np.array([-1., -1.]), act_max=np.array([1., 1.]), temperature=0.1,
            num_workers=1, state_rollouts=np.zeros((12, 5, 1)),
            random_generator=np.random.default_rng(7), trajectory=np.zeros((5, 2)),
            rollout_func=lambda *args, **kwargs: None,
            mpopi_params=dict(elite_fraction=0.5, learning_rate=0.5, covariance_floor=1e-3))

    def test_old_mean_moment_and_full_covariance(self):
        mean = np.array([[1., 2.]])
        covariance = np.eye(2)[None]
        elites = np.array([[[2., 4.]], [[3., 3.]]])
        weights = np.array([0.75, 0.25])
        new_mean, new_cov = adapt_proposal(mean, covariance, elites, weights, 0.5, 1e-3)
        np.testing.assert_allclose(new_mean, [[1.625, 2.875]])
        # Old-centered moment = [[1.75, 2], [2, 3.25]].
        np.testing.assert_allclose(new_cov, [[[1.375, 1.], [1., 2.125]]])
        _, floored = adapt_proposal(mean, covariance, mean[None], [1.], 1., 0.01)
        np.testing.assert_allclose(floored, np.eye(2)[None] * 0.01)

    def test_last_cycle_output_and_elite_update(self):
        for sampling in ('normal', 'cubic'):
            with self.subTest(sampling=sampling):
                solver = self.make_solver(sampling)
                batches = []
                costs = [np.array([0., 1., 2., 3.]), np.array([100., 103., 101., 102.])]
                def cost(actions):
                    batches.append(actions.copy())
                    return costs[len(batches)-1]
                result = optimize_mpopi_actions(solver, np.zeros(2), cost)
                weights = exponential_weights(costs[-1], solver.temperature)
                np.testing.assert_allclose(result, np.einsum('n,nhj->hj', weights, batches[-1]))
                indices = np.array([0, 2, 4]) if sampling == 'cubic' else np.arange(5)
                # First sample is nominal; next proposal uses the best two samples.
                np.testing.assert_allclose(batches[0][0], 0., atol=1e-15)
                ranks = normalized_rank_weights(2)
                expected = 0.5 * np.einsum('n,nkj->kj', ranks, batches[0][:2, indices])
                np.testing.assert_allclose(solver.mpopi_proposal_history[1], expected)
                self.assertTrue((batches[-1] >= -1.).all())
                self.assertTrue((batches[-1] <= 1.).all())
                values = np.linalg.eigvalsh(solver.mpopi_covariance_history[-1])
                self.assertTrue((values >= 1e-3 - 1e-12).all())

    def test_nonfinite_weights_and_zero_noise(self):
        weights = exponential_weights(np.array([np.nan, 0., np.inf, 1.]), 0.1)
        np.testing.assert_array_equal(weights[[0, 2]], 0.)
        self.assertAlmostEqual(weights.sum(), 1.)
        with self.assertRaises(FloatingPointError):
            exponential_weights(np.full(4, np.inf), 0.1)
        solver = self.make_solver()
        solver.noise_sigma[0] = 0.
        with self.assertRaises(ValueError):
            optimize_mpopi_actions(solver, np.zeros(2), lambda actions: np.zeros(4))

    def test_single_iteration_and_partial_finite_elites(self):
        solver = self.make_solver()
        solver.n_iterations = 1
        solver.n_samples = 4
        plan = optimize_mpopi_actions(solver, np.zeros(2), lambda actions: np.zeros(4))
        np.testing.assert_allclose(plan, solver.mpopi_final_actions.mean(axis=0))
        solver.n_iterations = 3
        solver.n_samples = 12
        plan = optimize_mpopi_actions(solver, np.zeros(2),
                                     lambda actions: np.array([0., np.nan, np.inf, np.inf]))
        self.assertTrue(np.isfinite(plan).all())
        np.testing.assert_allclose(plan, solver.mpopi_final_actions[0])


if __name__ == '__main__':
    unittest.main()
