"""Check weighted variance, covariance shift, and invalid-rollout handling."""
import unittest
from types import SimpleNamespace
import numpy as np
from whole_body_mppi.control.solvers.cma_mppi import optimize_cma_actions


class CMATests(unittest.TestCase):
    def make_agent(self, sample_type):
        return SimpleNamespace(
            sample_type=sample_type, horizon=5, n_knots=3, act_dim=1,
            n_samples=4, n_iterations=1, noise_sigma=np.array([0.2]),
            act_min=np.array([-1.]), act_max=np.array([1.]), temperature=0.1,
            num_workers=1, state_rollouts=np.zeros((4, 5, 1)),
            random_generator=np.random.default_rng(7), trajectory=np.zeros((5, 1)),
            cma_covariance=None, rollout_func=lambda *args, **kwargs: None,
            cma_params=dict(covariance_learning_rate=0.5, sigma_min=0.05,
                            sigma_max=3., elite_ratio=0., reward_normalization=True))

    def test_weighted_variance_and_one_step_shift(self):
        for sampling in ('normal', 'cubic'):
            with self.subTest(sampling=sampling):
                agent = self.make_agent(sampling)
                candidates = []
                def cost(actions):
                    candidates.append(actions.copy())
                    return np.zeros(4)  # equal weights, including zero cost span
                result = optimize_cma_actions(agent, np.zeros(2), cost)
                np.testing.assert_allclose(result, candidates[0].mean(axis=0))
                indices = np.array([0, 2, 4]) if sampling == 'cubic' else np.arange(5)
                variance = candidates[0][:, indices].var(axis=0)
                expected = np.clip(0.5*0.2**2 + 0.5*variance, 0.05**2, 3.**2)
                np.testing.assert_allclose(agent.cma_covariance_history[0], expected)
                shifted = np.interp(indices+1, indices, expected[:, 0])
                shifted[-1] = 0.2**2
                np.testing.assert_allclose(agent.cma_covariance[:, 0], shifted)

    def test_nonfinite_candidates(self):
        agent = self.make_agent('normal')
        result = optimize_cma_actions(agent, np.zeros(2), lambda a: np.array([0., np.inf, 1., np.nan]))
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(agent.exp_weights[1], 0.)
        self.assertEqual(agent.exp_weights[3], 0.)
        with self.assertRaises(FloatingPointError):
            optimize_cma_actions(agent, np.zeros(2), lambda a: np.full(4, np.inf))


if __name__ == '__main__':
    unittest.main()
