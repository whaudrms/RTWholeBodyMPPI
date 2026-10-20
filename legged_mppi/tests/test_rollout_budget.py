"""All solvers must consume the same total trajectory budget."""
import unittest
import mujoco
import numpy as np
from whole_body_mppi.control.solvers.base import MPPISolver
from whole_body_mppi.control.solvers.budget import sample_counts
from whole_body_mppi.control.solvers.options import apply_mppi_mode


class RolloutBudgetTests(unittest.TestCase):
    def test_partition_and_validation(self):
        self.assertEqual(sample_counts(10, 3), [4, 3, 3])
        self.assertEqual(sample_counts(11, 3), [4, 4, 3])
        self.assertEqual(sample_counts(30, 3), [10, 10, 10])
        self.assertEqual(sample_counts(10, 1), [10])
        for total, iterations in ((5, 3), (10, 0), (10.5, 3)):
            with self.assertRaises(ValueError):
                sample_counts(total, iterations)

    def test_all_modes_use_exact_budget_and_active_buffers(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
            <joint name="j" type="slide"/><geom size="0.1" mass="1"/>
            </body></worldbody><actuator><motor joint="j"/></actuator></mujoco>''')
        for sampling in ('normal', 'cubic'):
            for total in (10, 11, 30):
                for mode in ('mppi', 'dial-mpc', 'cma-mppi', 'mpopi', 'mbd'):
                    with self.subTest(sampling=sampling, total=total, mode=mode):
                        params = dict(dt=.01, horizon=5, n_samples=total, n_iterations=3,
                                      noise_sigma=[.1], n_workers=5, sample_type=sampling,
                                      n_knots=3, seed=42, **{'lambda': .1})
                        solver = MPPISolver(model, params, initial_action=[0.],
                                            action_min=[-1.], action_max=[1.])
                        try:
                            apply_mppi_mode(solver, mode)
                            counts = []
                            def rollout(states, actions, initial, **kwargs):
                                count = len(actions)
                                self.assertEqual(len(initial), count)
                                self.assertEqual(len(states), count)
                                # Poison the inactive part so a full-buffer cost
                                # callback would fail, including after a resize.
                                solver._rollout_storage[:] = np.nan
                                states[:] = 0.
                                counts.append(count)
                            solver.rollout_func = rollout
                            def cost(actions):
                                self.assertEqual(len(solver.state_rollouts), len(actions))
                                self.assertTrue(np.isfinite(solver.state_rollouts).all())
                                return np.sum(actions**2, axis=(1, 2))
                            for budget in (total, total + 1):
                                counts.clear()
                                solver.set_params(5, .1, budget)
                                result = solver.optimize_actions(np.zeros(2), cost)
                                self.assertEqual(sum(counts), budget)
                                self.assertEqual(counts, sample_counts(budget, solver.n_iterations))
                                self.assertEqual(solver.iteration_sample_counts, counts)
                                self.assertEqual(len(solver.exp_weights), counts[-1])
                                self.assertTrue(np.isfinite(result).all())
                                solver.commit_plan(result)
                        finally:
                            solver.shutdown()


if __name__ == '__main__':
    unittest.main()
