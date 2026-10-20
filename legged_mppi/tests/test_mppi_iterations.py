"""Iteration scheduling and annealing checks for the gait-based controllers."""
import unittest
import argparse
import contextlib
import io
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

from whole_body_mppi.control.tasks.locomotion import LocomotionController as MPPI
from whole_body_mppi.control.tasks.locomanipulation import LocomanipulationController as MPPI_box_push


from whole_body_mppi.control.solvers.options import (
    add_mppi_mode_arguments, apply_mppi_mode)


class IterationTests(unittest.TestCase):
    def test_mode_options(self):
        parser = argparse.ArgumentParser()
        add_mppi_mode_arguments(parser)
        for flags in ([], ['--dial-mpc'], ['--mppi'], ['--cma-mppi'], ['--mpopi'], ['--mbd']):
            agent = SimpleNamespace(n_iterations=3, beta_iteration=0.5, beta_horizon=0.5)
            apply_mppi_mode(agent, parser.parse_args(flags).mppi_mode)
            self.assertEqual(agent.n_iterations, 1 if flags == ['--mppi'] else 3)
            self.assertEqual(agent.beta_iteration, np.inf if flags == ['--mppi'] else 0.5)
            self.assertEqual(agent.beta_horizon, np.inf if flags == ['--mppi'] else 0.5)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(['--dial-mpc', '--mppi'])

    def test_updates_keep_references_and_advance_once(self):
        for cls, task, mode in ((MPPI, 'walk_straight', None),
                                (MPPI_box_push, 'push_box', None),
                                (MPPI, 'walk_straight', 'cma-mppi'),
                                (MPPI_box_push, 'push_box', 'cma-mppi'),
                                (MPPI, 'walk_straight', 'mpopi'),
                                (MPPI_box_push, 'push_box', 'mpopi'),
                                (MPPI, 'walk_straight', 'mbd'),
                                (MPPI_box_push, 'push_box', 'mbd')):
            with self.subTest(task=task):
                agent = cls(task)
                apply_mppi_mode(agent, mode)
                try:
                    data = mujoco.MjData(agent.model)
                    mujoco.mj_resetDataKeyframe(agent.model, data, 0)
                    obs = np.concatenate((data.qpos, data.qvel))
                    refs = []
                    original_cost = agent.cost_func

                    def cost(*args):
                        refs.append((args[-2].copy(), args[-1].copy()))
                        return original_cost(*args)

                    agent.cost_func = cost
                    with patch.object(agent.gait_scheduler, 'roll', wraps=agent.gait_scheduler.roll) as roll:
                        action = agent.update(obs)
                        self.assertEqual(roll.call_count, 1)
                    self.assertEqual(len(refs), 3)
                    self.assertTrue(np.isfinite(action).all())
                    for joints, body in refs[1:]:
                        np.testing.assert_array_equal(joints, refs[0][0])
                        np.testing.assert_array_equal(body, refs[0][1])
                    np.testing.assert_array_equal(agent.trajectory[:-1], agent.selected_trajectory[1:])
                    if mode == 'mpopi':
                        self.assertEqual(len(agent.mpopi_covariance_history), 3)
                        agent.update(obs)
                        np.testing.assert_allclose(agent.mpopi_covariance_history[0],
                            np.broadcast_to(np.eye(12), (agent.n_knots, 12, 12)))
                    if mode == 'cma-mppi':
                        self.assertEqual(len(agent.cma_covariance_history), 3)
                        cov = agent.cma_covariance.copy()
                        self.assertTrue(np.isfinite(cov).all())
                        self.assertTrue((cov >= 0.05**2).all())
                        self.assertTrue((cov <= 3.0**2).all())
                        agent.update(obs)
                        self.assertEqual(len(agent.cma_covariance_history), 3)
                        agent.reset_planner()
                        self.assertIsNone(agent.cma_covariance)
                    # Identical draws expose the exact iteration covariance factor.
                    agent.random_generator = np.random.default_rng(42)
                    first = agent.generate_noise((2, 4, 12), 0)
                    agent.random_generator = np.random.default_rng(42)
                    second = agent.generate_noise((2, 4, 12), 1)
                    np.testing.assert_allclose(second, first * np.exp(-0.5 / (0.5 * 3)))
                finally:
                    agent.shutdown()


if __name__ == '__main__':
    unittest.main()
