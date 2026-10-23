"""Independent optimizer/safety selection and short integration coverage."""
import argparse
import contextlib
import io
import unittest
import numpy as np
from whole_body_mppi.control.solvers.options import add_mppi_mode_arguments, apply_mppi_mode
from whole_body_mppi.control.tasks.locomotion import LocomotionController


class SafetyOptionsTests(unittest.TestCase):
    def test_parser_independent_groups(self):
        parser = argparse.ArgumentParser()
        add_mppi_mode_arguments(parser, include_safety=True)
        args = parser.parse_args(['--mbd', '--shield-mppi'])
        self.assertEqual((args.mppi_mode, args.safety_mode), ('mbd', 'shield-mppi'))
        for flags in (['--mpopi', '--mbd'], ['--shield-mppi', '--clf-cbf-qp']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(flags)

    def test_all_optimizer_safety_combinations(self):
        for optimizer in ('mppi', 'dial-mpc', 'cma-mppi', 'mpopi', 'mbd'):
            for safety in ('shield-mppi', 'clf-cbf-qp'):
                with self.subTest(optimizer=optimizer, safety=safety):
                    agent = LocomotionController('walk_straight_cmd_vel')
                    try:
                        apply_mppi_mode(agent, optimizer, safety)
                        self.assertEqual(agent.mppi_mode, optimizer)
                        self.assertEqual(agent.safety_mode, safety)
                        self.assertEqual(agent.n_iterations, 1 if optimizer == 'mppi' else agent.params['n_iterations'])
                        agent.sampling_init = agent.model.key('home').ctrl.copy()
                        agent.reset_planner()
                        obs = np.r_[agent.model.key('home').qpos, np.zeros(agent.model.nv)]
                        command = agent.update(obs)
                        self.assertTrue(np.isfinite(command).all())
                        self.assertEqual(len(agent.iteration_sample_counts), agent.n_iterations)
                        self.assertEqual(sum(agent.iteration_sample_counts), agent.n_samples)
                        self.assertEqual(len((agent.shield or agent.qp_filter).history), 1)
                    finally:
                        agent.shutdown()

    def test_safety_only_defaults_to_mppi_and_switch_restores_iterations(self):
        agent = LocomotionController('walk_straight_cmd_vel')
        try:
            apply_mppi_mode(agent, None, 'clf-cbf-qp')
            self.assertEqual(agent.mppi_mode, 'mppi')
            apply_mppi_mode(agent, 'mpopi', 'shield-mppi')
            self.assertEqual(agent.n_iterations, agent.params['n_iterations'])
            self.assertIsNone(agent.qp_filter)
        finally:
            agent.shutdown()


if __name__ == '__main__':
    unittest.main()
