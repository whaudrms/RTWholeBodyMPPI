"""EE velocity tasks retain arm objectives and use heading-frame body tracking."""
import unittest
import numpy as np
from whole_body_mppi.control.tasks.whole_body import WholeBodyController


class EECommandTests(unittest.TestCase):
    def test_continuous_task_commands_and_heading_cost(self):
        for task, target in [('ee_reach_cmd_vel', [0, 0, 0]), ('ee_walk_cmd_vel', [.2, 0, 0])]:
            with self.subTest(task=task):
                a = WholeBodyController(task)
                try:
                    np.testing.assert_allclose(a.command_target, target)
                    self.assertEqual(a.act_dim, 16)
                    a.next_goal()
                    self.assertFalse(a.task_success)
                    a.solve_task = lambda obs: np.zeros(16)
                    obs = np.r_[a.model.key('home').qpos, np.zeros(a.model.nv)]
                    self.assertEqual(a.update(obs).shape, (18,))
                    a.set_cmd_vel(.2, 0, .1)
                    a.update(obs)
                    self.assertGreater(a.command_velocity[2], 0)
                    x = np.tile(obs, (1, a.horizon, 1))
                    u = a.rollout_func.expand_actions(np.tile(a.sampling_init, (1, a.horizon, 1)))
                    x[..., 25:28] = [.2, 0, 0]
                    original = x.copy()
                    base = a.cost_func(x, u, a.joints_ref, a.body_ref)
                    np.testing.assert_array_equal(x, original)
                    x[..., :2] = [100, -100]
                    x[..., 3:7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
                    x[..., 25:28] = [0, .2, 0]
                    np.testing.assert_allclose(a.cost_func(x, u, a.joints_ref, a.body_ref), base)
                    # A world-frame EE target is intentionally NOT translation invariant.
                    self.assertEqual(a.ee_objective.frame, 'world' if 'reach' in task else 'base')
                    self.assertEqual(a.noise_sigma.shape, (16,))
                finally:
                    a.shutdown()


if __name__ == '__main__':
    unittest.main()
