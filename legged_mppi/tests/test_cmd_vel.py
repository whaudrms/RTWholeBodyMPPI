"""Command task semantics independent of waypoint position and absolute yaw."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from whole_body_mppi.control.tasks.locomotion import LocomotionController
from whole_body_mppi.control.costs.cmd_vel import heading_velocity


class CommandVelocityTests(unittest.TestCase):
    def setUp(self):
        self.agent = LocomotionController('walk_straight_cmd_vel')
        self.addCleanup(self.agent.shutdown)
        self.agent.joints_ref = self.agent.gait_scheduler.gait[:, self.agent.gait_scheduler.indices[:self.agent.horizon]]
        self.states = np.zeros((1, self.agent.horizon, 38))
        self.states[..., 3] = .27
        self.states[..., 4] = 1
        self.actions = np.zeros((1, self.agent.horizon, 12))

    def test_cost_is_translation_and_heading_invariant_and_preserves_states(self):
        self.agent.body_ref[7] = .2
        self.states[..., 20] = .2
        original = self.states.copy()
        baseline = self.agent.task_cost(self.states, self.actions)
        np.testing.assert_array_equal(self.states, original)
        rotated = original.copy()
        rotated[..., 1:3] = [100, -50]
        rotated[..., 4:8] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
        rotated[..., 20:22] = [0, .2]
        np.testing.assert_allclose(self.agent.task_cost(rotated, self.actions), baseline)
        rotated[..., 20:22] = 0
        self.assertGreater(self.agent.task_cost(rotated, self.actions)[0], baseline[0])

    def test_command_ramp_and_no_waypoint_completion(self):
        self.agent.solve_task = lambda obs: np.zeros(12)
        self.agent.set_cmd_vel(.2, -.1, .3)
        self.agent.update(self.states[0, 0, 1:])
        np.testing.assert_allclose(self.agent.command_velocity, [.005, -.005, .01])
        self.agent.next_goal()
        self.assertFalse(self.agent.task_success)
        self.assertEqual(self.agent.goal_index, 0)
        with self.assertRaises(ValueError):
            self.agent.set_cmd_vel(np.nan, 0, 0)

    def test_yaw_rate_with_tilt(self):
        roll, pitch, yaw = .2, -.3, .8
        rotation = Rotation.from_euler('xyz', [roll, pitch, yaw])
        # Pure world vertical rotation gives the requested Euler yaw rate.
        body_omega = rotation.inv().apply([0, 0, .4])[None]
        quat = rotation.as_quat()[[3, 0, 1, 2]][None]
        _, rate, _ = heading_velocity(quat, np.zeros((1, 3)), body_omega)
        np.testing.assert_allclose(rate, [.4])

    def test_rejects_arm_robot(self):
        with self.assertRaises(ValueError):
            LocomotionController('walk_straight_cmd_vel', robot='go1_dm')


if __name__ == '__main__':
    unittest.main()
