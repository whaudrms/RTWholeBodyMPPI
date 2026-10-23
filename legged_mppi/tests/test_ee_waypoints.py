"""Shared waypoint progression for position and velocity EE controllers."""
import unittest
from unittest.mock import patch
import numpy as np
import mujoco
from whole_body_mppi.control.tasks.whole_body import WholeBodyController
from whole_body_mppi.control.tasks.ee_velocity import EEVelocityController
from whole_body_mppi.control.tasks.definitions import get_task


class EEWaypointTests(unittest.TestCase):
    def test_all_tasks_advance_only_after_settling_and_hold_final_target(self):
        for task in ('ee_reach', 'ee_walk'):
            with self.subTest(task=task):
                cls = EEVelocityController if task.endswith('_cmd_vel') else WholeBodyController
                agent = cls(task=task)
                try:
                    self.assertNotIn('ee_target', get_task(task))
                    np.testing.assert_array_equal(agent.waypoints, get_task(task)['ee_waypoints'])
                    data = mujoco.MjData(agent.model)
                    mujoco.mj_resetDataKeyframe(agent.model, data, agent.model.key('home').id)
                    pos = agent.ee_objective.positions(data.qpos)
                    agent.waypoints = np.array([pos, pos+[.02, 0, 0]])
                    agent.ee_objective.target = pos.copy()
                    obs = np.r_[data.qpos, data.qvel]
                    with patch.object(agent, 'solve_task', return_value=agent.sampling_init.copy()):
                        # High speed at the target must not count as settled.
                        obs[agent.model.nq] = .2
                        for _ in range(40):
                            agent.update(obs)
                        self.assertEqual(agent.waypoint_index, 0)
                        obs[agent.model.nq:] = 0
                        for _ in range(31):
                            agent.update(obs)
                        self.assertEqual(agent.waypoint_index, 1)
                        np.testing.assert_allclose(agent.ee_objective.target, agent.waypoints[1])
                        obs[0] += .02
                        for _ in range(31):
                            agent.update(obs)
                        self.assertTrue(agent.ee_holding)
                        obs[0] += .1
                        agent.update(obs)
                        self.assertFalse(agent.ee_holding)
                        self.assertEqual(agent.waypoint_index, 1)
                        np.testing.assert_allclose(agent.ee_objective.target, agent.waypoints[-1])
                finally:
                    agent.shutdown()
