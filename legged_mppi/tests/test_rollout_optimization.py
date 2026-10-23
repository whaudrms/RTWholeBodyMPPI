"""Parallel batch equivalence and shared kinematics on optimized cost paths."""
import unittest
from unittest.mock import patch

import mujoco
from mujoco import rollout
import numpy as np

from whole_body_mppi.control.tasks.ee_velocity import EEVelocityController


class RolloutOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.agent = EEVelocityController()
        self.addCleanup(self.agent.shutdown)
        self.data = mujoco.MjData(self.agent.model)
        mujoco.mj_resetDataKeyframe(self.agent.model, self.data, self.agent.model.key('home').id)

    def test_parallel_uneven_batches_match_serial_rollout(self):
        agent = self.agent
        rng = np.random.default_rng(123)
        for count in (1, 2, 13, 43):
            with self.subTest(count=count):
                initial = np.tile(np.r_[0., self.data.qpos, self.data.qvel], (count, 1))
                initial[:, 1] += np.arange(count)*.001
                controls = np.clip(agent.sampling_init+rng.normal(0, .02, (count, 6, 18)), agent.act_min, agent.act_max)
                expected = np.empty((count, 6, initial.shape[1]))
                actual = np.empty_like(expected)
                rollout.rollout(agent.model, self.data, skip_checks=True, nroll=count, nstep=6,
                                initial_state=initial, control=controls, state=expected)
                agent.rollout_backend.threaded_rollout(actual, controls, initial, num_workers=100, nstep=6)
                np.testing.assert_array_equal(actual, expected)

    def test_single_candidate_diagnostic_does_not_submit_worker_task(self):
        agent = self.agent
        initial = np.array([np.r_[0., self.data.qpos, self.data.qvel]])
        controls = np.broadcast_to(agent.sampling_init, (1, 6, 18)).copy()
        states = np.empty((1, 6, len(initial[0])))
        with patch.object(agent.rollout_backend.executor, 'submit', side_effect=AssertionError('No worker for one candidate')):
            agent.rollout_backend.threaded_rollout(states, controls, initial, num_workers=5, nstep=6)
        self.assertTrue(np.all(np.isfinite(states)))

    def test_velocity_task_reuses_positions_from_velocity_kinematics(self):
        agent = self.agent
        agent.joints_ref = agent.gait_scheduler.gait[:, agent.gait_scheduler.indices[:agent.horizon]]
        states = np.broadcast_to(np.r_[0., self.data.qpos, self.data.qvel], (2, agent.horizon, 50)).copy()
        actions = np.broadcast_to(agent.sampling_init, (2, agent.horizon, 18)).copy()
        with patch.object(agent.ee_kinematics, 'positions', side_effect=AssertionError('Redundant FK')), \
                patch.object(agent, 'ee_position_velocity', wraps=agent.ee_position_velocity) as kinematics:
            costs = agent.task_cost(states, actions)
        self.assertEqual(kinematics.call_count, 1)
        self.assertTrue(np.all(np.isfinite(costs)))
