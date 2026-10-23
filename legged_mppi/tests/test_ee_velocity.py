"""Velocity feedback, floating-base kinematics and cycle-boundary costs."""
import unittest
from unittest.mock import patch
import mujoco
import numpy as np
from whole_body_mppi.control.tasks.ee_velocity import EEVelocityController


class EEVelocityTests(unittest.TestCase):
    def setUp(self):
        self.agent = EEVelocityController()
        self.addCleanup(self.agent.shutdown)

    def test_world_velocity_matches_mujoco_site_jacobian(self):
        a = self.agent
        d = mujoco.MjData(a.model)
        rng = np.random.default_rng(12)
        for _ in range(10):
            mujoco.mj_resetDataKeyframe(a.model, d, a.model.key('home').id)
            q = rng.normal(size=4)
            d.qpos[3:7] = q/np.linalg.norm(q)
            d.qpos[19:25] += rng.normal(0, .3, 6)
            d.qvel[:] = rng.normal(size=a.model.nv)
            mujoco.mj_forward(a.model, d)
            jac = np.zeros((3, a.model.nv))
            mujoco.mj_jacSite(a.model, d, jac, None, a.model.site('dm_tcp').id)
            pos, vel = a.ee_position_velocity(d.qpos, d.qvel)
            np.testing.assert_allclose(pos, d.site_xpos[a.model.site('dm_tcp').id], atol=1e-10)
            np.testing.assert_allclose(vel, jac@d.qvel, atol=1e-10)

    def test_commands_are_time_based_and_do_not_depend_on_position(self):
        a = self.agent
        self.assertFalse(hasattr(a, 'ee_objective'))
        self.assertFalse(hasattr(a, 'waypoints'))
        a.set_command_time(1.)
        np.testing.assert_allclose(a.ee_command, [.05, 0, 0])
        d = mujoco.MjData(a.model)
        mujoco.mj_resetDataKeyframe(a.model, d, a.model.key('home').id)
        with patch.object(a, 'solve_task', return_value=a.sampling_init.copy()):
            obs = np.r_[d.qpos, d.qvel]
            a.update(obs)
            obs[:2] += 10.
            a.update(obs)
        np.testing.assert_allclose(a.ee_command, [.05, 0, 0])
        a.set_command_time(5.)
        np.testing.assert_allclose(a.ee_command, 0.)
        a.set_command_time(100.)
        np.testing.assert_allclose(a.ee_command, 0.)

    def test_cost_is_invariant_to_world_xy_translation(self):
        a = self.agent
        a.set_command_time(1.)
        a.joints_ref = a.gait_scheduler.gait[:, a.gait_scheduler.indices[:a.horizon]]
        home = np.r_[0, a.model.key('home').qpos, np.zeros(a.model.nv)]
        states = np.broadcast_to(home, (1, a.horizon, len(home))).copy()
        actions = np.broadcast_to(a.previous_action, (1, a.horizon, 18)).copy()
        original = a.task_cost(states, actions)
        states[..., 1:3] += [10., -20.]
        np.testing.assert_allclose(a.task_cost(states, actions), original)

    def test_action_delta_includes_previous_applied_action(self):
        a = self.agent
        a.joints_ref = a.gait_scheduler.gait[:, a.gait_scheduler.indices[:a.horizon]]
        home = np.r_[0, a.model.key('home').qpos, np.zeros(a.model.nv)]
        states = np.broadcast_to(home, (1, a.horizon, len(home))).copy()
        actions = np.broadcast_to(a.previous_action, (1, a.horizon, 18)).copy()
        baseline = a.task_cost(states, actions)
        a.previous_action[12] += .2
        changed = a.task_cost(states, actions)
        np.testing.assert_allclose(changed-baseline, a.delta_weights[12]*.2**2)
