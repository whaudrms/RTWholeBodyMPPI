"""EE forward kinematics, full-body sampling and short closed-loop verification."""
import unittest

import mujoco
import numpy as np

from whole_body_mppi.control.tasks.whole_body import WholeBodyController
from whole_body_mppi.control.solvers.options import apply_mppi_mode
from whole_body_mppi.control.costs.end_effector import EndEffectorObjective, quaternion_matrix


class WholeBodyEETests(unittest.TestCase):
    def test_batched_kinematics_matches_mujoco(self):
        agent = WholeBodyController(task='ee_reach_cmd_vel')
        try:
            model = agent.model
            data = mujoco.MjData(model)
            rng = np.random.default_rng(7)
            states, expected, expected_base = [], [], []
            for _ in range(20):
                mujoco.mj_resetDataKeyframe(model, data, 0)
                data.qpos[:3] += rng.normal(0, .1, 3)
                q = rng.normal(size=4)
                data.qpos[3:7] = q/np.linalg.norm(q)
                data.qpos[19:] = rng.uniform(model.jnt_range[13:, 0], model.jnt_range[13:, 1])
                mujoco.mj_forward(model, data)
                tcp = data.site_xpos[model.site('dm_tcp').id].copy()
                expected.append(tcp)
                expected_base.append(quaternion_matrix(data.qpos[3:7]).T@(tcp-data.qpos[:3]))
                states.append(data.qpos.copy())
            np.testing.assert_allclose(agent.ee_objective.positions(np.array(states)), expected, atol=1e-12)
            base = EndEffectorObjective(model, [.45, .03, .55], frame='base')
            np.testing.assert_allclose(base.positions(np.array(states)), expected_base, atol=1e-12)
        finally:
            agent.shutdown()

    def test_full_body_actions_and_noise_in_every_mode(self):
        for mode in ('mppi', 'dial-mpc', 'cma-mppi', 'mpopi', 'mbd'):
            with self.subTest(mode=mode):
                agent = WholeBodyController(task='ee_reach_cmd_vel')
                try:
                    apply_mppi_mode(agent, mode)
                    agent.set_params(5, .1, max(6, 2*agent.n_iterations))
                    data = mujoco.MjData(agent.model)
                    mujoco.mj_resetDataKeyframe(agent.model, data, 0)
                    action = agent.update(np.r_[data.qpos, data.qvel])
                    self.assertEqual(agent.act_dim, 16)
                    self.assertEqual(action.shape, (18,))
                    self.assertEqual(agent.selected_trajectory.shape, (5, 16))
                    self.assertTrue(np.isfinite(agent.eval_best_trajectory()))
                    self.assertTrue(np.all(action[:16] >= agent.act_min))
                    self.assertTrue(np.all(action[:16] <= agent.act_max))
                    agent.next_goal()
                    self.assertEqual(agent.noise_sigma.shape, (16,))
                finally:
                    agent.shutdown()

    def test_invalid_robot_or_welded_arm_is_rejected(self):
        for robot, variant in (('go1', 'full'), ('go1_dm', 'simple')):
            with self.assertRaises(ValueError):
                WholeBodyController(task='ee_reach_cmd_vel', robot=robot, rollout_model=variant)

    def test_ee_tasks_load_separate_configs(self):
        from whole_body_mppi.control.tasks.definitions import get_task
        for task, frame, target in (('ee_reach_cmd_vel', 'world', [.03,0,0]),
                                    ('ee_walk_cmd_vel', 'base', [.03,0,0])):
            agent = WholeBodyController(task=task)
            try:
                self.assertEqual(agent.task, task)
                self.assertEqual(agent.ee_objective.frame, frame)
                np.testing.assert_allclose(agent.ee_objective.target, get_task(task)['ee_cmd_vel'])
                self.assertEqual(get_task(task)['initial_key'], 'home')
                self.assertEqual(get_task(task)['controller'], 'whole_body')
                self.assertTrue(get_task(task)['config_path'].endswith(f'mppi_{task}.yml'))
                np.testing.assert_allclose(agent.arm_noise, agent.params['whole_body']['arm_noise_sigma'])
                self.assertEqual(agent.Q.shape, (49, 49))
                self.assertEqual(agent.R.shape, (18, 18))
                np.testing.assert_allclose(agent.arm_joint_ref, get_task(task)['arm_joint_ref'])
            finally:
                agent.shutdown()
        with self.assertRaises(ValueError):
            WholeBodyController(task='stand')

    def test_cli_selects_task_and_rejects_removed_ee_flags(self):
        import contextlib
        import io
        from pathlib import Path
        import runpy
        import sys
        from unittest.mock import patch
        parse = runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/simulate_mppi.py'))['parse_args']
        argv = ['simulate_mppi.py', '--robot', 'go1_dm', '--task', 'ee_walk_cmd_vel']
        with patch.object(sys, 'argv', argv):
            args = parse()
            self.assertEqual(args.task, 'ee_walk_cmd_vel')
            self.assertFalse(hasattr(args, 'ee_target'))
        for extra in (['--ee-target', '.4', '0', '.8'], ['--ee-frame', 'base'],
                      ['--ee-weight', '1000'], ['--ee-terminal-weight', '5'],
                      ['--rollout-model', 'simple']):
            with patch.object(sys, 'argv', argv+extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    parse()
                self.assertEqual(caught.exception.code, 2)

    def test_velocity_schedule_and_cost(self):
        agent = WholeBodyController(task='ee_reach_cmd_vel')
        try:
            obj = agent.ee_objective
            np.testing.assert_allclose(obj.command_at([0., obj.duration-.01, obj.duration, obj.duration+1.]),
                                       [obj.target, obj.target, np.zeros(3), np.zeros(3)])
            qpos = np.tile(agent.model.key('home').qpos, (1, 3, 1))
            qvel = np.zeros((1, 3, agent.model.nv))
            qvel[..., 0] = .03  # Pure world translation exactly tracks command.
            self.assertAlmostEqual(obj.cost(qpos, qvel, np.array([.03,0,0]))[0], 0.)
            self.assertGreater(obj.cost(qpos, qvel, np.zeros(3))[0], 0.)
            agent.ee_command_time = obj.duration
            agent.solve_task = lambda obs: np.zeros(16)
            agent.update(np.r_[qpos[0,0], qvel[0,0]])
            np.testing.assert_array_equal(agent.ee_reference, np.zeros((agent.horizon,3)))
        finally:
            agent.shutdown()

    def test_velocity_matches_mujoco_jacobian_in_both_frames(self):
        from whole_body_mppi.control.costs.end_effector import EndEffectorVelocityObjective
        agent = WholeBodyController(task='ee_reach_cmd_vel')
        try:
            model = agent.model
            data = mujoco.MjData(model)
            rng = np.random.default_rng(17)
            for frame in ('world', 'base'):
                obj = EndEffectorVelocityObjective(model, [.03,0,0], 2., frame)
                for _ in range(10):
                    mujoco.mj_resetDataKeyframe(model, data, 0)
                    q = rng.normal(size=4); data.qpos[3:7] = q/np.linalg.norm(q)
                    data.qpos[19:] += rng.normal(0, .1, 6)
                    data.qvel[:] = rng.normal(0, .2, model.nv)
                    mujoco.mj_forward(model, data)
                    jp = np.zeros((3, model.nv)); jr = np.zeros_like(jp)
                    site = model.site('dm_tcp').id
                    mujoco.mj_jacSite(model, data, jp, jr, site)
                    expected = jp @ data.qvel
                    if frame == 'base':
                        R = quaternion_matrix(data.qpos[3:7])
                        expected = R.T @ (expected-data.qvel[:3])-np.cross(
                            data.qvel[3:6], R.T @ (data.site_xpos[site]-data.qpos[:3]))
                    np.testing.assert_allclose(obj.velocities(data.qpos,data.qvel), expected, atol=1e-10)
                    # With a fixed arm, base-relative EE velocity must be zero
                    # regardless of the trunk's translation and rotation.
                    data.qvel[6:] = 0
                    if frame == 'base':
                        np.testing.assert_allclose(obj.velocities(data.qpos,data.qvel), 0, atol=1e-12)
        finally:
            agent.shutdown()


if __name__ == '__main__':
    unittest.main()
