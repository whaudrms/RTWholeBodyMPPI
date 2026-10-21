"""Mounted-arm locomotion integration, including substep dynamics and terrain."""
import unittest

import mujoco
import numpy as np

from whole_body_mppi.control.tasks.base import PACKAGE_ROOT
from whole_body_mppi.control.tasks.definitions import get_task
from whole_body_mppi.control.tasks.locomotion import LocomotionController
from whole_body_mppi.control.solvers.options import apply_mppi_mode
from whole_body_mppi.models.go1_dm.build_model import load_model


class MountedLocomotionTests(unittest.TestCase):
    def test_substep_rollout_matches_simulation(self):
        agent = LocomotionController('walk_straight', robot='go1_dm', rollout_model='full')
        try:
            model = agent.model
            data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
            initial = np.r_[0., data.qpos, data.qvel][None]
            actions = np.broadcast_to(data.ctrl[:12], (1, 3, 12)).copy()
            states = np.empty((1, 3, initial.shape[1]))
            agent.rollout_func(states, actions, initial, num_workers=1, nstep=3)
            for step in range(3):
                data.ctrl[:] = agent.rollout_func.expand_actions(actions[0, step])
                for _ in range(agent.rollout_func.substeps):
                    mujoco.mj_step(model, data)
                np.testing.assert_allclose(states[0, step], np.r_[data.time, data.qpos, data.qvel], atol=1e-10)
            self.assertAlmostEqual(data.time, .03)
        finally:
            agent.shutdown()

    def test_all_optimizer_modes_hold_arm_and_produce_finite_cost(self):
        for mode in ('mppi', 'dial-mpc', 'cma-mppi', 'mpopi', 'mbd'):
            with self.subTest(mode=mode):
                agent = LocomotionController('walk_straight', robot='go1_dm')
                try:
                    apply_mppi_mode(agent, mode)
                    agent.set_params(5, 0.1, 6)
                    data = mujoco.MjData(agent.model)
                    mujoco.mj_resetDataKeyframe(agent.model, data, 0)
                    action = agent.update(np.r_[data.qpos, data.qvel])
                    self.assertEqual(action.shape, (18,))
                    self.assertEqual(agent.selected_trajectory.shape, (5, 12))
                    np.testing.assert_array_equal(action[12:], agent.model.key('home').ctrl[12:])
                    self.assertTrue(np.isfinite(agent.eval_best_trajectory()))
                finally:
                    agent.shutdown()

    def test_reduced_prediction_preserves_mass_com_and_arm_dynamics(self):
        from whole_body_mppi.models.go1_dm.rollout_model import load_rollout_model
        path = PACKAGE_ROOT / get_task('walk_straight')['model_path']
        full, reduced = load_model(path), load_rollout_model(path)
        self.assertEqual((reduced.nq, reduced.nv, reduced.nu), (25, 24, 18))
        self.assertEqual(reduced.neq, 0)
        self.assertEqual(reduced.nmesh, 5)
        self.assertAlmostEqual(full.body_mass.sum(), reduced.body_mass.sum())
        data = []
        for model in (full, reduced):
            d = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, d, 0)
            mujoco.mj_forward(model, d)
            self.assertFalse(any(c.dist < -1e-5 for c in d.contact))
            data.append(d)
        np.testing.assert_allclose(data[0].subtree_com[full.body('trunk').id],
                                   data[1].subtree_com[reduced.body('trunk').id], atol=1e-12)
        for i in range(1, 7):
            np.testing.assert_allclose(full.body(f'dm_link{i}').inertia, reduced.body(f'dm_link{i}').inertia)
            np.testing.assert_allclose(full.joint(f'dm_joint{i}').range, reduced.joint(f'dm_joint{i}').range)

    def test_same_model_single_step_rollout_and_simulation(self):
        from whole_body_mppi.interface.simulator import Simulator
        agent = LocomotionController('walk_straight', robot='go1_dm')
        try:
            simulator = Simulator(agent=agent, robot='go1_dm', viewer=False, T=4,
                                  model_path=PACKAGE_ROOT / get_task('walk_straight')['sim_path'])
            model = agent.model
            self.assertIs(model, agent.rollout_backend.model)
            self.assertEqual(agent.rollout_func.substeps, 1)
            self.assertTrue(agent.rollout_func.identity_state)
            self.assertEqual(model.nq, simulator.model.nq)
            self.assertEqual(model.nu, simulator.model.nu)
            self.assertEqual(model.nmesh, simulator.model.nmesh)
            self.assertAlmostEqual(model.opt.timestep, 0.01)
            self.assertAlmostEqual(simulator.model.opt.timestep, 0.01)
            self.assertEqual(simulator.update_ratio, 1)
            bare = mujoco.MjModel.from_xml_path(str(PACKAGE_ROOT / get_task('walk_straight')['model_path']))
            self.assertEqual(model.opt.integrator, bare.opt.integrator)
            np.testing.assert_array_equal(simulator.data.qpos[:19], bare.key('sit').qpos)
            np.testing.assert_array_equal(agent.sampling_init, [-0.3, 1.34, -2.83, 0.3, 1.34, -2.83]*2)
            data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, data, 0)
            obs = np.r_[0., data.qpos, data.qvel][None]
            actions = np.tile(data.ctrl[:12], (1, 3, 1))
            states = np.empty((1, 3, obs.shape[1]))
            agent.rollout_func(states, actions, obs, num_workers=1)
            for step in range(3):
                mujoco.mj_step(model, data)
                np.testing.assert_allclose(states[0, step], np.r_[data.time, data.qpos, data.qvel], atol=1e-10)
            self.assertEqual(agent.act_dim, 12)
            self.assertEqual(agent.n_samples, agent.params['n_samples'])
            self.assertEqual(agent.horizon, agent.params['horizon'])
            # Every primitive arm collision has a visible non-contact counterpart.
            for bid in range(model.nbody):
                if model.body(bid).name.startswith('dm_'):
                    ids = np.flatnonzero(model.geom_bodyid == bid)
                    self.assertTrue(any(model.geom_group[g] == 2 for g in ids))
                    self.assertTrue(all(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH for g in ids))
                    self.assertTrue(all(model.geom_condim[g] == 1 for g in ids))
        finally:
            agent.shutdown()

    def test_simple_payload_preserves_home_spatial_inertia_and_geometry(self):
        from whole_body_mppi.models.go1_dm.rollout_model import load_runtime_model
        path = PACKAGE_ROOT / get_task('walk_straight')['model_path']
        full = load_runtime_model(path, 'full')
        simple = load_runtime_model(path, 'simple')
        self.assertEqual((full.nq, full.nv, full.nu), (25, 24, 18))
        self.assertEqual((simple.nq, simple.nv, simple.nu), (19, 18, 12))
        self.assertEqual(simple.nbody, 15)
        self.assertAlmostEqual(full.body_mass.sum(), simple.body_mass.sum())
        states, inertias, geom_positions = [], [], []
        for model in (full, simple):
            data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, data, 0)
            mujoco.mj_forward(model, data)
            states.append(data)
            matrix = np.empty((model.nv, model.nv))
            mujoco.mj_fullM(model, matrix, data.qM)
            inertias.append(matrix[:6, :6])
            ids = [g for g in range(model.ngeom)
                   if model.body(model.geom_bodyid[g]).name.startswith('dm_')]
            geom_positions.append(data.geom_xpos[ids])
        np.testing.assert_allclose(inertias[0], inertias[1], atol=1e-12)
        np.testing.assert_allclose(states[0].subtree_com[0], states[1].subtree_com[0], atol=1e-12)
        np.testing.assert_allclose(geom_positions[0], geom_positions[1], atol=1e-12)
        np.testing.assert_allclose(states[0].site_xmat[full.site('dm_tcp').id],
                                   states[1].site_xmat[simple.site('dm_tcp').id], atol=1e-12)
        self.assertFalse(any(simple.joint(j).name.startswith('dm_') for j in range(simple.njnt)))

    def test_detailed_display_tracks_both_physics_variants_without_stepping(self):
        from whole_body_mppi.interface.simulator import Simulator
        path = PACKAGE_ROOT / get_task('walk_straight')['sim_path']
        for variant in ('simple', 'full'):
            with self.subTest(variant=variant):
                simulator = Simulator(robot='go1_dm', model_variant=variant,
                                      viewer=False, T=2, model_path=path, initial_key='home')
                self.assertIsNone(simulator._detailed_visual)
                simulator.initialize_rendering()
                self.assertEqual(simulator.render_model.nmesh, 53)
                self.assertEqual(simulator.model.nmesh, 5)
                self.assertIsNot(simulator.render_model, simulator.model)
                simulator.data.qpos[:3] += [0.1, 0.02, 0.05]
                if variant == 'full':
                    simulator.data.qpos[simulator.model.joint('dm_joint1').qposadr[0]] = 0.2
                simulator.data.time = 0.5
                mujoco.mj_forward(simulator.model, simulator.data)
                before = simulator.data.qpos.copy()
                simulator.sync_render_state()
                np.testing.assert_array_equal(simulator.data.qpos, before)
                self.assertAlmostEqual(simulator.render_data.time, 0.5)
                for name in ('trunk', 'FR_calf'):
                    np.testing.assert_allclose(
                        simulator.data.xpos[simulator.model.body(name).id],
                        simulator.render_data.xpos[simulator.render_model.body(name).id], atol=1e-12)
                source = simulator.model.site('dm_tcp').id
                target = simulator.render_model.site('dm_tcp').id
                np.testing.assert_allclose(simulator.data.site_xpos[source],
                                           simulator.render_data.site_xpos[target], atol=1e-12)
                np.testing.assert_allclose(simulator.data.site_xmat[source],
                                           simulator.render_data.site_xmat[target], atol=1e-12)

    def test_task_terrain_is_preserved(self):
        for task in ('stand', 'walk_straight', 'walk_octagon', 'stairs', 'big_box', 'climb_box_hw'):
            with self.subTest(task=task):
                path = PACKAGE_ROOT / get_task(task)['sim_path']
                original = mujoco.MjModel.from_xml_path(str(path))
                mounted = load_model(path)
                world_original = np.flatnonzero(original.geom_bodyid == 0)
                world_mounted = np.flatnonzero(mounted.geom_bodyid == 0)
                np.testing.assert_array_equal(original.geom_type[world_original], mounted.geom_type[world_mounted])
                np.testing.assert_allclose(original.geom_pos[world_original], mounted.geom_pos[world_mounted])
                np.testing.assert_allclose(original.geom_size[world_original], mounted.geom_size[world_mounted])
                self.assertEqual(mounted.nu, 19)


if __name__ == '__main__':
    unittest.main()
