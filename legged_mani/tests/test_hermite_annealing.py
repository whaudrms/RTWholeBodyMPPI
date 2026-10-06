"""Hermite geometry, PD feedforward, annealing and simulator regression tests."""
import unittest
from pathlib import Path
import numpy as np
import mujoco
from mani_mppi.control.controllers.annealed_search import bounded_hermite
from mani_mppi.control.controllers.mppi_locomani import MPPI
from mani_mppi.interface.simulator import Simulator


class HermiteGeometryTest(unittest.TestCase):
    def test_derivatives_match_and_entire_spline_stays_in_bounds(self):
        times = np.array([0., .13, .26, .39])
        q = np.array([[[.95], [-.95], [.9], [-.9]]])
        v = np.full_like(q, 100.)
        query = np.linspace(0, .39, 5001)
        positions, velocities = bounded_hermite(times, q, v, query, -1., 1.)
        self.assertLessEqual(float(positions.max()), 1. + 1e-12)
        self.assertGreaterEqual(float(positions.min()), -1. - 1e-12)
        finite_difference = np.gradient(positions, query, axis=1)
        # Hermite is C1; finite differences straddling knots sample a jump in q_ddot.
        interior = np.min(np.abs(query[:, None] - times), axis=1) > 2 * (query[1] - query[0])
        np.testing.assert_allclose(finite_difference[:, interior], velocities[:, interior], atol=3e-5)
        nodes, _ = bounded_hermite(times, q, v, times, -1., 1.)
        np.testing.assert_allclose(nodes, q)


class HermiteControllerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = MPPI(rollout_mode='hermite')
        cls.agent.adaptive_gait_enabled = False

    @classmethod
    def tearDownClass(cls):
        cls.agent.close()

    def observation(self):
        return np.concatenate((self.agent.model.keyframe('stand').qpos.copy(),
                               np.zeros(self.agent.model.nv)))

    def test_pd_velocity_feedforward_and_force_limits(self):
        a = self.agent
        data = mujoco.MjData(a.model)
        data.qpos[:] = self.observation()[:a.model.nq]
        joints = a.model.actuator_trnid[:, 0]
        qadr = a.model.jnt_qposadr[joints]
        dadr = a.model.jnt_dofadr[joints]
        data.qvel[dadr] = np.linspace(-.3, .3, a.act_dim)
        qref = data.qpos[qadr] + .03
        vref = np.linspace(-4., 4., a.act_dim)
        data.ctrl[:] = a._encode_pd_targets(qref, vref)
        mujoco.mj_forward(a.model, data)
        kp = a.model.actuator_gainprm[:, 0]
        kd = -a.model.actuator_biasprm[:, 2]
        expected = kp * (qref - data.qpos[qadr]) + kd * (vref - data.qvel[dadr])
        expected = np.clip(expected, a.model.actuator_forcerange[:, 0],
                           a.model.actuator_forcerange[:, 1])
        np.testing.assert_allclose(data.actuator_force, expected, atol=1e-9)
        data.ctrl[:] = a._encode_pd_targets(qref, vref * 1e5)
        mujoco.mj_forward(a.model, data)
        self.assertTrue(np.all(data.actuator_force <= a.model.actuator_forcerange[:, 1]))
        self.assertTrue(np.all(data.actuator_force >= a.model.actuator_forcerange[:, 0]))

    def test_noise_schedule_has_expected_standard_deviation_scaling(self):
        a = self.agent
        size = (2, a.n_knots, a.act_dim)
        a.random_generator = np.random.default_rng(99)
        a._iteration_scale = 1.
        first = a.generate_noise(size)
        a.random_generator = np.random.default_rng(99)
        a._iteration_scale = .5
        second = a.generate_noise(size)
        np.testing.assert_allclose(second, first * .5)
        raw = np.random.default_rng(99).normal(size=size) * a.noise_sigma
        expected = raw * a.horizon_noise_factor ** np.arange(a.n_knots)[::-1][None, :, None]
        np.testing.assert_allclose(first, expected)
        a._iteration_scale = 1.

    def test_headless_simulator_uses_same_pd_and_monotone_inner_search(self):
        a = self.agent
        scene = Path(__file__).resolve().parents[1] / 'mani_mppi/models/scene.xml'
        sim = Simulator(agent=a, viewer=False, T=2, dt=.01, model_path=str(scene), plot_enabled=False)
        self.assertTrue(np.array_equal(sim.model.actuator_ctrllimited, a.model.actuator_ctrllimited))
        for _ in range(2):
            obs = np.concatenate((sim.data.qpos.copy(), sim.data.qvel.copy()))
            ctrl = a.update(obs)
            np.testing.assert_allclose(ctrl, a._encode_pd_targets(
                a.selected_position_targets[0], a.selected_velocity_targets[0]))
            costs = [x['best_cost'] for x in a.iteration_diagnostics]
            self.assertEqual(len(costs), 3)
            self.assertTrue(np.all(np.diff(costs) <= 1e-7))
            cost, valid = a._evaluate_execution_candidate(a.selected_trajectory)
            self.assertTrue(valid)
            self.assertAlmostEqual(cost, a.cached_best_cost)
            sim.step(ctrl)
            self.assertTrue(np.isfinite(sim.data.qpos).all())
        self.assertEqual(a._iteration_scale, 1.)

    def test_legacy_configuration_stays_position_only(self):
        a = MPPI(rollout_mode='original_spline', anneal_iterations=1, horizon_noise_factor=1.)
        try:
            self.assertTrue(np.all(a.model.actuator_ctrllimited))
            a.adaptive_gait_enabled = False
            ctrl = a.update(self.observation())
            self.assertTrue(np.isfinite(ctrl).all())
            self.assertFalse(hasattr(a, 'selected_velocity_targets'))
        finally:
            a.close()


if __name__ == '__main__':
    unittest.main()
