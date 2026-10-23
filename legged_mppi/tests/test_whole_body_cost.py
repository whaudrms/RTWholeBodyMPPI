"""Verify arm cost indices and compatibility with the locomotion objective."""
import unittest
import numpy as np
from whole_body_mppi.control.tasks.whole_body import WholeBodyController
from whole_body_mppi.control.costs.locomotion import LocomotionCost


class WholeBodyCostTests(unittest.TestCase):
    def setUp(self):
        self.cost = WholeBodyController.__new__(WholeBodyController)
        self.cost.Q = np.zeros((49, 49))
        self.cost.R = np.zeros((18, 18))
        self.cost.arm_joint_ref = np.arange(6) * .1
        self.cost.arm_velocity_ref = np.arange(6) * .01
        self.body = np.zeros(13)
        self.body[3] = 1
        self.gait = np.zeros((24, 2))
        self.states = np.zeros((1, 2, 49))
        self.states[..., 3] = 1
        self.states[..., 19:25] = self.cost.arm_joint_ref
        self.states[..., 43:49] = self.cost.arm_velocity_ref
        self.actions = np.zeros((1, 2, 18))

    def evaluate(self):
        return self.cost.calculate_total_cost(self.states, self.actions, self.gait, self.body)[0]

    def test_each_arm_position_and_velocity_weight_uses_its_reference(self):
        for index in list(range(19, 25)) + list(range(43, 49)):
            with self.subTest(index=index):
                self.cost.Q[:] = 0
                self.cost.Q[index, index] = 3
                self.assertAlmostEqual(self.evaluate(), 0)
                self.states[0, 1, index] += .2
                self.assertAlmostEqual(self.evaluate(), 3 * .2**2)
                self.states[0, 1, index] -= .2

    def test_each_arm_torque_weight(self):
        self.actions[..., 12:] = self.states[..., 19:25] + 3/55*self.states[..., 43:49]
        for joint in range(6):
            with self.subTest(joint=joint):
                self.cost.R[:] = 0
                self.cost.R[12+joint, 12+joint] = .004
                self.assertAlmostEqual(self.evaluate(), 0)
                self.actions[0, 1, 12+joint] += .2
                self.assertAlmostEqual(self.evaluate(), .004*(55*.2)**2)
                self.actions[0, 1, 12+joint] -= .2

    def test_leg_cost_matches_existing_objective_and_input_is_unchanged(self):
        rng = np.random.default_rng(42)
        self.states += rng.normal(0, .1, self.states.shape)
        self.states[..., 3:7] /= np.linalg.norm(self.states[..., 3:7], axis=-1, keepdims=True)
        self.actions[:] = rng.normal(0, .1, self.actions.shape)
        indices = np.r_[0:19, 25:43]
        old = LocomotionCost()
        old.Q = np.diag(rng.uniform(.1, 3, 37))
        old.R = np.diag(rng.uniform(.1, 3, 12))
        self.cost.Q[np.ix_(indices, indices)] = old.Q
        self.cost.R[:12, :12] = old.R
        before = self.states.copy()
        expected = old.calculate_total_cost(self.states[..., indices], self.actions[..., :12], self.gait, self.body)
        np.testing.assert_allclose(self.evaluate(), expected[0])
        np.testing.assert_array_equal(self.states, before)

    def test_explicit_off_diagonal_costs_keep_dense_quadratic_semantics(self):
        self.states[0, 1, 19:21] += [.2, .3]
        self.cost.Q[19, 19] = 2
        self.cost.Q[20, 20] = 3
        self.cost.Q[19, 20] = self.cost.Q[20, 19] = .4
        self.cost.R[12, 12] = .01
        self.cost.R[13, 13] = .02
        self.cost.R[12, 13] = self.cost.R[13, 12] = .003
        errors = np.zeros((2, 49))
        errors[1, 19:21] = [.2, .3]
        torques = np.zeros((2, 18))
        torques[:, 12:] = 55*(self.actions[0, :, 12:]-self.states[0, :, 19:25])-3*self.states[0, :, 43:49]
        expected = sum(e@self.cost.Q@e + u@self.cost.R@u for e, u in zip(errors, torques))
        self.assertAlmostEqual(self.evaluate(), expected)
