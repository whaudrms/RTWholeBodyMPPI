"""Regression coverage for execution validation and immutable rollout costs."""
import unittest
from unittest.mock import Mock
import numpy as np
from mani_mppi.control.controllers.cem_whole_body_controller import CEMWholeBodyArmMPPI


class ExecutionSelectionTest(unittest.TestCase):
    def setUp(self):
        self.agent = CEMWholeBodyArmMPPI.__new__(CEMWholeBodyArmMPPI)
        self.agent.n_samples = 3
        self.agent.temperature = 1.0
        self.agent.act_min = np.array([-10.0])
        self.agent.act_max = np.array([10.0])
        self.agent.collision_valid_rollouts = np.ones(3, dtype=bool)
        self.agent.last_safe_trajectory = np.array([[3.0], [4.0]])
        self.actions = np.array([[[0.0], [0.0]], [[2.0], [2.0]], [[4.0], [4.0]]])
        self.costs = np.array([1., 2., 3.])

    def test_invalid_or_worse_mean_executes_best_sample(self):
        for result in [(0., False), (5., True), (float('nan'), False)]:
            with self.subTest(result=result):
                self.agent._evaluate_execution_candidate = Mock(return_value=result)
                selected = self.agent._select_updated_actions(self.actions, self.costs)
                np.testing.assert_array_equal(selected, self.actions[0])
                self.assertEqual(self.agent.cached_best_cost, 1.)
                self.assertEqual(self.agent.selection_source, 'best_sample')

    def test_improving_valid_mean_records_executed_cost(self):
        self.agent._evaluate_execution_candidate = Mock(return_value=(0.5, True))
        selected = self.agent._select_updated_actions(self.actions, self.costs)
        np.testing.assert_array_equal(selected, self.agent._evaluate_execution_candidate.call_args.args[0])
        self.assertEqual(self.agent.cached_best_cost, 0.5)
        self.assertEqual(self.agent.best_sample_cost, 1.)
        self.assertEqual(self.agent.selection_source, 'validated_mean')

    def test_nan_candidate_does_not_poison_mean(self):
        self.actions[2] = np.nan
        self.agent._evaluate_execution_candidate = Mock(return_value=(0.5, True))
        selected = self.agent._select_updated_actions(self.actions, self.costs)
        self.assertTrue(np.isfinite(selected).all())
        self.assertEqual(self.agent.exp_weights[2], 0.)

    def test_fallback_is_shifted_and_revalidated(self):
        self.agent.collision_valid_rollouts[:] = False
        self.agent._evaluate_execution_candidate = Mock(return_value=(7., True))
        selected = self.agent._select_updated_actions(self.actions, self.costs)
        np.testing.assert_array_equal(selected, [[4.], [4.]])
        self.assertEqual(self.agent.cached_best_cost, 7.)
        self.assertEqual(self.agent.selection_source, 'validated_fallback')

    def test_failed_fallback_does_not_replace_previous_plan(self):
        previous = self.agent.last_safe_trajectory.copy()
        self.agent.collision_valid_rollouts[:] = False
        self.agent._evaluate_execution_candidate = Mock(return_value=(0., False))
        with self.assertRaisesRegex(RuntimeError, 'fallback failed'):
            self.agent._select_updated_actions(self.actions, self.costs)
        np.testing.assert_array_equal(self.agent.last_safe_trajectory, previous)
        self.assertFalse(self.agent.execution_valid)
        self.assertEqual(self.agent.cached_best_cost, float('inf'))

    def test_validation_uses_matching_sensors_and_restores_diagnostics(self):
        agent = self.agent
        agent.obs = np.zeros(2)
        agent.joints_ref = np.zeros((2, 2))
        agent.body_ref = np.zeros(2)
        original_valid = agent.collision_valid_rollouts.copy()
        agent.last_cost_terms = {'robot': np.array([1., 2., 3.])}
        old_terms = agent.last_cost_terms
        sensors = np.array([[[42.], [43.]]])
        def rollout(obs, controls):
            agent.last_rollout_sensors = sensors
            return np.zeros((1, 2, 2))
        def cost(states, controls, joints, body, rollout_sensors=None):
            self.assertIs(rollout_sensors, sensors)
            agent.collision_valid_rollouts = np.array([False])
            agent.last_cost_terms = {'robot': np.array([9.])}
            return np.array([9.])
        agent.rollout_func = rollout
        agent.calculate_total_cost = cost
        self.assertEqual(agent._evaluate_execution_candidate(self.actions[0]), (9., False))
        np.testing.assert_array_equal(agent.collision_valid_rollouts, original_valid)
        self.assertIs(agent.last_cost_terms, old_terms)


class LocomaniRolloutTest(unittest.TestCase):
    def test_cost_is_repeatable_and_execution_cost_matches_rollout(self):
        from mani_mppi.control.controllers.mppi_locomani import MPPI
        agent = MPPI()
        try:
            obs = np.concatenate((agent.model.keyframe("stand").qpos.copy(),
                                  np.zeros(agent.model.nv)))
            obs[3:7] = [np.cos(.2), 0., 0., np.sin(.2)]
            obs[23:26] = [.1, .2, 0.]
            actions = agent.perturb_action()
            states = agent.rollout_func(obs, actions)
            original = states.copy()
            agent.obs = obs
            agent.joints_ref = agent._joint_reference()
            args = (states, actions, agent.joints_ref, agent.body_ref)
            costs = agent.calculate_total_cost(*args, rollout_sensors=agent.sensor_rollouts)
            np.testing.assert_array_equal(states, original)
            costs_again = agent.calculate_total_cost(*args, rollout_sensors=agent.sensor_rollouts)
            np.testing.assert_allclose(costs, costs_again)
            np.testing.assert_allclose(costs, sum(agent.last_cost_terms.values()))
            selected = agent._select_updated_actions(actions, costs)
            cost, valid = agent._evaluate_execution_candidate(selected)
            self.assertTrue(valid)
            self.assertAlmostEqual(cost, agent.cached_best_cost)
        finally:
            agent.close()


if __name__ == '__main__':
    unittest.main()
