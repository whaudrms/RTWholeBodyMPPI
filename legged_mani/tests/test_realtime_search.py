"""Soft-budget execution and conservative collision broad-phase regressions."""
from pathlib import Path
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from mani_mppi.control.collision.arm_torso_collision import ArmTorsoCollision
from mani_mppi.control.controllers.mppi_locomani import MPPI


class RealtimeSearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = MPPI(rollout_mode="hermite", performance_profile="rt20")
        cls.agent.adaptive_gait_enabled = False

    @classmethod
    def tearDownClass(cls):
        cls.agent.close()

    def observation(self):
        a = self.agent
        return np.r_[a.model.keyframe("stand").qpos, np.zeros(a.model.nv)]

    def test_profile_applies_before_buffer_and_worker_allocation(self):
        a = self.agent
        self.assertEqual(a.n_samples, 30)
        self.assertEqual(a.num_workers, 10)
        self.assertEqual(a.rollout_chunk_size, 1)
        self.assertEqual(a.state_rollouts.shape[:2], (30, 40))
        self.assertEqual(a.params["n_samples"], 30)
        self.assertEqual(a.planning_budget_ms, 20.)
        self.assertTrue(a.params["collision_broadphase"])

    def test_budget_keeps_fresh_valid_plan_and_skips_extra_iterations(self):
        a = self.agent
        # Force the one-batch path without asserting platform-dependent latency.
        with patch.object(a, "planning_budget_ms", 1e-9):
            control = a.update(self.observation())
            self.assertTrue(a.budget_limited)
            self.assertTrue(a.deadline_missed)
            self.assertEqual(len(a.iteration_diagnostics), 1)
            self.assertTrue(a.execution_valid)
            cost, valid = a._evaluate_execution_candidate(a.selected_trajectory)
            self.assertTrue(valid)
            self.assertAlmostEqual(cost, a.cached_best_cost)
            np.testing.assert_allclose(control, a.selected_trajectory[0])

    def test_optional_batch_only_selection_has_no_single_candidate_rollout(self):
        a = self.agent
        with (patch.object(a, "validate_mean", False),
              patch.object(a, "planning_budget_ms", 0.),
              patch.object(a, "_evaluate_execution_candidate",
                           side_effect=AssertionError("Unexpected separate validation")),
              patch.object(a, "rollout_func", wraps=a.rollout_func) as rollout):
            a.update(self.observation())
            self.assertEqual(rollout.call_count, a.anneal_iterations)
            for call in rollout.call_args_list:
                self.assertEqual(call.args[1].shape, (30, 40, a.act_dim))
            costs = [row["best_cost"] for row in a.iteration_diagnostics]
            self.assertTrue(np.all(np.diff(costs) <= 1e-7))
            self.assertEqual(a.selection_source, "best_sample")
        cost, valid = a._evaluate_execution_candidate(a.selected_trajectory)
        self.assertTrue(valid)
        self.assertAlmostEqual(cost, a.cached_best_cost)

    def test_profile_rejects_incompatible_sampler(self):
        with self.assertRaisesRegex(ValueError, "requires rollout_mode=hermite"):
            MPPI(performance_profile="rt20")

    def test_inline_validation_matches_threaded_rollout_and_sensors(self):
        a = self.agent
        control = a.trajectory[None].copy()
        batch = a.rollout_actions(self.observation(), np.repeat(control, 2, axis=0)).copy()
        sensors = a.last_rollout_sensors.copy()
        with patch.object(a.executor, "submit", side_effect=AssertionError("Unexpected worker wakeup")):
            single = a.rollout_actions(self.observation(), control)
        np.testing.assert_allclose(single[0], batch[0], atol=1e-12)
        np.testing.assert_allclose(a.last_rollout_sensors[0], sensors[0], atol=1e-12)


class CollisionBroadphaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = Path(__file__).resolve().parents[1] / "mani_mppi/models/scene.xml"
        cls.model = mujoco.MjModel.from_xml_path(str(model_path))

    def test_randomized_mask_matches_original_bounds_and_exact_distance(self):
        rng = np.random.default_rng(203)
        config = {"collision_hard_distance": .025, "collision_shoulder_exclusion": 0.}
        original = ArmTorsoCollision(self.model, config)
        fast = ArmTorsoCollision(self.model, {**config, "collision_broadphase": True})
        count = 6000
        states = np.zeros((count, 7))
        states[:, :3] = rng.normal(size=(count, 3))
        states[:, 3:7] = rng.normal(size=(count, 4))
        states[:, 3:7] /= np.linalg.norm(states[:, 3:7], axis=1, keepdims=True)
        local = rng.uniform(-1, 1, size=(count, 4, 3))
        # Include far pairs, crossings, near surfaces and degenerate segments.
        local[::7, 1:] = local[::7, :1]
        rotations = fast.quaternion_rotation_matrices(states[:, 3:7])
        body = local @ fast.body_local_mat.T + fast.body_local_pos
        world = np.einsum("nij,nkj->nki", rotations, body) + states[:, None, :3]
        expected = original.evaluate(states, world)
        actual = fast.evaluate(states, world)
        np.testing.assert_array_equal(actual.segment_valid, expected.segment_valid)
        exact = fast.segment_aabb_distance(local[:, :3], local[:, 1:], fast.body_half_size)
        exact -= fast.capsule_radii
        np.testing.assert_array_equal(actual.segment_valid, exact >= fast.hard_distance)
        self.assertLessEqual(actual.exact_evaluations, expected.exact_evaluations)
        self.assertTrue(actual.segment_valid.any())
        self.assertFalse(actual.segment_valid.all())

    def test_near_threshold_and_shoulder_trim_match_original(self):
        config = {"collision_hard_distance": .025, "collision_shoulder_exclusion": .07}
        slow = ArmTorsoCollision(self.model, config)
        fast = ArmTorsoCollision(self.model, {**config, "collision_broadphase": True})
        for delta in [-1e-8, 0., 1e-8, .1]:
            states = np.array([[0., 0., 0., 1., 0., 0., 0.]])
            points = np.zeros((1, 4, 3))
            points[..., 1] = fast.body_half_size[1] + fast.capsule_radii[0] + .025 + delta
            points[0, :, 0] = np.linspace(-.1, .1, 4)
            points = points @ fast.body_local_mat.T + fast.body_local_pos
            actual = fast.evaluate(states, points)
            expected = slow.evaluate(states, points)
            np.testing.assert_array_equal(actual.segment_valid, expected.segment_valid)


if __name__ == "__main__":
    unittest.main()
