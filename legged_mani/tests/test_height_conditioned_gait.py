"""Regression tests for locomani's height-conditioned gait selection."""

from __future__ import annotations

import time
import unittest
from concurrent.futures import Future
from threading import Event
from unittest.mock import patch

import mujoco
import numpy as np

from mani_mppi.control.controllers.mppi_locomani import (
    EXECUTE_PLAN,
    PLANNING,
    MPPI,
)
from mani_mppi.control.controllers.whole_body_arm_controller import (
    HEIGHT_GAIT_PATHS,
)
from mani_mppi.control.gait_scheduler.height_conditioned_scheduler import (
    HeightConditionedGaitScheduler,
)
from mani_mppi.control.planning import (
    BasePosePlan,
    CEMResult,
    CrossEntropyOptimizer,
)


class CrossEntropyOptimizerTest(unittest.TestCase):
    def test_evaluates_each_sample_once_without_final_recheck(self) -> None:
        optimizer = CrossEntropyOptimizer(
            lower=np.array([-2.0, -2.0]),
            upper=np.array([2.0, 2.0]),
            num_samples=32,
            num_iterations=4,
            elite_fraction=0.20,
            smoothing=0.10,
            min_std=np.array([0.01, 0.01]),
            seed=7,
        )
        target = np.array([0.6, -0.4])
        calls = 0
        evaluated_samples = 0

        def evaluate(samples):
            nonlocal calls, evaluated_samples
            calls += 1
            evaluated_samples += len(samples)
            costs = np.sum((samples - target) ** 2, axis=1)
            return costs, {"feasible": np.ones(len(samples), dtype=bool)}

        result = optimizer.optimize(
            mean=np.zeros(2),
            std=np.ones(2),
            evaluate=evaluate,
        )

        self.assertEqual(calls, optimizer.num_iterations)
        self.assertEqual(
            evaluated_samples,
            optimizer.num_samples * optimizer.num_iterations,
        )
        self.assertEqual(result.evaluations, evaluated_samples)
        np.testing.assert_allclose(result.solution, target, atol=0.15)


class HeightConditionedGaitSchedulerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scheduler = HeightConditionedGaitScheduler(
            HEIGHT_GAIT_PATHS["stance_hold"],
            name="stance_hold",
        )

    def test_interpolates_positions_and_morph_velocities(self) -> None:
        low_height = 0.35
        high_height = 0.45
        height = 0.40
        height_rate = -0.10
        indices = np.array([0, 1, 99])

        low = np.loadtxt(
            HEIGHT_GAIT_PATHS["stance_hold"][low_height],
            delimiter="\t",
            comments="#",
        )[:, indices]
        high = np.loadtxt(
            HEIGHT_GAIT_PATHS["stance_hold"][high_height],
            delimiter="\t",
            comments="#",
        )[:, indices]
        actual = self.scheduler.get_reference(
            height,
            height_rate=height_rate,
            indices=indices,
        )

        expected = 0.5 * (low + high)
        expected[16:] += (
            height_rate / (high_height - low_height)
        ) * (high[:16] - low[:16])
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_clamps_height_and_rolls_phase(self) -> None:
        below = self.scheduler.get_reference(0.1, horizon=4)
        minimum = np.loadtxt(
            HEIGHT_GAIT_PATHS["stance_hold"][0.35],
            delimiter="\t",
            comments="#",
        )[:, :4]
        np.testing.assert_allclose(below, minimum)

        self.scheduler.roll()
        rolled = self.scheduler.get_reference(0.35, horizon=4)
        expected_indices = np.array([1, 2, 3, 4])
        full = np.loadtxt(
            HEIGHT_GAIT_PATHS["stance_hold"][0.35],
            delimiter="\t",
            comments="#",
        )
        np.testing.assert_allclose(rolled, full[:, expected_indices])


class LocomaniAdaptiveMotionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent = MPPI(task="locomani")
        key_id = mujoco.mj_name2id(
            cls.agent.model,
            mujoco.mjtObj.mjOBJ_KEY,
            "stand",
        )
        qpos = cls.agent.model.key_qpos[key_id].copy()
        cls.stand_observation = np.concatenate(
            (qpos, np.zeros(cls.agent.model.nv))
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.agent.close()

    def setUp(self) -> None:
        if self.agent._planner_future is not None:
            self.agent._planner_future.result(timeout=5.0)
            self.agent._planner_future = None
        self.agent._planner_request_id += 1
        self.agent._planner_future_request_id = None
        self.agent._planner_future_goal_index = None
        self.agent.goal_index = 2
        self.agent._planned_goal_index = -1
        self.agent.base_height_cmd = self.agent.stand_base_height
        self.agent.base_height_rate_cmd = 0.0
        self.agent.base_xy_cmd = self.stand_observation[:2].copy()
        self.agent.base_xy_rate_cmd[:] = 0.0
        self.agent.motion_phase = EXECUTE_PLAN
        self.agent.plan_settled = False
        self.agent.planar_gait_active = False
        self.agent.plan_requires_replan = False
        self.agent.fallback_replan_count = 0
        self.agent._fallback_limit_reported = False
        self.agent.default_gait = self.agent.tracking_gait
        self.agent.gait_scheduler = self.agent.height_gaits[
            self.agent.tracking_gait
        ]
        self.agent.body_ref[:7] = self.stand_observation[:7]
        self.agent.body_ref[7:13] = 0.0

    @staticmethod
    def _fake_plan(
        base_xy,
        base_height,
        *,
        feasible=True,
        collision_valid=True,
    ) -> BasePosePlan:
        return BasePosePlan(
            base_xy=np.asarray(base_xy, dtype=float),
            base_height=float(base_height),
            cem_result=CEMResult(
                solution=np.array([0.0, 0.0, base_height]),
                cost=0.0,
                metrics={
                    "feasible": feasible,
                    "collision_valid": collision_valid,
                    "residual": 0.0 if feasible else 0.04,
                },
                evaluations=0,
                iterations=0,
            ),
        )

    def test_background_submission_does_not_wait(self) -> None:
        observation = self.stand_observation.copy()
        started = Event()
        release = Event()
        fake_plan = self._fake_plan(
            observation[:2],
            self.agent.stand_base_height,
        )

        def delayed_plan(_observation, _target):
            started.set()
            if not release.wait(timeout=2.0):
                raise TimeoutError("test did not release background planner")
            return fake_plan

        try:
            with patch.object(
                self.agent.base_pose_planner,
                "plan",
                side_effect=delayed_plan,
            ):
                start = time.perf_counter()
                self.agent._update_motion_reference(observation)
                elapsed = time.perf_counter() - start

                self.assertTrue(started.wait(timeout=0.5))
                self.assertLess(elapsed, 0.10)
                self.assertEqual(self.agent.motion_phase, PLANNING)
                self.assertFalse(self.agent._planner_future.done())
                release.set()
                self.agent._planner_future.result(timeout=2.0)
                self.agent._update_motion_reference(observation)
        finally:
            release.set()

        self.assertEqual(self.agent.motion_phase, EXECUTE_PLAN)
        self.assertTrue(self.agent.plan_settled)

    def test_floor_goal_selects_crouch_and_sequences_motion(self) -> None:
        observation = self.stand_observation.copy()
        with (
            patch.object(
                self.agent,
                "rollout_func",
                side_effect=AssertionError("CEM must not run MPPI rollout"),
            ),
            patch.object(
                self.agent,
                "_solve_arm_ik",
                wraps=self.agent._solve_arm_ik,
            ) as solve_ik,
            patch.object(
                self.agent,
                "_arm_torso_clearance",
                wraps=self.agent._arm_torso_clearance,
            ) as collision,
            patch.object(
                self.agent.base_pose_planner.arm_kinematics,
                "solve_ik",
                wraps=self.agent.base_pose_planner.arm_kinematics.solve_ik,
            ) as planner_solve_ik,
            patch.object(
                self.agent.base_pose_planner.arm_collision,
                "evaluate",
                wraps=self.agent.base_pose_planner.arm_collision.evaluate,
            ) as planner_collision,
        ):
            self.agent._update_motion_reference(observation)
            self.assertEqual(self.agent.motion_phase, PLANNING)
            self.agent._planner_future.result(timeout=5.0)
            self.agent._update_motion_reference(observation)

        self.assertIsNotNone(self.agent.last_cem_result)
        self.assertTrue(self.agent.last_cem_result.metrics["feasible"])
        self.assertEqual(solve_ik.call_count, 0)
        self.assertEqual(collision.call_count, 0)
        self.assertEqual(
            planner_solve_ik.call_count,
            self.agent.last_cem_result.evaluations,
        )
        self.assertEqual(
            planner_collision.call_count,
            self.agent.base_pose_planner.optimizer.num_iterations,
        )

        self.assertEqual(self.agent.motion_phase, EXECUTE_PLAN)
        self.assertLess(
            self.agent.planned_base_height,
            self.agent.stand_base_height,
        )
        self.assertGreater(
            np.linalg.norm(
                self.agent.base_xy_cmd - observation[:2]
            ),
            0.0,
        )
        self.assertLess(
            self.agent.base_height_cmd,
            self.agent.stand_base_height,
        )
        self.assertEqual(
            self.agent.default_gait,
            self.agent.approach_gait,
        )

        for _ in range(500):
            observation[:2] = self.agent.base_xy_cmd
            observation[2] = self.agent.base_height_cmd
            self.agent._update_motion_reference(observation)
            if self.agent.plan_settled:
                break

        self.assertTrue(self.agent.plan_settled)
        self.assertEqual(self.agent.motion_phase, EXECUTE_PLAN)
        self.assertEqual(
            self.agent.default_gait,
            self.agent.tracking_gait,
        )
        self.assertAlmostEqual(
            self.agent.body_ref[2],
            self.agent.planned_base_height,
        )
        np.testing.assert_allclose(
            self.agent.body_ref[3:7],
            [1.0, 0.0, 0.0, 0.0],
        )

    def test_raises_and_moves_continuously_without_recover(self) -> None:
        observation = self.stand_observation.copy()
        observation[2] = 0.40
        self.agent.base_height_cmd = 0.40
        self.agent.base_xy_cmd = observation[:2].copy()
        next_base_xy = observation[:2] + np.array([0.30, 0.0])
        self.agent._apply_base_pose_plan(
            observation,
            self._fake_plan(next_base_xy, self.agent.stand_base_height),
        )

        initial_xy_cmd = self.agent.base_xy_cmd.copy()
        self.agent._update_motion_reference(observation)
        self.assertEqual(self.agent.motion_phase, EXECUTE_PLAN)
        self.assertGreater(
            self.agent.base_xy_cmd[0],
            initial_xy_cmd[0],
        )
        self.assertGreater(self.agent.base_height_cmd, 0.40)

        for _ in range(500):
            observation[:2] = self.agent.base_xy_cmd
            observation[2] = self.agent.base_height_cmd
            self.agent._update_motion_reference(observation)
            if self.agent.plan_settled:
                break

        self.assertTrue(self.agent.plan_settled)
        self.assertAlmostEqual(
            self.agent.base_height_cmd,
            self.agent.stand_base_height,
        )

    def test_goal_completion_uses_actual_ee_without_settled_gate(self) -> None:
        observation = self.stand_observation.copy()
        self.agent._planned_goal_index = self.agent.goal_index
        self.agent.motion_phase = EXECUTE_PLAN
        target_quat = self.agent.ee_goal_quat[self.agent.goal_index]
        with patch.object(
            self.agent,
            "_ee_pose",
            return_value=(
                self.agent.ee_goal_pos[self.agent.goal_index].copy(),
                target_quat.copy(),
            ),
        ):
            self.agent.plan_settled = False
            self.assertTrue(self.agent.goal_reached(observation))

    def test_settled_fallback_replans_same_goal(self) -> None:
        observation = self.stand_observation.copy()
        fallback_xy = observation[:2] + np.array([0.30, 0.0])
        self.agent._apply_base_pose_plan(
            observation,
            self._fake_plan(
                fallback_xy,
                self.agent.stand_base_height,
                feasible=False,
            ),
        )
        self.agent.base_xy_cmd = fallback_xy.copy()
        observation[:2] = fallback_xy
        self.agent.plan_settled = True

        with (
            patch.object(
                self.agent,
                "_ee_goal_satisfied",
                return_value=False,
            ),
            patch.object(
                self.agent,
                "_start_background_plan",
            ) as start_plan,
        ):
            self.agent._update_motion_reference(observation)

        self.assertEqual(self.agent.fallback_replan_count, 1)
        self.assertEqual(self.agent._planned_goal_index, -1)
        start_plan.assert_called_once_with(observation)

    def test_collision_fallback_is_not_executed(self) -> None:
        observation = self.stand_observation.copy()
        unsafe_xy = observation[:2] + np.array([0.30, 0.0])
        self.agent._apply_base_pose_plan(
            observation,
            self._fake_plan(
                unsafe_xy,
                self.agent.min_base_height,
                feasible=False,
                collision_valid=False,
            ),
        )

        np.testing.assert_allclose(
            self.agent.planned_base_xy,
            observation[:2],
        )
        self.assertAlmostEqual(
            self.agent.planned_base_height,
            self.agent.base_height_cmd,
        )

    def test_discards_stale_background_result(self) -> None:
        observation = self.stand_observation.copy()
        stale_future = Future()
        stale_future.set_result(
            (
                self.agent._planner_request_id - 1,
                self.agent.goal_index - 1,
                self._fake_plan(
                    observation[:2],
                    self.agent.stand_base_height,
                ),
                0.01,
            )
        )
        self.agent._planner_future = stale_future

        with patch.object(
            self.agent,
            "_start_background_plan",
        ) as restart:
            ready = self.agent._poll_background_plan(observation)

        self.assertFalse(ready)
        restart.assert_called_once_with(observation)
        self.assertEqual(self.agent._planned_goal_index, -1)


if __name__ == "__main__":
    unittest.main()
