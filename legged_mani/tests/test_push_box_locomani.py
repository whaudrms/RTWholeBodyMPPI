"""Regression tests for the CEM-guided push-box controller."""

from __future__ import annotations

import time
import unittest
from concurrent.futures import Future
from unittest.mock import patch

import mujoco
import numpy as np

from mani_mppi.control.controllers.mppi_push_box import (
    EXECUTE_PLAN,
    PLAN,
    MPPI,
)
from mani_mppi.control.planning import BasePosePlan, CEMResult


class PushBoxLocomaniTest(unittest.TestCase):
    """Exercise push-specific logic without launching an MPPI rollout."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.agent = MPPI(task="push_box")
        key_id = mujoco.mj_name2id(
            cls.agent.model,
            mujoco.mjtObj.mjOBJ_KEY,
            "stand",
        )
        qpos = cls.agent.model.key_qpos[key_id].copy()
        cls.stand_observation = np.concatenate(
            (qpos, np.zeros(cls.agent.model.nv, dtype=float))
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.agent.close()

    def test_named_box_state_split_is_robot_45_plus_box_13(self) -> None:
        observation = self.stand_observation.copy()
        nq = self.agent.model.nq
        observation[nq:] = np.arange(self.agent.model.nv, dtype=float)

        robot_state, box_state = self.agent._split_observation(observation)

        self.assertEqual(observation.shape, (58,))
        self.assertEqual(robot_state.shape, (45,))
        self.assertEqual(box_state.shape, (13,))
        expected_robot = np.concatenate(
            (
                observation[:nq][self.agent.robot_qpos_indices],
                observation[nq:][self.agent.robot_dof_indices],
            )
        )
        expected_box = np.concatenate(
            (
                observation[
                    self.agent.box_qpos_adr:self.agent.box_qpos_adr + 7
                ],
                observation[
                    nq + self.agent.box_dof_adr:
                    nq + self.agent.box_dof_adr + 6
                ],
            )
        )
        np.testing.assert_array_equal(robot_state, expected_robot)
        np.testing.assert_array_equal(box_state, expected_box)

    def test_box_position_cost_is_weighted_l1_xy_only(self) -> None:
        box_states = np.zeros((4, 13), dtype=float)
        box_states[:, :3] = self.agent.x_box_ref
        box_states[:, 3] = 1.0
        box_states[1, 0] += 0.1
        box_states[2, 1] -= 0.1
        box_states[3, 2] += 4.0
        box_states[3, 3:7] = [0.0, 0.0, 0.0, 1.0]
        box_states[3, 7:] = np.arange(6, dtype=float) + 1.0

        costs = self.agent._box_position_cost(box_states)

        np.testing.assert_allclose(costs, [0.0, 100.0, 150.0, 0.0])

    def test_contact_target_uses_rotated_box_surface(self) -> None:
        box_state = np.zeros((1, 13), dtype=float)
        box_state[0, :3] = [0.0, 0.0, self.agent.box_half_size[2]]
        yaw = np.pi / 4.0
        box_state[0, 3:7] = [
            np.cos(0.5 * yaw),
            0.0,
            0.0,
            np.sin(0.5 * yaw),
        ]
        old_ref = self.agent.x_box_ref.copy()
        self.agent.x_box_ref[:2] = [1.0, 1.0]
        try:
            target = self.agent._batch_contact_targets(
                box_state,
                penetration=0.0,
            )[0]
        finally:
            self.agent.x_box_ref[:] = old_ref

        direction = np.array([1.0, 1.0]) / np.sqrt(2.0)
        expected = box_state[0, :3].copy()
        expected[:2] -= self.agent.box_half_size[0] * direction
        expected[2] += self.agent.ee_contact_height_offset
        np.testing.assert_allclose(target, expected, atol=1e-10)

    def test_predicted_box_motion_translates_contact_target(self) -> None:
        box_states = np.zeros((2, 13), dtype=float)
        box_states[:, :3] = [2.0, 0.0, 0.19]
        box_states[:, 3] = 1.0
        translation = np.array([0.04, -0.03, 0.0])
        box_states[1, :3] += translation

        old_ref = self.agent.x_box_ref.copy()
        # Translate the goal with the second box so both rows have the same
        # box-to-goal direction. Only the moving reference should then differ.
        self.agent.x_box_ref[:2] = [3.0, 0.0]
        first = self.agent._batch_contact_targets(
            box_states[:1],
            penetration=0.0,
        )[0]
        self.agent.x_box_ref[:2] += translation[:2]
        second = self.agent._batch_contact_targets(
            box_states[1:],
            penetration=0.0,
        )[0]
        self.agent.x_box_ref[:] = old_ref

        np.testing.assert_allclose(second - first, translation, atol=1e-10)

    def test_box_goal_requires_configured_hold_steps(self) -> None:
        observation = self.stand_observation.copy()
        observation[
            self.agent.box_qpos_adr:self.agent.box_qpos_adr + 3
        ] = self.agent.x_box_ref
        old_count = self.agent._box_goal_hold_count
        old_success = self.agent.task_success
        try:
            self.agent.task_success = False
            self.agent._box_goal_hold_count = 0
            for _ in range(self.agent.box_goal_hold_steps - 1):
                self.assertFalse(self.agent.goal_reached(observation))
            self.assertTrue(self.agent.goal_reached(observation))
        finally:
            self.agent._box_goal_hold_count = old_count
            self.agent.task_success = old_success

    def test_contact_engagement_moves_reference_through_box_face(self) -> None:
        _, box_state = self.agent._split_observation(
            self.stand_observation
        )
        old_engaged = self.agent.contact_engaged
        try:
            self.agent.contact_engaged = False
            precontact = self.agent._batch_contact_targets(
                box_state[None, :]
            )[0]
            self.agent.contact_engaged = True
            pushing = self.agent._batch_contact_targets(
                box_state[None, :]
            )[0]
        finally:
            self.agent.contact_engaged = old_engaged

        direction = (
            self.agent.x_box_ref[:2] - box_state[:2]
        )
        direction /= np.linalg.norm(direction)
        expected_delta = (
            self.agent.precontact_gap + self.agent.push_penetration
        ) * direction
        np.testing.assert_allclose(
            pushing[:2] - precontact[:2],
            expected_delta,
            atol=1e-10,
        )

    def test_pending_refresh_keeps_active_plan_executing(self) -> None:
        observation = self.stand_observation.copy()
        pending = Future()
        saved = {
            "future": self.agent._planner_future,
            "has_plan": self.agent._has_active_plan,
            "phase": self.agent.motion_phase,
            "planned_xy": self.agent.planned_base_xy.copy(),
            "planned_height": self.agent.planned_base_height,
            "base_xy_cmd": self.agent.base_xy_cmd.copy(),
            "contact": self.agent.contact_engaged,
        }
        try:
            self.agent._planner_future = pending
            self.agent._has_active_plan = True
            self.agent.motion_phase = EXECUTE_PLAN
            self.agent.planned_base_xy = observation[:2] + [0.20, 0.0]
            self.agent.planned_base_height = float(observation[2])
            self.agent.base_xy_cmd = observation[:2].copy()
            self.agent.contact_engaged = False

            self.agent._update_motion_reference(observation)

            self.assertEqual(self.agent.motion_phase, EXECUTE_PLAN)
            self.assertGreater(self.agent.body_ref[0], observation[0])
        finally:
            pending.cancel()
            self.agent._planner_future = saved["future"]
            self.agent._has_active_plan = saved["has_plan"]
            self.agent.motion_phase = saved["phase"]
            self.agent.planned_base_xy = saved["planned_xy"]
            self.agent.planned_base_height = saved["planned_height"]
            self.agent.base_xy_cmd = saved["base_xy_cmd"]
            self.agent.contact_engaged = saved["contact"]

    def test_stale_cem_result_is_discarded_after_box_motion(self) -> None:
        observation = self.stand_observation.copy()
        self.agent._update_task_references(
            observation, allow_engagement=False
        )
        target = self.agent.ee_target_pos.copy()
        box_xy = self.agent.box_state[:2].copy()
        plan = BasePosePlan(
            base_xy=observation[:2] + [0.1, 0.0],
            base_height=float(observation[2]),
            cem_result=CEMResult(
                solution=np.array([0.1, 0.0, observation[2]]),
                cost=1.0,
                metrics={
                    "feasible": True,
                    "collision_valid": True,
                    "residual": 0.0,
                },
                evaluations=144,
                iterations=3,
            ),
        )
        completed = Future()
        completed.set_result(
            (
                self.agent._planner_request_id,
                target,
                box_xy,
                plan,
                0.01,
            )
        )
        moved_observation = observation.copy()
        moved_observation[self.agent.box_qpos_adr] += (
            self.agent.cem_target_stale_tolerance + 0.05
        )
        old_future = self.agent._planner_future
        old_force = self.agent._force_cem_replan
        try:
            self.agent._planner_future = completed
            self.agent._force_cem_replan = False
            self.assertFalse(
                self.agent._apply_completed_cem_plan(moved_observation)
            )
            self.assertTrue(self.agent._force_cem_replan)
        finally:
            self.agent._planner_future = old_future
            self.agent._force_cem_replan = old_force

    def test_submit_cem_does_not_launch_an_mppi_rollout(self) -> None:
        observation = self.stand_observation.copy()
        base_xy = observation[:2] + np.array([0.10, -0.02])
        plan = BasePosePlan(
            base_xy=base_xy,
            base_height=float(observation[2]),
            cem_result=CEMResult(
                solution=np.array([0.10, -0.02, observation[2]]),
                cost=1.0,
                metrics={
                    "feasible": True,
                    "collision_valid": True,
                    "residual": 0.0,
                },
                evaluations=144,
                iterations=3,
            ),
        )

        with (
            patch.object(
                self.agent,
                "rollout_func",
                side_effect=AssertionError(
                    "background CEM must not run an MPPI rollout"
                ),
            ),
            patch.object(
                self.agent.base_pose_planner,
                "plan",
                return_value=plan,
            ) as planner,
        ):
            self.agent._submit_cem_plan(observation)
            self.assertEqual(self.agent.motion_phase, PLAN)
            deadline = time.monotonic() + 2.0
            while (
                self.agent.motion_phase != EXECUTE_PLAN
                and time.monotonic() < deadline
            ):
                self.agent._apply_completed_cem_plan(observation)
                time.sleep(0.001)

        planner.assert_called_once()
        self.assertEqual(self.agent.motion_phase, EXECUTE_PLAN)
        np.testing.assert_allclose(self.agent.planned_base_xy, base_xy)


if __name__ == "__main__":
    unittest.main()
