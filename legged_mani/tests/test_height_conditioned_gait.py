"""Regression tests for locomani's height-conditioned gait selection."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from mani_mppi.control.controllers.mppi_locomani import (
    APPROACH,
    LOWER,
    RECOVER,
    TRACK,
    MPPI,
)
from mani_mppi.control.controllers.whole_body_arm_controller import (
    HEIGHT_GAIT_PATHS,
)
from mani_mppi.control.gait_scheduler.height_conditioned_scheduler import (
    HeightConditionedGaitScheduler,
)


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
        self.agent.goal_index = 2
        self.agent._planned_goal_index = -1
        self.agent.base_height_cmd = self.agent.stand_base_height
        self.agent.base_height_rate_cmd = 0.0
        self.agent.motion_phase = TRACK
        self.agent.default_gait = self.agent.tracking_gait
        self.agent.gait_scheduler = self.agent.height_gaits[
            self.agent.tracking_gait
        ]
        self.agent.body_ref[:7] = self.stand_observation[:7]
        self.agent.body_ref[7:13] = 0.0

    def test_floor_goal_selects_crouch_and_sequences_motion(self) -> None:
        observation = self.stand_observation.copy()
        self.agent._activate_plan(observation)

        self.assertEqual(self.agent.motion_phase, APPROACH)
        self.assertLess(
            self.agent.planned_base_height,
            self.agent.stand_base_height,
        )

        observation[:2] = self.agent.planned_base_xy
        self.agent._update_motion_reference(observation)
        self.assertEqual(self.agent.motion_phase, LOWER)

        max_steps = int(
            np.ceil(
                (
                    self.agent.stand_base_height
                    - self.agent.planned_base_height
                )
                / (
                    self.agent.base_height_rate
                    * self.agent.model.opt.timestep
                )
            )
        ) + 2
        for _ in range(max_steps):
            self.agent._update_motion_reference(observation)
            observation[2] = self.agent.base_height_cmd

        self.agent._update_motion_reference(observation)
        self.assertEqual(self.agent.motion_phase, TRACK)
        self.assertAlmostEqual(
            self.agent.body_ref[2],
            self.agent.planned_base_height,
        )
        np.testing.assert_allclose(
            self.agent.body_ref[3:7],
            [1.0, 0.0, 0.0, 0.0],
        )

    def test_recovers_to_stand_before_walking_to_a_new_goal(self) -> None:
        observation = self.stand_observation.copy()
        observation[2] = 0.40
        self.agent.base_height_cmd = 0.40
        next_base_xy = observation[:2] + np.array([0.30, 0.0])

        with patch.object(
            self.agent,
            "_plan_base_reference",
            return_value=(next_base_xy, self.agent.stand_base_height),
        ):
            self.agent._activate_plan(observation)

        self.assertEqual(self.agent.motion_phase, RECOVER)
        held_xy = observation[:2].copy()
        while self.agent.motion_phase == RECOVER:
            self.agent._update_motion_reference(observation)
            observation[2] = self.agent.base_height_cmd
            np.testing.assert_allclose(self.agent.body_ref[:2], held_xy)

        self.assertEqual(self.agent.motion_phase, APPROACH)
        self.assertAlmostEqual(
            self.agent.base_height_cmd,
            self.agent.stand_base_height,
        )


if __name__ == "__main__":
    unittest.main()
