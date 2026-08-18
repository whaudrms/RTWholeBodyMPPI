"""Model-contract tests for the roll- and jaw-free Go2 + SO-ARM100."""

from __future__ import annotations

import unittest
from pathlib import Path

import mujoco
import numpy as np

from mani_mppi.control.controllers.mppi_locomotion import MPPI


MODELS = Path(__file__).resolve().parents[1] / "mani_mppi" / "models"


class Go2SOArmModelTest(unittest.TestCase):
    def test_legged_mppi_mujoco_physics_contract(self) -> None:
        for filename in ("scene.xml", "scene_big_box.xml", "scene_push_box.xml"):
            with self.subTest(filename=filename):
                model = mujoco.MjModel.from_xml_path(str(MODELS / filename))
                self.assertEqual(
                    model.opt.integrator, mujoco.mjtIntegrator.mjINT_EULER
                )
                self.assertAlmostEqual(model.opt.timestep, 0.01)
                floor = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
                )
                base = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM, "base_collision"
                )
                self.assertEqual(model.geom_condim[floor], 3)
                self.assertEqual(model.geom_condim[base], 1)
                np.testing.assert_allclose(
                    model.geom_friction[floor], [0.8, 0.005, 0.0001]
                )

    def test_scene_dimensions_and_no_wrist_roll_or_jaw(self) -> None:
        expected = {
            "scene.xml": (23, 22, 16),
            "scene_big_box.xml": (23, 22, 16),
            "scene_push_box.xml": (30, 28, 16),
        }
        for filename, dimensions in expected.items():
            with self.subTest(filename=filename):
                model = mujoco.MjModel.from_xml_path(
                    str(MODELS / filename)
                )
                self.assertEqual((model.nq, model.nv, model.nu), dimensions)
                actuator_names = {
                    mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_ACTUATOR, index
                    )
                    for index in range(model.nu)
                }
                joint_names = {
                    mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_JOINT, index
                    )
                    for index in range(model.njnt)
                }
                self.assertNotIn("Wrist_Roll", actuator_names | joint_names)
                self.assertNotIn("Jaw", actuator_names | joint_names)
                self.assertGreaterEqual(
                    mujoco.mj_name2id(
                        model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        "fixed_wrist_roll_housing_visual",
                    ),
                    0,
                )
                for body_name in ("Fixed_Jaw", "Moving_Jaw"):
                    self.assertEqual(
                        mujoco.mj_name2id(
                            model, mujoco.mjtObj.mjOBJ_BODY, body_name
                        ),
                        -1,
                    )

    def test_raw_go2_state_is_canonicalized_to_actuator_order(self) -> None:
        agent = MPPI(task="stand", rollout_mode="safe_spline")
        try:
            key_id = mujoco.mj_name2id(
                agent.model, mujoco.mjtObj.mjOBJ_KEY, "stand"
            )
            raw = np.concatenate(
                (agent.model.key_qpos[key_id], np.zeros(agent.model.nv))
            )[None, :]
            canonical = agent.canonical_robot_state(raw)

            self.assertEqual(canonical.shape, (1, 45))
            np.testing.assert_allclose(
                canonical[0, 7:23], agent.model.key_ctrl[key_id]
            )
            np.testing.assert_allclose(canonical[0, 29:45], 0.0)
        finally:
            agent.close()


if __name__ == "__main__":
    unittest.main()
