"""MuJoCo settings shared with the legged_mppi runtime baseline."""

from pathlib import Path
import unittest

import mujoco
import numpy as np


MODELS = Path(__file__).resolve().parents[1] / "mani_mppi" / "models"


class MujocoPhysicsContractTest(unittest.TestCase):
    def test_active_scenes_use_legged_mppi_contact_settings(self) -> None:
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
                for leg in ("FR", "FL", "RR", "RL"):
                    foot = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_foot"
                    )
                    self.assertEqual(model.geom_condim[foot], 6)

                if filename == "scene_push_box.xml":
                    box = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom"
                    )
                    box_body = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_BODY, "box"
                    )
                    self.assertEqual(model.geom_condim[box], 3)
                    self.assertEqual(model.geom_priority[box], 0)
                    self.assertAlmostEqual(model.geom_margin[box], 0.0)
                    np.testing.assert_allclose(
                        model.geom_size[box], [0.19, 0.19, 0.19]
                    )
                    np.testing.assert_allclose(
                        model.geom_friction[box], [0.0, 0.005, 0.0001]
                    )
                    np.testing.assert_allclose(
                        model.body_pos[box_body], [1.0, 0.0, 0.19]
                    )
                    self.assertAlmostEqual(model.body_mass[box_body], 3.70)
                    np.testing.assert_allclose(
                        model.body_inertia[box_body],
                        [0.0749, 0.0749, 0.0849],
                    )
                    box_joint = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_JOINT, "box_joint"
                    )
                    box_qpos = model.jnt_qposadr[box_joint]
                    for key_name in ("sit", "stand"):
                        key = mujoco.mj_name2id(
                            model, mujoco.mjtObj.mjOBJ_KEY, key_name
                        )
                        np.testing.assert_allclose(
                            model.key_qpos[key, box_qpos:box_qpos + 7],
                            [1.0, 0.0, 0.19, 1.0, 0.0, 0.0, 0.0],
                        )


if __name__ == "__main__":
    unittest.main()
