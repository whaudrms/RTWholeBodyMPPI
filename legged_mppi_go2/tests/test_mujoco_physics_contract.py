"""MuJoCo settings shared with the legged_mppi runtime baseline."""

from pathlib import Path
import unittest

import mujoco
import numpy as np


MODELS = Path(__file__).resolve().parents[1] / "whole_body_mppi" / "models" / "go2"


class MujocoPhysicsContractTest(unittest.TestCase):
    def test_active_scenes_use_legged_mppi_contact_settings(self) -> None:
        for filename in (
            "scene.xml", "scene_big_box.xml", "scene_push_box.xml",
            "scene_stairs.xml", "scene_climb_box.xml",
        ):
            with self.subTest(filename=filename):
                model = mujoco.MjModel.from_xml_path(str(MODELS / filename))
                self.assertEqual(
                    model.opt.integrator, mujoco.mjtIntegrator.mjINT_EULER
                )
                self.assertAlmostEqual(model.opt.timestep, 0.01)
                floor = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
                )
                self.assertEqual(model.geom_condim[floor], 3)
                np.testing.assert_allclose(
                    model.geom_friction[floor], [0.8, 0.005, 0.0001]
                )

                active = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
                foot_geoms = np.flatnonzero(
                    active
                    & (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE)
                    & np.isclose(model.geom_size[:, 0], 0.022)
                )
                self.assertEqual(len(foot_geoms), 4)
                np.testing.assert_array_equal(model.geom_condim[foot_geoms], 6)


if __name__ == "__main__":
    unittest.main()
