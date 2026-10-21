"""Integration checks for the mounted arm, actuator mapping and free dynamics."""
from pathlib import Path

import unittest

import mujoco
import numpy as np

MODELS = Path(__file__).resolve().parents[1] / 'whole_body_mppi/models'


def load():
    model = mujoco.MjModel.from_xml_path(str(MODELS / 'go1_dm/scene.xml'))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_forward(model, data)
    return model, data


class Go1DMModelTests(unittest.TestCase):
    def test_mapping_and_mass(self):
        model, data = load()
        bare = mujoco.MjModel.from_xml_path(str(MODELS / 'go1/go1_scene_mppi.xml'))
        assert (model.nq, model.nv, model.nu) == (27, 26, 19)
        for i in range(12):
            assert model.actuator(i).name == bare.actuator(i).name
            assert model.jnt_qposadr[model.actuator_trnid[i, 0]] == bare.jnt_qposadr[bare.actuator_trnid[i, 0]]
        assert [model.actuator(i).name for i in range(12, 19)] == [
            *(f'dm_joint{i}' for i in range(1, 7)), 'dm_finger_left']
        np.testing.assert_allclose(model.body_mass.sum() - bare.body_mass.sum(), 5.1291, atol=1e-6)
        assert model.body('dm_mount').parentid == model.body('trunk').id
        assert model.body('dm_base_link').parentid == model.body('dm_mount').id
        # Home must start without arm/body or arm/ground penetrations.
        assert all(c.dist >= -1e-5 for c in data.contact)


    def test_gripper_mimic_and_posture_hold(self):
        model, data = load()
        # Open the gripper while the floating robot holds its standing posture.
        data.ctrl[model.actuator('dm_finger_left').id] = 0.035
        for _ in range(2500):
            mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all()
        assert not np.any(data.warning.number)
        assert 0.18 < data.qpos[2] < 0.32
        assert abs(data.qpos[3]) > 0.98
        left = data.qpos[model.joint('dm_finger_left').qposadr[0]]
        right = data.qpos[model.joint('dm_finger_right').qposadr[0]]
        assert left > 0.03
        assert abs(left + right) < 1e-4
        np.testing.assert_allclose(data.qpos[19:25], model.key('home').qpos[19:25], atol=0.2)


if __name__ == "__main__":
    unittest.main()
