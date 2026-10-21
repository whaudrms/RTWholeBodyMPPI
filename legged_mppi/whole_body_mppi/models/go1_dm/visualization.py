"""Detailed DM appearance driven by a simplified physics model, without stepping."""
import mujoco

from whole_body_mppi.models.go1_dm.build_model import load_model


class DetailedDMVisual:
    def __init__(self, source_path, physics_model):
        self.model = load_model(source_path)
        self.data = mujoco.MjData(self.model)
        # Unmapped arm joints (rigid payload mode) and fingers stay at home.
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key('home').id)
        self.joint_slices = []
        for j in range(physics_model.njnt):
            joint = physics_model.joint(j)
            target = self.model.joint(joint.name) if joint.name else self.model.joint(j)
            kind = physics_model.jnt_type[j]
            width = 7 if kind == mujoco.mjtJoint.mjJNT_FREE else (4 if kind == mujoco.mjtJoint.mjJNT_BALL else 1)
            src, dst = int(joint.qposadr[0]), int(target.qposadr[0])
            self.joint_slices.append((slice(src, src+width), slice(dst, dst+width)))

    def sync(self, physics_data):
        for source, target in self.joint_slices:
            self.data.qpos[target] = physics_data.qpos[source]
        self.data.time = physics_data.time
        # Update transforms/COM for geometry, cameras and lights. No contact
        # detection, constraint solve, integration or controller execution.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
