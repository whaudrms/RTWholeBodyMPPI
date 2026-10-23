"""Batched DM TCP kinematics for rollout costs, without per-state mj_forward."""
import mujoco
import numpy as np


def quaternion_matrix(q):
    q = np.asarray(q, dtype=float)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=-1).reshape(q.shape[:-1]+(3,3))


class EndEffectorObjective:
    def __init__(self, model, target, frame='world', weight=2000., terminal_weight=5.):
        self.target = np.asarray(target, dtype=float)
        if self.target.shape != (3,) or not np.isfinite(self.target).all():
            raise ValueError('EE target must contain three finite coordinates in metres')
        if frame not in ('world', 'base'):
            raise ValueError('EE frame must be world or base')
        if not np.isfinite([weight, terminal_weight]).all() or weight <= 0 or terminal_weight < 0:
            raise ValueError('EE weight must be positive and terminal weight nonnegative')
        self.frame, self.weight, self.terminal_weight = frame, weight, terminal_weight
        site = model.site('dm_tcp').id
        self.site_pos = model.site_pos[site].copy()
        body = int(model.site_bodyid[site])
        trunk = model.body('trunk').id
        chain = []
        while body != trunk:
            chain.append(body)
            body = int(model.body_parentid[body])
        self.chain = []
        for body in reversed(chain):
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, model.body_quat[body])
            joints = []
            for j in range(model.body_jntadr[body], model.body_jntadr[body]+model.body_jntnum[body]):
                if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError('DM TCP chain requires hinge joints')
                axis = model.jnt_axis[j].copy()
                x,y,z = axis
                skew = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
                adr = model.jnt_qposadr[j]
                joints.append((adr, model.qpos0[adr], model.jnt_pos[j].copy(), skew))
            self.chain.append((model.body_pos[body].copy(), rotation.reshape(3,3), joints))

    def positions(self, qpos):
        qpos = np.asarray(qpos)
        rotation = quaternion_matrix(qpos[..., 3:7]) if self.frame == 'world' else np.broadcast_to(np.eye(3), qpos.shape[:-1]+(3,3)).copy()
        position = qpos[..., :3].copy() if self.frame == 'world' else np.zeros(qpos.shape[:-1]+(3,))
        for offset, orientation, joints in self.chain:
            position += np.einsum('...ij,j->...i', rotation, offset)
            rotation = rotation @ orientation
            for adr, zero, anchor, skew in joints:
                angle = qpos[..., adr]-zero
                relative = np.eye(3)+np.sin(angle)[...,None,None]*skew+(1-np.cos(angle))[...,None,None]*(skew@skew)
                new_rotation = rotation @ relative
                position += np.einsum('...ij,j->...i', rotation-new_rotation, anchor)
                rotation = new_rotation
        return position+np.einsum('...ij,j->...i', rotation, self.site_pos)

    def cost(self, qpos):
        error = np.sum((self.positions(qpos)-self.target)**2, axis=-1)
        return self.weight*(error.sum(axis=1)+self.terminal_weight*error[:, -1])

    def target_world(self, qpos):
        if self.frame == 'world':
            return self.target.copy()
        return qpos[:3]+quaternion_matrix(qpos[3:7])@self.target


class EndEffectorVelocityObjective(EndEffectorObjective):
    """Instantaneous TCP linear velocity; base means derivative relative to trunk."""
    velocity_tracking = True

    def __init__(self, model, command, duration, frame='world', weight=2000., terminal_weight=5.):
        super().__init__(model, command, frame, weight, terminal_weight)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError('EE command duration must be finite and positive')
        self.duration = float(duration)
        self.dof_addresses = {int(model.jnt_qposadr[j]): int(model.jnt_dofadr[j]) for j in range(model.njnt)}

    def command_at(self, time):
        time = np.asarray(time)
        return np.where((time < self.duration)[..., None], self.target, 0.)

    def velocities(self, qpos, qvel):
        qpos, qvel = np.asarray(qpos), np.asarray(qvel)
        rotation = quaternion_matrix(qpos[..., 3:7]) if self.frame == 'world' else np.broadcast_to(np.eye(3), qpos.shape[:-1]+(3,3)).copy()
        position = qpos[..., :3].copy() if self.frame == 'world' else np.zeros(qpos.shape[:-1]+(3,))
        axes = []
        for offset, orientation, joints in self.chain:
            position += np.einsum('...ij,j->...i', rotation, offset)
            rotation = rotation @ orientation
            for adr, zero, anchor, skew in joints:
                axis = np.array([skew[2,1], skew[0,2], skew[1,0]])
                axes.append((np.einsum('...ij,j->...i', rotation, axis),
                             position + np.einsum('...ij,j->...i', rotation, anchor),
                             qvel[..., self.dof_addresses[adr]]))
                angle = qpos[..., adr]-zero
                relative = np.eye(3)+np.sin(angle)[...,None,None]*skew+(1-np.cos(angle))[...,None,None]*(skew@skew)
                new_rotation = rotation @ relative
                position += np.einsum('...ij,j->...i', rotation-new_rotation, anchor)
                rotation = new_rotation
        tcp = position + np.einsum('...ij,j->...i', rotation, self.site_pos)
        velocity = np.zeros_like(tcp)
        for axis, origin, rate in axes:
            velocity += np.cross(axis, tcp-origin)*rate[...,None]
        if self.frame == 'world':
            world_omega = np.einsum('...ij,...j->...i', quaternion_matrix(qpos[...,3:7]), qvel[...,3:6])
            velocity += qvel[...,:3] + np.cross(world_omega, tcp-qpos[...,:3])
        return velocity

    def cost(self, qpos, qvel, reference):
        error = np.sum((self.velocities(qpos, qvel)-reference)**2, axis=-1)
        return self.weight*(error.sum(axis=1)+self.terminal_weight*error[:,-1])
