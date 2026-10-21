"""Leg-action rollouts with held arm controls and optional welded-finger model."""
import mujoco
import numpy as np


class MountedArmRollout:
    def __init__(self, backend, control_dt, observation_model=None):
        self.backend = backend
        self.model = observation_model if observation_model is not None else backend.model
        prediction = backend.model
        self.arm_control = self.model.key('home').ctrl[12:].copy()
        self.prediction_control = prediction.key('home').ctrl[12:].copy()
        ratio = control_dt / prediction.opt.timestep
        self.substeps = round(ratio)
        if self.substeps < 1 or not np.isclose(ratio, self.substeps):
            raise ValueError('Control timestep must be an integer multiple of physics timestep')
        # Map by joint name instead of relying on reduced/full state offsets.
        self.state_indices = [0]
        for field, offset in (('qpos', 1), ('qvel', 1+self.model.nq)):
            for j in range(prediction.njnt):
                name = prediction.joint(j).name
                source = self.model.joint(name) if name else self.model.joint(j)
                kind = prediction.jnt_type[j]
                width = (7 if field == 'qpos' else 6) if kind == mujoco.mjtJoint.mjJNT_FREE else 1
                adr = int(getattr(source, 'qposadr' if field == 'qpos' else 'dofadr')[0])
                self.state_indices.extend(range(offset+adr, offset+adr+width))
        self.state_indices = np.asarray(self.state_indices)
        self.identity_state = np.array_equal(self.state_indices, np.arange(1+self.model.nq+self.model.nv))
        self.full_home = np.r_[0., self.model.key('home').qpos, np.zeros(self.model.nv)]
        self._state_buffer = None
        self._control_buffer = None

    def expand_actions(self, actions):
        actions = np.asarray(actions)
        arm = np.broadcast_to(self.arm_control, actions.shape[:-1] + self.arm_control.shape)
        return np.ascontiguousarray(np.concatenate((actions, arm), axis=-1))

    def __call__(self, state, ctrl, initial_state, num_workers=32, nstep=5):
        count, horizon = ctrl.shape[:2]
        shape = (count, horizon*self.substeps)
        state_shape = shape + (len(self.state_indices),)
        control_shape = shape + (self.backend.model.nu,)
        if self._state_buffer is None or self._state_buffer.shape != state_shape:
            self._state_buffer = np.empty(state_shape)
            self._control_buffer = np.empty(control_shape)
        # Fill the repeated commands in-place, avoiding np.repeat allocations.
        commands = self._control_buffer.reshape(count, horizon, self.substeps, -1)
        commands[..., :12] = ctrl[:, :, None, :]
        commands[..., 12:] = self.prediction_control
        if self.substeps == 1 and self.identity_state:
            self.backend.threaded_rollout(state, self._control_buffer,
                                          np.ascontiguousarray(initial_state),
                                          num_workers=num_workers, nstep=horizon)
            return
        initial = np.ascontiguousarray(initial_state[:, self.state_indices])
        self.backend.threaded_rollout(self._state_buffer, self._control_buffer, initial,
                                      num_workers=num_workers, nstep=horizon*self.substeps)
        sampled = self._state_buffer[:, self.substeps-1::self.substeps]
        state[:] = self.full_home
        state[:, :, self.state_indices] = sampled
