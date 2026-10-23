"""Optimize legs and arm J1-J4 while holding J5/J6 home position commands."""
import numpy as np


class FixedWristRollout:
    def __init__(self, backend):
        self.backend = backend
        if backend.model.nu != 18:
            raise ValueError('Fixed wrist mapping requires the 18-actuator Go1+DM model')
        self.fixed_commands = backend.model.key('home').ctrl[16:18].copy()

    def expand_actions(self, actions):
        actions = np.asarray(actions)
        if actions.shape[-1] != 16:
            raise ValueError('Expected 16 optimized controls')
        fixed = np.broadcast_to(self.fixed_commands, actions.shape[:-1]+(2,))
        return np.ascontiguousarray(np.concatenate((actions, fixed), axis=-1))

    def __call__(self, state, ctrl, initial_state, num_workers=32, nstep=5):
        return self.backend.threaded_rollout(state, self.expand_actions(ctrl), initial_state,
                                             num_workers=num_workers, nstep=nstep)
