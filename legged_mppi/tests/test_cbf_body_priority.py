"""Regressions for EE corrections disturbing locomotion and body dynamics."""
import unittest
from unittest.mock import patch

import numpy as np

from whole_body_mppi.control.tasks.ee_velocity import EEVelocityController
from whole_body_mppi.control.safety.cbf import configure_cbf


class BodyPriorityCBFTests(unittest.TestCase):
    def agent(self):
        a = EEVelocityController('ee_walk_cmd_vel')
        self.addCleanup(a.shutdown)
        a.set_params(12, .1, max(6, 2*a.n_iterations))
        configure_cbf(a, 'on')
        return a

    @staticmethod
    def observation(a):
        return np.r_[a.model.key('home').qpos, np.zeros(a.model.nv)]

    def test_cmd_body_envelope_detects_height_and_angular_motion_before_fall(self):
        a = self.agent()
        obs = self.observation(a)
        h0, _ = a.cbf.barriers(obs)
        obs[2] = a.body_ref[2]+a.cbf.options.body_height_tolerance+.01
        h1, _ = a.cbf.barriers(obs)
        self.assertLess(h1[a.cbf.barrier_groups == 0].min(), 0.)
        obs = self.observation(a)
        obs[a.model.nq+3] = a.cbf.options.body_angular_speed_max*1.1
        h2, _ = a.cbf.barriers(obs)
        self.assertLess(h2[a.cbf.barrier_groups == 0].min(), 0.)
        self.assertGreaterEqual(h0[a.cbf.barrier_groups == 0].min(), 0.)

    def test_body_guard_detects_reaction_beyond_short_qp_window(self):
        a = self.agent()
        initial = np.r_[0., self.observation(a)]
        nominal = np.tile(initial, (a.horizon, 1))
        candidates = np.tile(nominal, (4, 1, 1))
        late = a.cbf.options.shield_steps+1
        angle = np.deg2rad(5.)
        candidates[0, late:, 4:8] = [np.cos(angle/2), np.sin(angle/2), 0., 0.]
        candidates[1, late:, 3] += .02
        candidates[2, late:, 4+a.model.nq] += .4
        candidates[3, late:, 3] += .001
        np.testing.assert_array_equal(a.cbf._body_guard(nominal, candidates), [False, False, False, True])

    def test_filter_rejects_ee_improvement_with_delayed_body_tilt(self):
        a = self.agent()
        cbf, obs = a.cbf, self.observation(a)
        actions = np.tile(a.sampling_init, (a.horizon, 1))
        seen = []
        def predict(plans, observation, return_states=False):
            seen.append(plans.copy())
            delta = plans[:, 0, 12]-actions[0, 12]
            states = np.tile(np.r_[0., obs], (len(plans), plans.shape[1], 1))
            # The arm improves EE, but a delayed reaction tilts the base. A
            # short-prefix verification and global-min-only acceptance miss it.
            late = cbf.options.shield_steps+1
            angle = np.deg2rad(1000.)*delta
            if plans.shape[1] > late:
                states[:, late:, 4] = np.cos(angle[:, None]/2)
                states[:, late:, 5] = np.sin(angle[:, None]/2)
            values = np.ones((len(plans), plans.shape[1], 3*len(cbf.barrier_groups)))
            ee = np.tile(cbf.barrier_groups == 2, 3)
            values[..., ee] = -1+10*delta[:, None, None]
            values = values.reshape(len(plans), -1)
            return (values, states) if return_states else values
        with patch.object(cbf, '_predict', side_effect=predict):
            corrected = cbf.filter_plan(actions, obs)
        np.testing.assert_array_equal(corrected, actions)
        self.assertTrue(cbf.last['body_guard_rejected'])
        self.assertEqual(cbf.last['leg_correction_norm'], 0.)
        self.assertTrue(any(len(batch) > 1 for batch in seen))
        for batch in seen:
            np.testing.assert_array_equal(batch[..., :12], np.broadcast_to(actions[:batch.shape[1], :12], batch[..., :12].shape))

    def test_body_recovery_defers_ee_filter_instead_of_changing_gait(self):
        a = self.agent()
        obs = self.observation(a)
        obs[2] = .1
        actions = np.tile(a.sampling_init, (a.horizon, 1))
        corrected = a.cbf.filter_plan(actions, obs)
        np.testing.assert_array_equal(corrected, actions)
        self.assertTrue(a.cbf.last['ee_correction_deferred'])
        self.assertEqual(a.cbf.last['extra_rollouts'], 1)
        self.assertTrue(a.cbf.last['infeasible'])


if __name__ == '__main__':
    unittest.main()
