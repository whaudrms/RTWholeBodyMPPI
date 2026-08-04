import unittest

import numpy as np

from whole_body_mppi.control.controllers.mppi_locomotion import MPPI


class CpuRolloutBackendTest(unittest.TestCase):
    def test_update_runs_with_installed_mujoco_rollout_api(self):
        controller = MPPI("stand", backend="cpu")
        self.addCleanup(controller.shutdown)
        controller.set_params(4, controller.temperature, 5)
        observation = np.concatenate(
            (controller.model.qpos0, np.zeros(controller.model.nv))
        )

        action = controller.update(observation, advance_steps=0)

        self.assertEqual(action.shape, (controller.model.nu,))
        self.assertEqual(
            controller.selected_trajectory.shape,
            (controller.horizon, controller.model.nu),
        )
        self.assertTrue(np.all(np.isfinite(action)))


if __name__ == "__main__":
    unittest.main()
