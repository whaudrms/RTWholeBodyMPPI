import unittest

try:
    import numpy as np
except ImportError:
    np = None

try:
    from whole_body_mppi.control.controllers.warp_mppi import (
        locomotion_cost,
        push_box_cost,
    )
except (ImportError, ModuleNotFoundError):
    locomotion_cost = None
    push_box_cost = None


@unittest.skipUnless(
    np is not None and locomotion_cost is not None,
    "NumPy/JAX/MJX is not installed",
)
class WarpCostParityTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def _rollout_inputs(self, controller, samples=3, horizon=4):
        qpos = np.tile(controller.model.qpos0, (samples, horizon, 1))
        qpos[..., :3] += self.rng.normal(0.0, 0.02, (samples, horizon, 3))
        qvel = self.rng.normal(
            0.0, 0.1, (samples, horizon, controller.model.nv)
        )
        actions = self.rng.normal(0.0, 0.2, (samples, horizon, 12))
        joints_ref = controller.gait_scheduler.gait[
            :, controller.gait_scheduler.indices[:horizon]
        ]
        joint_qvel_indices = controller.model.jnt_dofadr[
            controller.model.actuator_trnid[:, 0]
        ]
        return qpos, qvel, actions, joints_ref, joint_qvel_indices

    def test_locomotion_cost_matches_numpy(self):
        from whole_body_mppi.control.controllers.mppi_locomotion import MPPI

        controller = MPPI("stand")
        self.addCleanup(controller.shutdown)
        controller.set_params(4, controller.temperature, 3)
        qpos, qvel, actions, joints_ref, joint_qvel_indices = (
            self._rollout_inputs(controller)
        )

        states = np.concatenate((qpos, qvel), axis=-1)
        expected = controller.calculate_total_cost(
            states.copy(),
            actions.copy(),
            joints_ref.copy(),
            controller.body_ref.copy(),
        )
        actual = np.asarray(
            locomotion_cost(
                qpos,
                qvel,
                actions,
                joints_ref,
                controller.body_ref,
                np.diag(controller.Q),
                np.diag(controller.R),
                controller.joint_qpos_indices,
                joint_qvel_indices,
            )
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=0.1)

    def test_push_box_cost_matches_numpy(self):
        from whole_body_mppi.control.controllers.mppi_locomanipulation import (
            MPPI_box_push,
        )

        controller = MPPI_box_push("push_box")
        self.addCleanup(controller.shutdown)
        controller.set_params(4, controller.temperature, 3)
        qpos, qvel, actions, joints_ref, joint_qvel_indices = (
            self._rollout_inputs(controller)
        )

        robot_state = np.concatenate((qpos[..., 7:26], qvel[..., 6:24]), axis=-1)
        box_state = np.concatenate((qpos[..., :7], qvel[..., :6]), axis=-1)
        expected = controller.calculate_total_cost(
            robot_state.copy(),
            box_state.copy(),
            actions.copy(),
            joints_ref.copy(),
            controller.body_ref.copy(),
        )
        actual = np.asarray(
            push_box_cost(
                qpos,
                qvel,
                actions,
                joints_ref,
                controller.body_ref,
                controller.x_box_ref,
                np.diag(controller.Q_robot),
                np.diag(controller.Q_box),
                np.diag(controller.R),
                controller.joint_qpos_indices,
                joint_qvel_indices,
            )
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=0.1)


if __name__ == "__main__":
    unittest.main()
