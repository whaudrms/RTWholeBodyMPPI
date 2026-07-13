"""Checks for the 32-row B2-Z1 reference and locomotion MPPI integration."""

import numpy as np

from legged_mani.control import B2Z1LocomotionMPPI
from legged_mani.control.mppi_locomotion import MPPI
from legged_mani.control.gait_scheduler import JointReferenceScheduler
from legged_mani.interface import B2Z1Env
from legged_mani.utils.tasks import get_task


def run_check() -> None:
    assert B2Z1LocomotionMPPI is MPPI
    task = get_task("in_place")
    assert task.control_decimation == 1
    assert task.keyframe == "sit"
    synthetic = np.arange(32 * 3, dtype=float).reshape(32, 3)
    scheduler = JointReferenceScheduler(synthetic)
    assert scheduler.horizon(5).shape == (32, 5)
    assert np.array_equal(scheduler.horizon(5), synthetic[:, [0, 1, 2, 0, 1]])
    scheduler.advance(2)
    assert np.array_equal(scheduler.horizon(3), synthetic[:, [2, 0, 1]])

    env = B2Z1Env()
    observation = env.reset("sit")
    stand_control = env.model.key_ctrl[env.model.key("stand").id]
    with B2Z1LocomotionMPPI() as controller:
        expected_sampling_init = np.array(
            [-0.3, 1.34, -2.83, 0.3, 1.34, -2.83] * 2
            + [0.0, 1.0, -0.6, 0.0]
        )
        assert np.allclose(controller.sampling_init, expected_sampling_init)
        assert np.allclose(controller.body_ref[:7], env.model.key_qpos[env.model.key("stand").id][:7])
        assert np.allclose(controller.noise_sigma[:12], [0.06, 0.1, 0.1] * 4)
        assert np.allclose(np.diag(controller.Q)[7:23], 800.0)
        assert np.allclose(np.diag(controller.R), 0.001)
        assert controller.cost_kp == 50.0
        assert controller.cost_kd == 3.0
        assert controller.model.opt.cone == 0
        assert controller.model.opt.enableflags & 1
        action = controller.update(observation)
        assert action.shape == (16,)
        assert controller.joints_ref.shape == (32, 40)
        assert controller.reference.shape == (40, 45)
        full_gait = controller.gait_scheduler.gait
        assert full_gait.shape == (32, 100)
        assert np.allclose(full_gait[:12].mean(axis=1), stand_control[:12])
        assert np.any(np.ptp(full_gait[:12], axis=1) > 0.1)
        assert np.allclose(full_gait[12:16], stand_control[12:16, None])
        assert np.any(np.abs(full_gait[16:28]) > 0.1)
        assert np.allclose(full_gait[28:32], 0.0)
        assert set(controller.gaits) == {"in_place", "walk_fast"}
        assert np.isfinite(controller.last_costs).all()
        selected = controller.selected_trajectory.copy()
        controller.advance_reference(5)
        assert controller.joint_reference.phase == 5
        assert controller.reference_step == 5
        assert np.allclose(controller.trajectory[0], selected[5])

    walk_task = get_task("walk_straight")
    assert walk_task.keyframe == "sit"
    assert walk_task.control_decimation == 1
    with B2Z1LocomotionMPPI(task="walk_straight") as controller:
        assert controller.goal_pos.shape == (3, 3)
        assert np.allclose(controller.goal_pos[:, 0], [0.0, 1.0, 1.0])
        assert controller.desired_gait == ["in_place", "walk_fast", "in_place"]
        assert controller.gait_scheduler.name == "in_place"

        controller.next_goal()
        assert controller.goal_index == 1
        assert controller.gait_scheduler.name == "walk_fast"
        assert np.allclose(controller.body_ref[:3], [1.0, 0.0, 0.55])
        assert np.allclose(controller.body_ref[7:10], [0.2, 0.0, 0.0])

        controller.next_goal()
        assert controller.goal_index == 2
        assert controller.gait_scheduler.name == "in_place"
        assert np.allclose(controller.body_ref[7:10], 0.0)

        controller.next_goal()
        assert controller.task_success

    print("locomotion: PASS (joint_ref=(32, 40), state_ref=(40, 45), action=(16,))")


if __name__ == "__main__":
    run_check()
