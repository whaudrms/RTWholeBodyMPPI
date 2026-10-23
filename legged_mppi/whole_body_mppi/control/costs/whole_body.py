"""Whole-body state and torque costs in MuJoCo qpos + qvel order."""
import numpy as np

from whole_body_mppi.utils.transforms import batch_world_to_local_velocity


class WholeBodyCost:
    def calculate_total_cost(self, states, actions, joints_ref, body_ref):
        # 25 positions: base(7), legs(12), arm(6).
        # 24 velocities: base(6), legs(12), arm(6).
        samples, steps = states.shape[:2]
        x = states.reshape(-1, 49).copy()
        u = actions.reshape(-1, 18)
        gait = np.broadcast_to(joints_ref.T, (samples, steps, 24)).reshape(-1, 24)
        ref = np.empty_like(x)
        ref[:, :7] = body_ref[:7]
        ref[:, 7:19] = gait[:, :12]
        ref[:, 19:25] = self.arm_joint_ref
        ref[:, 25:31] = body_ref[7:]
        ref[:, 31:43] = gait[:, 12:]
        ref[:, 43:49] = self.arm_velocity_ref
        if getattr(self, 'velocity_tracking', False):
            from whole_body_mppi.control.costs.cmd_vel import heading_velocity
            velocity, yaw_rate, tilt = heading_velocity(x[:, 3:7], x[:, 25:28], x[:, 28:31])
            x[:, 25:28] = velocity
            x[:, 30] = yaw_rate
            x[:, 3:7] = tilt
        else:
            x[:, 25:28] = batch_world_to_local_velocity(x[:, 3:7], x[:, 25:28])
        error = x - ref
        error[:, 3:7] = self.quaternion_distance_np(x[:, 3:7], ref[:, 3:7])[:, None]
        # Preserve the locomotion objective's L1 base-position cost.
        position_cost = np.abs(error[:, :3] @ self.Q[:3, :3]).sum(axis=1)
        error[:, :3] = 0
        torque = np.empty_like(u)
        torque[:, :12] = 50 * (u[:, :12] - x[:, 7:19]) - 3 * x[:, 31:43]
        torque[:, 12:] = 55 * (u[:, 12:] - x[:, 19:25]) - 3 * x[:, 43:49]
        cost = (np.einsum('bi,ij,bj->b', error, self.Q, error)
                + np.einsum('bi,ij,bj->b', torque, self.R, torque) + position_cost)
        return cost.reshape(samples, steps).sum(axis=1)
