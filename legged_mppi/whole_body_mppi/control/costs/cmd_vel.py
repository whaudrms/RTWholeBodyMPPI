"""Go1 velocity tracking in a yaw-only heading frame, without waypoint/yaw costs."""
import numpy as np
from scipy.spatial.transform import Rotation


def heading_velocity(quaternions, world_velocity, body_omega):
    """Return heading linear velocity, Euler yaw rate, and yaw-free orientation.

    MuJoCo free-joint linear velocity is world-frame and angular velocity is
    body-frame. Euler yaw is singular at vertical pitch; clamp its denominator
    for finite cost evaluation of fallen rollout samples.
    """
    rotation = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]])
    matrix = rotation.as_matrix()
    yaw = np.arctan2(matrix[:, 1, 0], matrix[:, 0, 0])
    heading = Rotation.from_rotvec(np.column_stack((np.zeros_like(yaw), np.zeros_like(yaw), yaw)))
    velocity = heading.inv().apply(world_velocity)
    # d(R e_x)/dt = R (omega_body cross e_x).
    axis_dot = matrix[:, :, 1] * body_omega[:, 2, None] - matrix[:, :, 2] * body_omega[:, 1, None]
    yaw_rate = (matrix[:, 0, 0] * axis_dot[:, 1] - matrix[:, 1, 0] * axis_dot[:, 0]) / np.maximum(
        matrix[:, 0, 0]**2 + matrix[:, 1, 0]**2, 1e-8)
    tilt = (heading.inv() * rotation).as_quat()[:, [3, 0, 1, 2]]
    return velocity, yaw_rate, tilt


def command_velocity_cost(controller, states, actions):
    samples, steps = states.shape[:2]
    x = states[..., 1:].reshape(-1, 37).copy()
    velocity, yaw_rate, tilt = heading_velocity(x[:, 3:7], x[:, 19:22], x[:, 22:25])
    x[:, 19:22] = velocity
    x[:, 24] = yaw_rate
    x[:, 3:7] = tilt
    body = np.broadcast_to(controller.body_ref, (samples * steps, 13))
    gait = np.broadcast_to(controller.joints_ref.T, (samples, steps, 24)).reshape(-1, 24)
    ref = np.concatenate((body[:, :7], gait[:, :12], body[:, 7:], gait[:, 12:]), axis=1)
    costs = controller.quadruped_cost_np(x, actions.reshape(-1, 12), ref)
    return costs.reshape(samples, steps).sum(axis=1)
