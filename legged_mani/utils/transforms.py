"""Frame and orientation transforms adapted from the original MPPI utilities."""

import numpy as np
from scipy.spatial.transform import Rotation as R


def batch_world_to_local_velocity(quaternions, world_velocities):
    """Transform batched world velocities into each quaternion's body frame.

    Quaternions use MuJoCo's ``[w, x, y, z]`` ordering.
    """
    quaternions = np.asarray(quaternions, dtype=float)
    world_velocities = np.asarray(world_velocities, dtype=float)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"Expected quaternion shape (N, 4), got {quaternions.shape}")
    if world_velocities.shape != (quaternions.shape[0], 3):
        raise ValueError(
            f"Expected velocity shape {(quaternions.shape[0], 3)}, "
            f"got {world_velocities.shape}"
        )
    rotation = R.from_quat(quaternions[:, [1, 2, 3, 0]])
    return rotation.inv().apply(world_velocities)


def calculate_orientation_quaternion(current_point, goal_point):
    """Return a ``[w, x, y, z]`` quaternion pointing toward a 3-D goal."""
    current_point = np.asarray(current_point, dtype=float)
    goal_point = np.asarray(goal_point, dtype=float)
    direction = goal_point - current_point
    length = np.linalg.norm(direction)
    if length <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    direction_normalized = direction / length
    yaw = np.arctan2(direction_normalized[1], direction_normalized[0])
    pitch = -np.arctan2(
        direction_normalized[2],
        np.hypot(direction_normalized[0], direction_normalized[1]),
    )
    yaw_quat = R.from_euler("z", yaw).as_quat()
    pitch_quat = R.from_euler("y", pitch).as_quat()
    quaternion = (R.from_quat(yaw_quat) * R.from_quat(pitch_quat)).as_quat()
    return np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
