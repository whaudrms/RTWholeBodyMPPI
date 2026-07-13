"""Named slices for ``concat(qpos, qvel)`` B2-Z1 observations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class B2Z1StateLayout:
    base_position: slice = slice(0, 3)
    base_quaternion: slice = slice(3, 7)
    leg_position: slice = slice(7, 19)
    arm_position: slice = slice(19, 23)
    base_linear_velocity: slice = slice(23, 26)
    base_angular_velocity: slice = slice(26, 29)
    leg_velocity: slice = slice(29, 41)
    arm_velocity: slice = slice(41, 45)
    joint_position: slice = slice(7, 23)
    joint_velocity: slice = slice(29, 45)
    state_dim: int = 45
