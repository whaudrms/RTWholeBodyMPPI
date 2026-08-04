"""JAX versions of the Warp cost equations used for numerical parity tests.

The runtime planner is implemented in :mod:`raw_warp_mppi`; these vectorized
functions provide an independent implementation of the same equations.
"""

import jax.numpy as jnp


def _world_to_local_velocity(quaternion, velocity):
    """Rotate world-frame vectors into local frames for [w, x, y, z] quats."""
    vector = quaternion[..., 1:]
    inverse_vector = -vector
    twice_cross = 2.0 * jnp.cross(inverse_vector, velocity)
    return (
        velocity
        + quaternion[..., :1] * twice_cross
        + jnp.cross(inverse_vector, twice_cross)
    )


def _robot_cost(
    qpos,
    qvel,
    actions,
    joints_ref,
    body_ref,
    q_diag,
    r_diag,
    joint_qpos_indices,
    joint_qvel_indices,
    robot_qpos_adr,
    robot_qvel_adr,
    robot_quaternion_only=False,
):
    """Return per-sample robot costs, matching the NumPy controller formula."""
    qpos = jnp.asarray(qpos)
    qvel = jnp.asarray(qvel)
    actions = jnp.asarray(actions)
    joints_ref = jnp.asarray(joints_ref)
    body_ref = jnp.asarray(body_ref)
    q_diag = jnp.asarray(q_diag)
    r_diag = jnp.asarray(r_diag)
    body_qpos = qpos[..., robot_qpos_adr : robot_qpos_adr + 7]
    body_qvel = qvel[..., robot_qvel_adr : robot_qvel_adr + 6]
    joint_qpos = jnp.take(qpos, joint_qpos_indices, axis=-1)
    joint_qvel = jnp.take(qvel, joint_qvel_indices, axis=-1)

    local_linear_velocity = _world_to_local_velocity(
        body_qpos[..., 3:7], body_qvel[..., :3]
    )
    body_qvel = body_qvel.at[..., :3].set(local_linear_velocity)
    robot_state = jnp.concatenate(
        (body_qpos, joint_qpos, body_qvel, joint_qvel), axis=-1
    )

    # joints_ref is [24, horizon] on the host.
    joints_ref_t = jnp.swapaxes(joints_ref, 0, 1)
    robot_ref = jnp.concatenate(
        (
            jnp.broadcast_to(body_ref[:7], (qpos.shape[1], 7)),
            joints_ref_t[:, :12],
            jnp.broadcast_to(body_ref[7:], (qpos.shape[1], body_ref.shape[0] - 7)),
            joints_ref_t[:, 12:],
        ),
        axis=-1,
    )

    error = robot_state - robot_ref[None, ...]
    quaternion_distance = 1.0 - jnp.abs(
        jnp.sum(robot_state[..., 3:7] * robot_ref[None, :, 3:7], axis=-1)
    )
    if robot_quaternion_only:
        error = error.at[..., 3].set(quaternion_distance)
        error = error.at[..., 4:7].set(0.0)
    else:
        error = error.at[..., 3:7].set(quaternion_distance[..., None])

    position_error = robot_state[..., :3] - robot_ref[None, :, :3]
    error = error.at[..., :3].set(0.0)
    position_cost = jnp.sum(
        jnp.abs(position_error * q_diag[None, None, :3]), axis=-1
    )

    torque_error = 50.0 * (actions - joint_qpos) - 3.0 * joint_qvel
    state_cost = jnp.sum(error * error * q_diag[None, None, :], axis=-1)
    torque_cost = jnp.sum(
        torque_error * torque_error * r_diag[None, None, :], axis=-1
    )
    return state_cost + torque_cost + position_cost


def locomotion_cost(
    qpos,
    qvel,
    actions,
    joints_ref,
    body_ref,
    q_diag,
    r_diag,
    joint_qpos_indices,
    joint_qvel_indices,
):
    """All locomotion trajectory cost terms, summed over the horizon."""
    step_cost = _robot_cost(
        qpos,
        qvel,
        actions,
        joints_ref,
        body_ref,
        q_diag,
        r_diag,
        joint_qpos_indices,
        joint_qvel_indices,
        robot_qpos_adr=0,
        robot_qvel_adr=0,
    )
    return jnp.sum(step_cost, axis=1)


def push_box_cost(
    qpos,
    qvel,
    actions,
    joints_ref,
    body_ref,
    box_ref,
    q_robot_diag,
    q_box_diag,
    r_diag,
    joint_qpos_indices,
    joint_qvel_indices,
):
    """All push-box robot and box cost terms, summed over the horizon."""
    qpos = jnp.asarray(qpos)
    box_ref = jnp.asarray(box_ref)
    q_box_diag = jnp.asarray(q_box_diag)
    robot_cost = _robot_cost(
        qpos,
        qvel,
        actions,
        joints_ref,
        body_ref,
        q_robot_diag,
        r_diag,
        joint_qpos_indices,
        joint_qvel_indices,
        robot_qpos_adr=7,
        robot_qvel_adr=6,
        robot_quaternion_only=True,
    )
    box_position_error = qpos[..., :3] - box_ref[None, None, :3]
    box_cost = jnp.sum(
        jnp.abs(box_position_error * q_box_diag[None, None, :3]), axis=-1
    )
    return jnp.sum(robot_cost + box_cost, axis=1)
