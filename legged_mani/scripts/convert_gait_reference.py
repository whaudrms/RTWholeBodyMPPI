"""Retarget a 24-row Go1 gait reference to the B2-Z1 model.

The source file contains Go1 leg joint positions and velocities in the order
``FR, FL, RR, RL x hip, thigh, calf``.  Joint offsets cannot be copied between
robots directly: the same angles produce different foot positions because the
hip offsets and leg lengths differ.  This converter therefore performs:

1. Go1 joint angles -> Go1 foot trajectories using the Go1 leg geometry.
2. Go1 foot trajectory displacement -> B2 stand-foot displacement.
3. A bounded numerical IK solve for every B2 foot sample.
4. Periodic differentiation of the resulting B2 positions for ``dq``.

The generated file remains the scheduler's 32-row format:
``16 B2 joint positions, followed by 16 B2 joint velocities``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mujoco
import numpy as np

# Allow direct execution from the scripts directory, matching the other
# command-line entry points in this workspace.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legged_mani.mani_mppi.interface.environment import MODEL_PATH


DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "mani_mppi"
    / "control"
    / "gait_scheduler"
    / "gaits"
    / "FAST"
    / "walking_gait_raibert_FAST_0_0_10cm_100hz.tsv"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "mani_mppi"
    / "control"
    / "gait_scheduler"
    / "gaits"
    / "FAST"
    / "b2_z1_in_place_FAST_0_0_10cm_100hz.tsv"
)

SOURCE_DT = 0.01
GO1_THIGH_LENGTH = 0.213
GO1_CALF_LENGTH = 0.213
B2_FOOT_RADIUS = 0.032

# The XML and the controller use this order.  The signs describe the Go1
# hip-frame locations in the base frame: front/rear and right/left.
LEG_NAMES = ("FR", "FL", "RR", "RL")
FRONT_REAR_SIGN = np.array((1.0, 1.0, -1.0, -1.0))
RIGHT_LEFT_SIGN = np.array((-1.0, 1.0, -1.0, 1.0))
GO1_HIP_ORIGIN_X = 0.1881
GO1_HIP_ORIGIN_Y = 0.04675
GO1_HFE_OFFSET = 0.08
B2_SITE_NAMES = tuple(LEG_NAMES)


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def _rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def go1_foot_positions(leg_position: np.ndarray) -> np.ndarray:
    """Return Go1 foot centers in the base frame, shaped ``(4, 3, N)``."""
    if leg_position.shape[0] != 12:
        raise ValueError(f"Expected 12 Go1 leg position rows, got {leg_position.shape}")

    n_steps = leg_position.shape[1]
    q = leg_position.reshape(4, 3, n_steps)
    feet = np.empty((4, 3, n_steps), dtype=float)

    for leg in range(4):
        front_rear = FRONT_REAR_SIGN[leg]
        right_left = RIGHT_LEFT_SIGN[leg]
        hip_origin = np.array(
            (front_rear * GO1_HIP_ORIGIN_X, right_left * GO1_HIP_ORIGIN_Y, 0.0)
        )
        for step in range(n_steps):
            hip_rotation = _rot_x(q[leg, 0, step])
            thigh_rotation = _rot_y(q[leg, 1, step])
            calf_rotation = _rot_y(q[leg, 2, step])
            feet[leg, :, step] = (
                hip_origin
                + hip_rotation @ np.array((0.0, right_left * GO1_HFE_OFFSET, 0.0))
                + hip_rotation @ thigh_rotation @ np.array(
                    (0.0, 0.0, -GO1_THIGH_LENGTH)
                )
                + hip_rotation
                @ thigh_rotation
                @ calf_rotation
                @ np.array((0.0, 0.0, -GO1_CALF_LENGTH))
            )
    return feet


def _joint_indices(model: mujoco.MjModel, leg: str) -> tuple[np.ndarray, np.ndarray]:
    qpos_indices = []
    dof_indices = []
    for joint_suffix in ("hip_joint", "thigh_joint", "calf_joint"):
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{joint_suffix}"
        )
        if joint_id < 0:
            raise ValueError(f"Missing B2 joint {leg}_{joint_suffix}")
        qpos_indices.append(model.jnt_qposadr[joint_id])
        dof_indices.append(model.jnt_dofadr[joint_id])
    return np.asarray(qpos_indices), np.asarray(dof_indices)


def _b2_stand_feet(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Return stand foot positions and the joint addresses used by the IK."""
    stand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if stand_id < 0:
        raise ValueError("B2 model does not contain a 'stand' keyframe")
    mujoco.mj_resetDataKeyframe(model, data, stand_id)
    mujoco.mj_forward(model, data)

    stand_feet = np.array([data.site(name).xpos.copy() for name in B2_SITE_NAMES])
    qpos_indices = []
    dof_indices = []
    for leg in LEG_NAMES:
        qpos, dof = _joint_indices(model, leg)
        qpos_indices.append(qpos)
        dof_indices.append(dof)
    return stand_feet, data.qpos.copy(), qpos_indices, dof_indices


def _solve_leg_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    target: np.ndarray,
    qpos_indices: np.ndarray,
    dof_indices: np.ndarray,
    max_iterations: int = 100,
) -> tuple[np.ndarray, float]:
    """Solve one 3-DoF leg position with bounded damped least squares."""
    jacobian = np.zeros((3, model.nv), dtype=float)
    for _ in range(max_iterations):
        mujoco.mj_forward(model, data)
        error = target - data.site(site_id).xpos
        if np.linalg.norm(error) < 1e-8:
            break

        mujoco.mj_jacSite(model, data, jacobian, None, site_id)
        jac = jacobian[:, dof_indices]
        damping = 1e-8
        delta = jac.T @ np.linalg.solve(
            jac @ jac.T + damping * np.eye(3), error
        )
        data.qpos[qpos_indices] += np.clip(delta, -0.15, 0.15)
        # The three joints have scalar hinge ranges.  The addresses are passed
        # in leg order, so clipping against their corresponding joint ranges is
        # independent of the leg name.
        for qpos_index, dof_index in zip(qpos_indices, dof_indices):
            joint_id = int(np.flatnonzero(model.jnt_dofadr == dof_index)[0])
            data.qpos[qpos_index] = np.clip(
                data.qpos[qpos_index], model.jnt_range[joint_id, 0], model.jnt_range[joint_id, 1]
            )

    mujoco.mj_forward(model, data)
    residual = float(np.linalg.norm(target - data.site(site_id).xpos))
    return data.qpos[qpos_indices].copy(), residual


def _periodic_derivative(position: np.ndarray, dt: float) -> np.ndarray:
    """Differentiate a cyclic reference without an end-point spike."""
    return (np.roll(position, -1, axis=1) - np.roll(position, 1, axis=1)) / (2.0 * dt)


def convert(source: Path, output: Path) -> np.ndarray:
    legacy = np.loadtxt(source, delimiter="\t")
    if legacy.ndim != 2 or legacy.shape[0] != 24:
        raise ValueError(f"Expected Go1 gait shape (24, N), got {legacy.shape}")
    if legacy.shape[1] < 3 or not np.isfinite(legacy).all():
        raise ValueError("Go1 gait must contain at least three finite samples")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    stand_feet, stand_qpos, qpos_indices, dof_indices = _b2_stand_feet(model, data)

    # Retarget the actual Cartesian foot displacement, rather than copying
    # Go1 joint-angle amplitudes.  The source is an in-place gait, so this
    # preserves its vertical swing amplitude and any x/y displacement if a
    # walking reference is passed later.
    source_feet = go1_foot_positions(legacy[:12])
    source_center = source_feet.mean(axis=2, keepdims=True)
    target_feet = stand_feet[:, :, None] + source_feet - source_center

    # Put the lowest point of every foot exactly at the ground-contact center.
    # The B2 foot geom is a sphere of radius 32 mm; consequently its center
    # must not be below this value when the base is held at the stand pose.
    target_feet[:, 2, :] += (
        B2_FOOT_RADIUS - target_feet[:, 2, :].min(axis=1, keepdims=True)
    )

    leg_position = np.empty((12, legacy.shape[1]), dtype=float)
    residuals = []
    site_ids = [model.site(name).id for name in B2_SITE_NAMES]
    for step in range(legacy.shape[1]):
        # Start every sample from the previous solution for a continuous IK
        # branch.  The first sample starts at the B2 stand keyframe.
        if step == 0:
            data.qpos[:] = stand_qpos
        for leg in range(4):
            q, residual = _solve_leg_ik(
                model,
                data,
                site_ids[leg],
                target_feet[leg, :, step],
                qpos_indices[leg],
                dof_indices[leg],
            )
            leg_position[3 * leg : 3 * leg + 3, step] = q
            residuals.append(residual)

    # Keep the fixed Z1 arm at its stand configuration and derive velocities
    # from the B2 positions that were actually produced by IK.
    stand_control = model.key_ctrl[model.key("stand").id].copy()
    arm_position = np.repeat(stand_control[12:, None], legacy.shape[1], axis=1)
    leg_velocity = _periodic_derivative(leg_position, SOURCE_DT)
    arm_velocity = np.zeros((4, legacy.shape[1]))
    converted = np.vstack((leg_position, arm_position, leg_velocity, arm_velocity))

    if converted.shape != (32, legacy.shape[1]) or not np.isfinite(converted).all():
        raise ValueError(f"Invalid converted gait shape/content: {converted.shape}")

    limits = model.actuator_ctrlrange[:12]
    if np.any(leg_position < limits[:, 0, None] - 1e-9) or np.any(
        leg_position > limits[:, 1, None] + 1e-9
    ):
        raise ValueError("IK gait exceeds B2 actuator position limits")
    if max(residuals) > 2e-5:
        raise ValueError(f"B2 foot IK did not converge; max residual={max(residuals):.3e} m")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output,
        converted,
        delimiter="\t",
        fmt="%.10f",
        header=(
            "B2-Z1 32-row gait retargeted through Go1/B2 foot kinematics from "
            + source.name
            + "; rows: 16 q then 16 dq; periodic dq at 100 Hz"
        ),
    )
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    converted = convert(args.source.resolve(), args.output.resolve())
    print(f"wrote: {args.output.resolve()}")
    print(f"shape: {converted.shape}")
    print(f"leg q range: [{converted[:12].min():.4f}, {converted[:12].max():.4f}]")
    print(f"leg dq abs max: {np.abs(converted[16:28]).max():.4f}")


if __name__ == "__main__":
    main()
