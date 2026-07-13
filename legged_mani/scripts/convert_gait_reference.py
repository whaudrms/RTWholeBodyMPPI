"""Convert the original 24-row Go1 in-place gait for B2-Z1."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from legged_mani.mani_mppi.interface.environment import MODEL_PATH


DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "legged_mppi"
    / "whole_body_mppi"
    / "control"
    / "gait_scheduler"
    / "gaits"
    / "FAST"
    / "walking_gait_raibert_FAST_0_0_10cm_100hz.tsv"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "whole_body_mppi"
    / "control"
    / "gait_scheduler"
    / "gaits"
    / "FAST"
    / "b2_z1_in_place_FAST_0_0_10cm_100hz.tsv"
)


def convert(source: Path, output: Path) -> np.ndarray:
    legacy = np.loadtxt(source, delimiter="\t")
    if legacy.ndim != 2 or legacy.shape[0] != 24:
        raise ValueError(f"Expected legacy gait shape (24, N), got {legacy.shape}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    stand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    stand_control = model.key_ctrl[stand_id].copy()

    # Preserve each Go1 leg trajectory's variation while moving its mean to
    # the B2 stand keyframe. Joint ordering is FR, FL, RR, RL × hip/thigh/calf.
    leg_position = legacy[:12].copy()
    leg_position += stand_control[:12, None] - leg_position.mean(axis=1, keepdims=True)
    leg_low = model.actuator_ctrlrange[:12, 0, None]
    leg_high = model.actuator_ctrlrange[:12, 1, None]
    if np.any(leg_position < leg_low) or np.any(leg_position > leg_high):
        raise ValueError("Mean-aligned leg gait exceeds B2 actuator limits")

    arm_position = np.repeat(stand_control[12:, None], legacy.shape[1], axis=1)
    leg_velocity = legacy[12:24].copy()
    arm_velocity = np.zeros((4, legacy.shape[1]))
    converted = np.vstack(
        (leg_position, arm_position, leg_velocity, arm_velocity)
    )
    if converted.shape != (32, legacy.shape[1]) or not np.isfinite(converted).all():
        raise ValueError(f"Invalid converted gait shape/content: {converted.shape}")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output,
        converted,
        delimiter="\t",
        fmt="%.10f",
        header=(
            "B2-Z1 32-row gait adapted from " + source.name
            + "; rows: 16 q then 16 dq; Go1 leg means aligned to B2 stand"
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
