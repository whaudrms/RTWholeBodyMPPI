"""Generate phase-compatible height gait assets used by locomani.

The bank contains moving ``in_place``/``walk_fast`` references and a static
``stance_hold`` reference at three base heights.  All files use the common
32-row B2-Z1 layout expected by the whole-body MPPI controllers.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from convert_gait_reference import (
    DEFAULT_ARM_POSITION,
    DEFAULT_MODEL,
    GAITS_DIR,
    LEG_JOINT_NAMES,
    LEG_NAMES,
    _named_ids,
    assemble_output,
    convert,
    solve_leg_ik,
)


HEIGHT_LEVELS = (
    (0.543542, 0.10, "h_0p543542"),
    (0.45, 0.07, "h_0p450"),
    (0.35, 0.05, "h_0p350"),
)
SOURCE_GAITS = {
    "in_place": (
        GAITS_DIR
        / "FAST/go1/walking_gait_raibert_FAST_0_0_10cm_100hz.tsv"
    ),
    "walk_fast": (
        GAITS_DIR
        / "FAST/go1/walking_gait_raibert_FAST_0_1_10cm_100hz.tsv"
    ),
}
OUTPUT_ROOT = GAITS_DIR / "FAST/b2_height_conditioned"
PHASE_LENGTH = 100


def stance_hold_reference(
    base_height: float,
    *,
    model_path: Path = DEFAULT_MODEL,
    phase_length: int = PHASE_LENGTH,
) -> tuple[np.ndarray, float]:
    """Return a grounded four-foot static reference at ``base_height``."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    keyframe_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, "stand"
    )
    if keyframe_id < 0:
        raise ValueError("Model does not contain the stand keyframe")
    model.key_qpos[keyframe_id, 2] = float(base_height)

    joint_ids = _named_ids(
        model, mujoco.mjtObj.mjOBJ_JOINT, LEG_JOINT_NAMES
    )
    foot_ids = _named_ids(
        model, mujoco.mjtObj.mjOBJ_SITE, LEG_NAMES
    )
    qpos_addresses = model.jnt_qposadr[joint_ids]
    initial_leg_q = model.key_qpos[keyframe_id, qpos_addresses].copy()

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    mujoco.mj_forward(model, data)
    targets = data.site_xpos[foot_ids].copy()
    targets[:, 2] = model.site_size[foot_ids, 0]

    leg_q, residual = solve_leg_ik(
        model,
        initial_leg_q[:, None],
        targets[None, :, :],
        keyframe_id,
        joint_ids,
        foot_ids,
        damping=1e-4,
        tolerance=1e-7,
        max_iterations=100,
    )
    repeated_q = np.repeat(leg_q, phase_length, axis=1)
    reference = assemble_output(
        repeated_q,
        np.zeros_like(repeated_q),
        DEFAULT_ARM_POSITION,
    )
    return reference, residual


def write_reference(path: Path, reference: np.ndarray, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        reference,
        delimiter="\t",
        fmt="%.17g",
        header=header,
    )


def generate() -> list[Path]:
    outputs: list[Path] = []
    for base_height, step_height, directory in HEIGHT_LEVELS:
        output_dir = OUTPUT_ROOT / directory
        for name, source in SOURCE_GAITS.items():
            output = output_dir / f"{name}.tsv"
            convert(
                source,
                output,
                mode="transfer",
                model_path=DEFAULT_MODEL,
                keyframe="stand",
                base_height=base_height,
                step_height=step_height,
                rate=100.0,
            )
            outputs.append(output)
            print(f"wrote: {output}")

        stance, residual = stance_hold_reference(base_height)
        stance_output = output_dir / "stance_hold.tsv"
        write_reference(
            stance_output,
            stance,
            (
                "B2-Z1 four-foot stance hold; "
                f"base height={base_height:.6g}m; "
                f"max IK residual={residual:.6g}m; "
                "rows: 12 leg q, 4 arm q, 12 leg dq, 4 arm dq"
            ),
        )
        outputs.append(stance_output)
        print(f"wrote: {stance_output}")
    return outputs


if __name__ == "__main__":
    generated = generate()
    print(f"generated {len(generated)} height-conditioned gait files")
