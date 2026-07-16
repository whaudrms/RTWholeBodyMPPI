"""Convert legacy Go1 gait TSVs into B2-Z1 scheduler references.

Two conversion modes are available:

``padding``
    Preserve all 12 Go1 leg position/velocity rows exactly and append four
    fixed Z1 arm positions plus four zero arm velocities.  This is the legacy
    behavior.

``transfer``
    Evaluate the source leg angles on the B2 model, preserve the resulting
    foot-path x/y coordinates and normalized swing shape, place stance on the
    ground, set the requested swing height, solve B2 leg IK, and regenerate
    periodic joint velocities.  Arm rows are padded as in ``padding`` mode.

Input files must have 24 rows: 12 leg q followed by 12 leg dq.  Output files
have 32 rows: 12 leg q, 4 arm q, 12 leg dq, and 4 arm dq.

Examples
--------
    python convert_gait_reference.py --mode padding
    python convert_gait_reference.py --mode transfer
    python convert_gait_reference.py --mode transfer --step-height 0.10 --rate 100
    python convert_gait_reference.py --mode padding --all
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
GAITS_DIR = PACKAGE_ROOT / "mani_mppi/control/gait_scheduler/gaits"
DEFAULT_SOURCE = (
    GAITS_DIR
    / "FAST/go1/walking_gait_raibert_FAST_0_0_10cm_100hz.tsv"
)
DEFAULT_MODEL = PACKAGE_ROOT / "mani_mppi/models/b2_z1_4dof.xml"
DEFAULT_ARM_POSITION = np.array([0.0, 1.0, -0.6, 0.0], dtype=float)

MODES = ("padding", "transfer")
LEG_NAMES = ("FR", "FL", "RR", "RL")
LEG_JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in LEG_NAMES
    for joint in ("hip", "thigh", "calf")
)


def default_output(source: Path, mode: str = "padding") -> Path:
    """Return a non-ambiguous output path for one source gait."""
    source = source.resolve()
    if mode not in MODES:
        raise ValueError(f"Unsupported conversion mode: {mode}")
    if source.parent.name != "go1" or source.parent.parent.parent != GAITS_DIR:
        raise ValueError(
            "When --output is omitted, source must be inside "
            "gaits/<speed>/go1/<file>.tsv"
        )

    if mode == "padding":
        return source.parent.parent / "b2_z1" / source.name
    return source.parent.parent / "b2_retargeted" / source.name


def load_source(source: Path) -> np.ndarray:
    source_data = np.loadtxt(source, delimiter="\t", comments="#")
    if source_data.ndim != 2 or source_data.shape[0] != 24:
        raise ValueError(
            f"Expected a 24-row Go1 gait [12 q + 12 dq], got {source_data.shape}"
        )
    if source_data.shape[1] < 2 or not np.isfinite(source_data).all():
        raise ValueError("Source gait must contain at least two finite samples")
    return source_data


def validate_arm_position(arm_position: np.ndarray) -> np.ndarray:
    arm_position = np.asarray(arm_position, dtype=float)
    if arm_position.shape != (4,) or not np.isfinite(arm_position).all():
        raise ValueError("arm_position must contain four finite values")
    return arm_position


def assemble_output(
    leg_q: np.ndarray,
    leg_dq: np.ndarray,
    arm_position: np.ndarray,
) -> np.ndarray:
    if leg_q.shape != leg_dq.shape or leg_q.ndim != 2 or leg_q.shape[0] != 12:
        raise ValueError(
            f"Expected matching (12, N) leg q/dq arrays, got "
            f"{leg_q.shape} and {leg_dq.shape}"
        )
    arm_position = validate_arm_position(arm_position)
    n_samples = leg_q.shape[1]
    arm_q = np.repeat(arm_position[:, None], n_samples, axis=1)
    arm_dq = np.zeros((4, n_samples), dtype=float)
    output = np.vstack((leg_q, arm_q, leg_dq, arm_dq))
    if output.shape != (32, n_samples) or not np.isfinite(output).all():
        raise RuntimeError(f"Unexpected converted gait shape/content: {output.shape}")
    return output


def convert_padding(
    source_data: np.ndarray,
    arm_position: np.ndarray = DEFAULT_ARM_POSITION,
) -> np.ndarray:
    """Apply legacy dimension padding without changing source leg rows."""
    return assemble_output(source_data[:12], source_data[12:24], arm_position)


def value_from_filename(path: Path, suffix: str, scale: float = 1.0) -> float:
    match = re.search(rf"(?:^|_)(\d+(?:\.\d+)?){re.escape(suffix)}(?:_|\.)", path.name)
    if match is None:
        raise ValueError(
            f"Could not infer {suffix} value from {path.name}; provide it explicitly"
        )
    return float(match.group(1)) * scale


def periodic_velocity(positions: np.ndarray, rate: float) -> np.ndarray:
    """Differentiate hinge positions with a periodic central difference."""
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("Sample rate must be finite and positive")
    return 0.5 * rate * (
        np.roll(positions, -1, axis=1) - np.roll(positions, 1, axis=1)
    )


def _named_ids(model, object_type, names: tuple[str, ...]) -> np.ndarray:
    import mujoco

    ids = np.asarray(
        [mujoco.mj_name2id(model, object_type, name) for name in names], dtype=int
    )
    missing = [name for name, object_id in zip(names, ids) if object_id < 0]
    if missing:
        raise ValueError(f"Model is missing required names: {missing}")
    return ids


def source_foot_trajectories(
    model,
    source_leg_q: np.ndarray,
    keyframe_id: int,
    joint_ids: np.ndarray,
    foot_ids: np.ndarray,
) -> np.ndarray:
    """Evaluate source joint angles with the B2 forward kinematics."""
    import mujoco

    data = mujoco.MjData(model)
    qpos_addresses = model.jnt_qposadr[joint_ids]
    lower = model.jnt_range[joint_ids, 0, None]
    upper = model.jnt_range[joint_ids, 1, None]
    if np.any(source_leg_q < lower) or np.any(source_leg_q > upper):
        raise ValueError("Source leg positions violate B2 joint limits")

    trajectories = np.empty((source_leg_q.shape[1], len(foot_ids), 3), dtype=float)
    for sample in range(source_leg_q.shape[1]):
        mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
        data.qpos[qpos_addresses] = source_leg_q[:, sample]
        mujoco.mj_forward(model, data)
        trajectories[sample] = data.site_xpos[foot_ids]
    return trajectories


def grounded_targets(
    model,
    foot_ids: np.ndarray,
    source_trajectories: np.ndarray,
    nominal_foot_positions: np.ndarray,
    step_height: float,
) -> np.ndarray:
    """Center x/y on the B2 stand feet and normalize swing above the ground."""
    if not np.isfinite(step_height) or step_height <= 0.0:
        raise ValueError("step_height must be finite and positive")
    if nominal_foot_positions.shape != (len(foot_ids), 3):
        raise ValueError(
            "nominal_foot_positions must contain one xyz position per foot"
        )

    targets = source_trajectories.copy()
    source_xy_centers = source_trajectories[:, :, :2].mean(axis=0)
    targets[:, :, :2] += (
        nominal_foot_positions[:, :2] - source_xy_centers
    )[None, :, :]
    for foot_index, site_id in enumerate(foot_ids):
        source_z = source_trajectories[:, foot_index, 2]
        source_span = float(np.ptp(source_z))
        if source_span <= 1e-9:
            raise ValueError(
                f"Foot {LEG_NAMES[foot_index]} has no measurable swing-height variation"
            )
        normalized_height = (source_z - source_z.min()) / source_span
        foot_radius = float(model.site_size[site_id, 0])
        targets[:, foot_index, 2] = foot_radius + step_height * normalized_height
    return targets


def solve_leg_ik(
    model,
    source_leg_q: np.ndarray,
    targets: np.ndarray,
    keyframe_id: int,
    joint_ids: np.ndarray,
    foot_ids: np.ndarray,
    damping: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, float]:
    """Solve independent damped-least-squares IK for all B2 legs/samples."""
    import mujoco

    if damping <= 0.0 or tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("Invalid IK damping, tolerance, or iteration count")

    data = mujoco.MjData(model)
    qpos_addresses = model.jnt_qposadr[joint_ids]
    dof_addresses = model.jnt_dofadr[joint_ids]
    lower = model.jnt_range[joint_ids, 0]
    upper = model.jnt_range[joint_ids, 1]
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    transferred = np.empty_like(source_leg_q)
    max_residual = 0.0

    for sample in range(source_leg_q.shape[1]):
        # The source sample selects the continuous IK branch. Each B2 leg is
        # independent once the floating base is fixed at the keyframe.
        mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
        data.qpos[qpos_addresses] = source_leg_q[:, sample]

        for leg_index, site_id in enumerate(foot_ids):
            leg_slice = slice(3 * leg_index, 3 * leg_index + 3)
            leg_qpos = qpos_addresses[leg_slice]
            leg_dofs = dof_addresses[leg_slice]
            leg_lower = lower[leg_slice]
            leg_upper = upper[leg_slice]

            for _ in range(max_iterations):
                mujoco.mj_forward(model, data)
                error = targets[sample, leg_index] - data.site_xpos[site_id]
                if np.linalg.norm(error) <= tolerance:
                    break
                mujoco.mj_jacSite(model, data, jacp, jacr, int(site_id))
                jacobian = jacp[:, leg_dofs]
                system = jacobian @ jacobian.T + damping**2 * np.eye(3)
                delta = jacobian.T @ np.linalg.solve(system, error)
                delta_norm = np.linalg.norm(delta)
                if delta_norm > 0.1:
                    delta *= 0.1 / delta_norm
                data.qpos[leg_qpos] = np.clip(
                    data.qpos[leg_qpos] + delta, leg_lower, leg_upper
                )

            mujoco.mj_forward(model, data)
            residual = float(
                np.linalg.norm(targets[sample, leg_index] - data.site_xpos[site_id])
            )
            max_residual = max(max_residual, residual)
            if residual > max(1e-4, 10.0 * tolerance):
                raise RuntimeError(
                    f"IK failed at sample {sample}, foot {LEG_NAMES[leg_index]}: "
                    f"residual={residual:.6g} m"
                )

        transferred[:, sample] = data.qpos[qpos_addresses]

    return transferred, max_residual


def convert_transfer(
    source_data: np.ndarray,
    source: Path,
    model_path: Path = DEFAULT_MODEL,
    keyframe: str = "stand",
    step_height: float | None = None,
    rate: float | None = None,
    arm_position: np.ndarray = DEFAULT_ARM_POSITION,
    ik_damping: float = 1e-4,
    ik_tolerance: float = 1e-7,
    ik_max_iterations: int = 100,
) -> tuple[np.ndarray, float, float, float]:
    """Retarget one source gait and return data plus transfer diagnostics."""
    import mujoco

    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    step_height = (
        value_from_filename(source, "cm", 0.01)
        if step_height is None
        else float(step_height)
    )
    rate = value_from_filename(source, "hz") if rate is None else float(rate)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    if keyframe_id < 0:
        raise ValueError(f"Unknown model keyframe: {keyframe}")
    joint_ids = _named_ids(model, mujoco.mjtObj.mjOBJ_JOINT, LEG_JOINT_NAMES)
    foot_ids = _named_ids(model, mujoco.mjtObj.mjOBJ_SITE, LEG_NAMES)

    source_trajectories = source_foot_trajectories(
        model, source_data[:12], keyframe_id, joint_ids, foot_ids
    )
    nominal_data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, nominal_data, keyframe_id)
    mujoco.mj_forward(model, nominal_data)
    nominal_foot_positions = nominal_data.site_xpos[foot_ids].copy()
    targets = grounded_targets(
        model,
        foot_ids,
        source_trajectories,
        nominal_foot_positions,
        step_height,
    )
    transferred_q, max_residual = solve_leg_ik(
        model,
        source_data[:12],
        targets,
        keyframe_id,
        joint_ids,
        foot_ids,
        damping=ik_damping,
        tolerance=ik_tolerance,
        max_iterations=ik_max_iterations,
    )
    transferred_dq = periodic_velocity(transferred_q, rate)
    output = assemble_output(transferred_q, transferred_dq, arm_position)
    original_height = float(np.max(source_trajectories[:, :, 2], axis=0).mean()
                            - np.min(source_trajectories[:, :, 2], axis=0).mean())
    return output, max_residual, original_height, step_height


def convert(
    source: Path,
    output: Path | None = None,
    arm_position: np.ndarray = DEFAULT_ARM_POSITION,
    mode: str = "padding",
    model_path: Path = DEFAULT_MODEL,
    keyframe: str = "stand",
    step_height: float | None = None,
    rate: float | None = None,
    ik_damping: float = 1e-4,
    ik_tolerance: float = 1e-7,
    ik_max_iterations: int = 100,
) -> np.ndarray:
    """Convert one gait using either exact padding or kinematic transfer."""
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if mode not in MODES:
        raise ValueError(f"Unsupported conversion mode: {mode}")
    output = default_output(source, mode) if output is None else output.resolve()
    source_data = load_source(source)
    arm_position = validate_arm_position(arm_position)

    if mode == "padding":
        output_data = convert_padding(source_data, arm_position)
        header = (
            "B2-Z1 dimension-padded gait; source leg rows preserved; "
            "rows: 12 leg q, 4 arm q, 12 leg dq, 4 arm dq"
        )
    else:
        output_data, residual, source_height, target_height = convert_transfer(
            source_data,
            source,
            model_path=model_path,
            keyframe=keyframe,
            step_height=step_height,
            rate=rate,
            arm_position=arm_position,
            ik_damping=ik_damping,
            ik_tolerance=ik_tolerance,
            ik_max_iterations=ik_max_iterations,
        )
        header = (
            "B2-Z1 kinematically transferred gait; foot x/y centered on the "
            "B2 stand keyframe, stance grounded, and swing "
            f"height={target_height:.6g}m; source B2-FK height={source_height:.6g}m; "
            f"max IK residual={residual:.6g}m; rows: 12 leg q, 4 arm q, "
            "12 leg dq, 4 arm dq"
        )
        print(
            f"transfer: source swing={source_height:.4f} m, "
            f"target swing={target_height:.4f} m, max IK residual={residual:.3g} m"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output,
        output_data,
        delimiter="\t",
        fmt="%.17g",
        header=header,
    )
    return output_data


def convert_all(
    mode: str = "padding",
    model_path: Path = DEFAULT_MODEL,
    keyframe: str = "stand",
    step_height: float | None = None,
    rate: float | None = None,
    arm_position: np.ndarray = DEFAULT_ARM_POSITION,
    ik_damping: float = 1e-4,
    ik_tolerance: float = 1e-7,
    ik_max_iterations: int = 100,
) -> list[Path]:
    """Convert every source TSV with the selected mode."""
    outputs = []
    for source in sorted(GAITS_DIR.glob("*/go1/*.tsv")):
        output = default_output(source, mode)
        convert(
            source,
            output,
            arm_position=arm_position,
            mode=mode,
            model_path=model_path,
            keyframe=keyframe,
            step_height=step_height,
            rate=rate,
            ik_damping=ik_damping,
            ik_tolerance=ik_tolerance,
            ik_max_iterations=ik_max_iterations,
        )
        outputs.append(output)
        print(f"wrote: {output}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="padding")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="One 24-row source TSV below gaits/<speed>/go1",
    )
    parser.add_argument("--output", type=Path, help="Explicit output path")
    parser.add_argument(
        "--all", action="store_true", help="Convert every TSV in gaits/*/go1/"
    )
    parser.add_argument(
        "--arm-position",
        type=float,
        nargs=4,
        default=DEFAULT_ARM_POSITION,
        metavar=("J1", "J2", "J3", "J4"),
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--keyframe", default="stand")
    parser.add_argument(
        "--step-height",
        type=float,
        help="Transfer swing height in meters; inferred from NNcm filename if omitted",
    )
    parser.add_argument(
        "--rate",
        type=float,
        help="Transfer sample rate in Hz; inferred from NNHz filename if omitted",
    )
    parser.add_argument("--ik-damping", type=float, default=1e-4)
    parser.add_argument("--ik-tolerance", type=float, default=1e-7)
    parser.add_argument("--ik-max-iterations", type=int, default=100)
    args = parser.parse_args()
    if args.all and args.output is not None:
        parser.error("--all cannot be combined with --output")
    return args


def main() -> None:
    args = parse_args()
    kwargs = dict(
        arm_position=np.asarray(args.arm_position, dtype=float),
        mode=args.mode,
        model_path=args.model,
        keyframe=args.keyframe,
        step_height=args.step_height,
        rate=args.rate,
        ik_damping=args.ik_damping,
        ik_tolerance=args.ik_tolerance,
        ik_max_iterations=args.ik_max_iterations,
    )
    if args.all:
        outputs = convert_all(**kwargs)
        print(f"converted {len(outputs)} gait files in {args.mode} mode")
        return

    converted = convert(args.source, args.output, **kwargs)
    output = (
        default_output(args.source, args.mode)
        if args.output is None
        else args.output.resolve()
    )
    print(f"mode: {args.mode}")
    print(f"wrote: {output}")
    print(f"shape: {converted.shape}")
    print(f"leg q range: [{converted[:12].min():.4f}, {converted[:12].max():.4f}]")
    print(f"leg dq abs max: {np.abs(converted[16:28]).max():.4f}")


if __name__ == "__main__":
    main()
