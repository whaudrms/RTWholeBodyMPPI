"""Replay a gait reference and draw foot trajectories in MuJoCo.

The gait is applied kinematically: the floating base stays at a named model
keyframe while the 16 actuated joint positions are copied from a TSV or from
Locomani's height-conditioned gait interpolation. This makes the viewer useful
for checking reference data without controller or contact-dynamics effects.

Examples
--------
Replay the default TSV::

    python3 legged_mani/tests/visualize_gait_trajectory.py

Replay the exact reference returned by Locomani's height interpolation::

    python3 legged_mani/tests/visualize_gait_trajectory.py \
        --gait-name walk_fast --height 0.40 --height-rate -0.10

Validate without opening a viewer::

    python3 legged_mani/tests/visualize_gait_trajectory.py \
        --gait-name stance_hold --height 0.40 --headless
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
import sys

import mujoco
import mujoco.viewer
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from mani_mppi.control.controllers.whole_body_arm_controller import (  # noqa: E402
    HEIGHT_GAIT_PATHS,
)
from mani_mppi.control.gait_scheduler.height_conditioned_scheduler import (  # noqa: E402
    HeightConditionedGaitScheduler,
)


DEFAULT_MODEL = PACKAGE_ROOT / "mani_mppi/models/go2_so_arm_base.xml"
DEFAULT_GAIT = (
    PACKAGE_ROOT
    / "mani_mppi/control/gait_scheduler/gaits/"
    "FAST/go2_so_arm/walking_gait_raibert_FAST_0_0_10cm_100hz.tsv"
)

JOINT_NAMES = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
)
FOOT_NAMES = ("FR", "FL", "RR", "RL")
FOOT_COLORS = np.asarray(
    (
        (0.95, 0.20, 0.15, 0.85),  # FR: red
        (0.20, 0.85, 0.25, 0.85),  # FL: green
        (0.20, 0.45, 1.00, 0.85),  # RR: blue
        (1.00, 0.75, 0.10, 0.85),  # RL: yellow
    ),
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    gait_source = parser.add_mutually_exclusive_group()
    gait_source.add_argument(
        "--gait",
        type=Path,
        help=f"Replay one gait TSV directly (default: {DEFAULT_GAIT})",
    )
    gait_source.add_argument(
        "--gait-name",
        choices=sorted(HEIGHT_GAIT_PATHS),
        help="Replay a height-interpolated Locomani gait bank",
    )
    parser.add_argument(
        "--height",
        type=float,
        help="Interpolation/base height in meters (default: 0.40)",
    )
    parser.add_argument(
        "--height-rate",
        type=float,
        default=0.0,
        help="Height morphing rate in m/s (default: 0)",
    )
    parser.add_argument(
        "--phase-time",
        type=int,
        default=0,
        help="First gait phase index for interpolation mode (default: 0)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        help=(
            "Playback rate in Hz "
            "(default: 80 for direct TSV, 100 for interpolation)"
        ),
    )
    parser.add_argument(
        "--keyframe",
        default="stand",
        help="Base pose keyframe held during kinematic playback (default: stand)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Validate and print trajectory statistics without opening a viewer",
    )
    parser.add_argument(
        "--save-trajectory",
        type=Path,
        help="Optionally save time and FR/FL/RR/RL xyz trajectories as a TSV",
    )
    parser.add_argument(
        "--hide-ui",
        action="store_true",
        help="Hide the MuJoCo left and right control panels",
    )
    return parser.parse_args()


def load_gait(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    gait = np.loadtxt(path, delimiter="\t", comments="#")
    if gait.ndim != 2 or gait.shape[0] != 32 or gait.shape[1] < 2:
        raise ValueError(
            "Expected gait shape (32, N) with N >= 2: "
            "[16 joint positions; 16 joint velocities], "
            f"got {gait.shape}"
        )
    if not np.isfinite(gait).all():
        raise ValueError("Gait contains NaN or infinite values")
    return gait


def interpolate_gait_reference(
    gait_name: str,
    height: float,
    height_rate: float,
    phase_time: int,
) -> tuple[np.ndarray, float, str]:
    """Return the exact height-conditioned reference used by Locomani."""
    scheduler = HeightConditionedGaitScheduler(
        HEIGHT_GAIT_PATHS[gait_name],
        name=gait_name,
        phase_time=phase_time,
    )
    clamped_height = float(
        np.clip(height, scheduler.min_height, scheduler.max_height)
    )
    gait = scheduler.get_reference(
        clamped_height,
        height_rate=height_rate,
        horizon=scheduler.phase_length,
    )
    lower, upper, alpha, _span = scheduler._bracket(clamped_height)
    lower_height = scheduler.heights[lower]
    upper_height = scheduler.heights[upper]
    lower_reference = scheduler.gaits[lower][:, scheduler.indices]
    upper_reference = scheduler.gaits[upper][:, scheduler.indices]
    linear_reference = (
        (1.0 - alpha) * lower_reference + alpha * upper_reference
    )
    velocity_morph = gait[scheduler.position_dim:] - linear_reference[
        scheduler.position_dim:
    ]
    description = (
        f"interpolated {gait_name}: height={clamped_height:.4f} m, "
        f"height_rate={height_rate:+.4f} m/s, "
        f"bracket=[{lower_height:.6g}, {upper_height:.6g}] m, "
        f"alpha={alpha:.6f}, phase_time={scheduler.phase_time}, "
        f"max_velocity_morph={np.max(np.abs(velocity_morph)):.6g} rad/s"
    )
    return gait, clamped_height, description


def named_ids(
    model: mujoco.MjModel, object_type: mujoco.mjtObj, names: tuple[str, ...]
) -> np.ndarray:
    ids = np.asarray(
        [mujoco.mj_name2id(model, object_type, name) for name in names], dtype=int
    )
    missing = [name for name, object_id in zip(names, ids) if object_id < 0]
    if missing:
        raise ValueError(f"Model is missing required names: {missing}")
    return ids


def apply_sample(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gait: np.ndarray,
    sample: int,
    keyframe_id: int,
    qpos_addresses: np.ndarray,
    dof_addresses: np.ndarray,
    base_height: float | None = None,
) -> None:
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    if base_height is not None:
        data.qpos[2] = base_height
    data.qpos[qpos_addresses] = gait[:16, sample]
    data.qvel[dof_addresses] = gait[16:, sample]
    mujoco.mj_forward(model, data)


def compute_foot_trajectories(
    model: mujoco.MjModel,
    gait: np.ndarray,
    keyframe_id: int,
    joint_ids: np.ndarray,
    foot_ids: np.ndarray,
    base_height: float | None = None,
) -> np.ndarray:
    data = mujoco.MjData(model)
    qpos_addresses = model.jnt_qposadr[joint_ids]
    dof_addresses = model.jnt_dofadr[joint_ids]
    trajectories = np.empty((gait.shape[1], len(foot_ids), 3), dtype=float)
    for sample in range(gait.shape[1]):
        apply_sample(
            model,
            data,
            gait,
            sample,
            keyframe_id,
            qpos_addresses,
            dof_addresses,
            base_height,
        )
        trajectories[sample] = data.site_xpos[foot_ids]
    return trajectories


def print_report(
    model: mujoco.MjModel,
    gait: np.ndarray,
    joint_ids: np.ndarray,
    foot_ids: np.ndarray,
    trajectories: np.ndarray,
    rate: float,
) -> int:
    print(
        f"Gait: 32 rows x {gait.shape[1]} samples, "
        f"{rate:g} Hz, cycle {gait.shape[1] / rate:.4f} s"
    )
    print("Joint position ranges [rad]:")
    violation_count = 0
    for row, (name, joint_id) in enumerate(zip(JOINT_NAMES, joint_ids)):
        values = gait[row]
        lower, upper = model.jnt_range[joint_id]
        outside = int(np.count_nonzero((values < lower) | (values > upper)))
        violation_count += outside
        print(
            f"  {name:16s} {values.min():8.4f} .. {values.max():8.4f}  "
            f"limit {lower:8.4f} .. {upper:8.4f}  outside={outside}"
        )

    print("Foot-site world trajectories with the base fixed at the keyframe [m]:")
    for foot_index, (name, site_id) in enumerate(zip(FOOT_NAMES, foot_ids)):
        points = trajectories[:, foot_index]
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        span = np.ptp(points, axis=0)
        radius = model.site_size[site_id, 0]
        clearance = minimum[2] - radius
        print(
            f"  {name}: min={np.round(minimum, 4)}, "
            f"max={np.round(maximum, 4)}, span={np.round(span, 4)}, "
            f"min_surface_z={clearance:.4f}"
        )

    print(f"Joint-limit violations: {violation_count}")
    return violation_count


def save_trajectories(
    path: Path, trajectories: np.ndarray, rate: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time_values = np.arange(trajectories.shape[0], dtype=float) / rate
    output = np.column_stack((time_values, trajectories.reshape(len(time_values), -1)))
    labels = ["time_s"] + [
        f"{foot}_{axis}"
        for foot in FOOT_NAMES
        for axis in ("x_m", "y_m", "z_m")
    ]
    np.savetxt(path, output, delimiter="\t", header="\t".join(labels), comments="# ")
    print(f"Saved foot trajectory: {path}")


def add_line(
    scene: mujoco.MjvScene,
    start: np.ndarray,
    end: np.ndarray,
    rgba: np.ndarray,
    width: float,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("MuJoCo user scene has no free geometry slots")
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).ravel(),
        rgba,
    )
    mujoco.mjv_connector(
        geom, mujoco.mjtGeom.mjGEOM_LINE, width, start, end
    )
    scene.ngeom += 1


def add_sphere(
    scene: mujoco.MjvScene,
    position: np.ndarray,
    rgba: np.ndarray,
    radius: float,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("MuJoCo user scene has no free geometry slots")
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray((radius, radius, radius), dtype=float),
        position,
        np.eye(3).ravel(),
        rgba,
    )
    scene.ngeom += 1


def draw_user_scene(
    scene: mujoco.MjvScene,
    trajectories: np.ndarray,
    current_sample: int,
) -> None:
    scene.ngeom = 0

    # Ground-reference grid at z=0 makes foot clearance immediately visible.
    grid_color = np.asarray((0.45, 0.45, 0.45, 0.28), dtype=np.float32)
    grid_extent = 1.0
    for coordinate in np.linspace(-grid_extent, grid_extent, 9):
        add_line(
            scene,
            np.asarray((-grid_extent, coordinate, 0.0)),
            np.asarray((grid_extent, coordinate, 0.0)),
            grid_color,
            1.0,
        )
        add_line(
            scene,
            np.asarray((coordinate, -grid_extent, 0.0)),
            np.asarray((coordinate, grid_extent, 0.0)),
            grid_color,
            1.0,
        )

    for foot_index, color in enumerate(FOOT_COLORS):
        points = trajectories[:, foot_index]
        for sample in range(len(points)):
            next_sample = (sample + 1) % len(points)
            if np.linalg.norm(points[next_sample] - points[sample]) > 1e-10:
                add_line(scene, points[sample], points[next_sample], color, 4.0)
        marker_color = color.copy()
        marker_color[3] = 1.0
        add_sphere(scene, points[current_sample], marker_color, 0.042)


def run_viewer(
    model: mujoco.MjModel,
    gait: np.ndarray,
    trajectories: np.ndarray,
    keyframe_id: int,
    joint_ids: np.ndarray,
    rate: float,
    hide_ui: bool,
    base_height: float | None = None,
) -> None:
    data = mujoco.MjData(model)
    qpos_addresses = model.jnt_qposadr[joint_ids]
    dof_addresses = model.jnt_dofadr[joint_ids]
    sample = 0
    apply_sample(
        model,
        data,
        gait,
        sample,
        keyframe_id,
        qpos_addresses,
        dof_addresses,
        base_height,
    )

    playback = {"paused": False, "step": 0, "reset": False}
    playback_lock = threading.Lock()

    def key_callback(keycode: int) -> None:
        with playback_lock:
            if keycode == 32:  # Space
                playback["paused"] = not playback["paused"]
            elif keycode == 262:  # GLFW right arrow
                playback["paused"] = True
                playback["step"] += 1
            elif keycode == 263:  # GLFW left arrow
                playback["paused"] = True
                playback["step"] -= 1
            elif keycode in (ord("R"), ord("r")):
                playback["reset"] = True

    print("Viewer colors: FR=red, FL=green, RR=blue, RL=yellow")
    print("Controls: Space=pause/resume, Left/Right=step, R=first sample")
    period = 1.0 / rate
    next_sample_time = time.monotonic() + period

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
        show_left_ui=not hide_ui,
        show_right_ui=not hide_ui,
    ) as viewer:
        viewer.cam.lookat[:] = data.qpos[:3]
        viewer.cam.distance = 2.2
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -18.0

        while viewer.is_running():
            now = time.monotonic()
            update = False
            with playback_lock:
                paused = bool(playback["paused"])
                requested_step = int(playback["step"])
                playback["step"] = 0
                reset = bool(playback["reset"])
                playback["reset"] = False

            if reset:
                sample = 0
                update = True
                next_sample_time = now + period
            elif requested_step:
                sample = (sample + requested_step) % gait.shape[1]
                update = True
                next_sample_time = now + period
            elif not paused and now >= next_sample_time:
                sample = (sample + 1) % gait.shape[1]
                update = True
                next_sample_time += period
                if next_sample_time <= now:
                    next_sample_time = now + period

            if update:
                apply_sample(
                    model,
                    data,
                    gait,
                    sample,
                    keyframe_id,
                    qpos_addresses,
                    dof_addresses,
                    base_height,
                )

            with viewer.lock():
                draw_user_scene(viewer.user_scn, trajectories, sample)
            viewer.sync()
            time.sleep(min(0.005, period / 2.0))


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)

    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    if args.gait_name is not None:
        height = 0.225 if args.height is None else float(args.height)
        gait, base_height, source_description = interpolate_gait_reference(
            args.gait_name,
            height,
            args.height_rate,
            args.phase_time,
        )
        rate = 100.0 if args.rate is None else float(args.rate)
    else:
        if (
            args.height is not None
            or args.height_rate != 0.0
            or args.phase_time != 0
        ):
            raise ValueError(
                "--height, --height-rate, and --phase-time require --gait-name"
            )
        gait_path = DEFAULT_GAIT if args.gait is None else args.gait
        gait_path = gait_path.resolve()
        gait = load_gait(gait_path)
        base_height = None
        source_description = f"TSV: {gait_path}"
        rate = 80.0 if args.rate is None else float(args.rate)
    if rate <= 0.0:
        raise ValueError("--rate must be positive")

    joint_ids = named_ids(model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAMES)
    foot_ids = named_ids(model, mujoco.mjtObj.mjOBJ_SITE, FOOT_NAMES)
    keyframe_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe
    )
    if keyframe_id < 0:
        raise ValueError(f"Unknown keyframe: {args.keyframe}")

    trajectories = compute_foot_trajectories(
        model,
        gait,
        keyframe_id,
        joint_ids,
        foot_ids,
        base_height,
    )
    print(f"Model: {args.model.resolve()}")
    print(f"Gait source: {source_description}")
    print(f"Base keyframe: {args.keyframe}")
    if base_height is not None:
        print(f"Fixed base height: {base_height:.4f} m")
        if args.height_rate != 0.0:
            print(
                "Note: height-rate morphing changes joint dq; this kinematic "
                "viewer displays joint q and therefore has the same geometry "
                "for different rates at a fixed height."
            )
    print_report(model, gait, joint_ids, foot_ids, trajectories, rate)

    if args.save_trajectory is not None:
        save_trajectories(
            args.save_trajectory.resolve(),
            trajectories,
            rate,
        )
    if not args.headless:
        run_viewer(
            model,
            gait,
            trajectories,
            keyframe_id,
            joint_ids,
            rate,
            args.hide_ui,
            base_height,
        )


if __name__ == "__main__":
    main()
