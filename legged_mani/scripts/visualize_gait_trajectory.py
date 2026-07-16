"""Replay a joint-angle gait and draw foot trajectories in MuJoCo.

The gait is applied kinematically: the floating base stays at a named model
keyframe while the 16 actuated joint positions are copied from the TSV.  This
makes the viewer useful for checking reference data without controller or
contact-dynamics effects.

Examples
--------
    python3 -m legged_mani.scripts.visualize_gait_trajectory
    python3 -m legged_mani.scripts.visualize_gait_trajectory --headless
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PACKAGE_ROOT / "mani_mppi/models/b2_z1_4dof.xml"
DEFAULT_GAIT = (
    PACKAGE_ROOT
    / "mani_mppi/control/gait_scheduler/gaits/"
    "FAST/b2_retargeted/walking_gait_raibert_FAST_0_0_10cm_100hz.tsv"
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
    "joint1",
    "joint2",
    "joint3",
    "joint4",
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
    parser.add_argument("--gait", type=Path, default=DEFAULT_GAIT)
    parser.add_argument(
        "--rate",
        type=float,
        default=80.0,
        help="Playback sample rate in Hz (default: 80)",
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
) -> None:
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    data.qpos[qpos_addresses] = gait[:16, sample]
    data.qvel[dof_addresses] = gait[16:, sample]
    mujoco.mj_forward(model, data)


def compute_foot_trajectories(
    model: mujoco.MjModel,
    gait: np.ndarray,
    keyframe_id: int,
    joint_ids: np.ndarray,
    foot_ids: np.ndarray,
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
) -> None:
    data = mujoco.MjData(model)
    qpos_addresses = model.jnt_qposadr[joint_ids]
    dof_addresses = model.jnt_dofadr[joint_ids]
    sample = 0
    apply_sample(
        model, data, gait, sample, keyframe_id, qpos_addresses, dof_addresses
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
                )

            with viewer.lock():
                draw_user_scene(viewer.user_scn, trajectories, sample)
            viewer.sync()
            time.sleep(min(0.005, period / 2.0))


def main() -> None:
    args = parse_args()
    if args.rate <= 0.0:
        raise ValueError("--rate must be positive")
    if not args.model.is_file():
        raise FileNotFoundError(args.model)

    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    gait = load_gait(args.gait.resolve())
    joint_ids = named_ids(model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAMES)
    foot_ids = named_ids(model, mujoco.mjtObj.mjOBJ_SITE, FOOT_NAMES)
    keyframe_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe
    )
    if keyframe_id < 0:
        raise ValueError(f"Unknown keyframe: {args.keyframe}")

    trajectories = compute_foot_trajectories(
        model, gait, keyframe_id, joint_ids, foot_ids
    )
    print(f"Model: {args.model.resolve()}")
    print(f"Gait:  {args.gait.resolve()}")
    print(f"Base keyframe: {args.keyframe}")
    print_report(model, gait, joint_ids, foot_ids, trajectories, args.rate)

    if args.save_trajectory is not None:
        save_trajectories(args.save_trajectory.resolve(), trajectories, args.rate)
    if not args.headless:
        run_viewer(
            model,
            gait,
            trajectories,
            keyframe_id,
            joint_ids,
            args.rate,
            args.hide_ui,
        )


if __name__ == "__main__":
    main()
