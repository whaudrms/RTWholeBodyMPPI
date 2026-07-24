"""Visualize B2-Z1 arm-to-torso capsule clearance.

The torso collision box and the three arm capsules match the geometry used by
``mppi_locomani.MPPI._arm_torso_clearance``.  Move the four sliders to inspect
how each arm link enters the soft-cost region or violates the hard clearance.

Examples
--------
Open the interactive visualization::

    python3 legged_mani/tests/visualize_arm_body_clearance.py

Save one pose without opening a window::

    python3 legged_mani/tests/visualize_arm_body_clearance.py \
        --arm-q 0.0 1.0 -0.6 0.0 --save /tmp/arm_clearance.png --no-show
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import yaml
from matplotlib.patches import Patch
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PACKAGE_ROOT / "mani_mppi/models/b2_z1_4dof.xml"
DEFAULT_CONFIG = (
    PACKAGE_ROOT
    / "mani_mppi/control/controllers/configs/mppi_locomani.yml"
)

ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4")
ARM_POINT_BODY_NAMES = ("link02", "link03", "link04")
LINK_NAMES = ("upper arm", "forearm", "wrist")
EE_SITE_NAME = "gripper_center"

SAFE_COLOR = "#27ae60"
SOFT_COLOR = "#f39c12"
HARD_COLOR = "#c0392b"
BODY_COLOR = "#5d6d7e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--keyframe", default="stand")
    parser.add_argument(
        "--arm-q",
        nargs=4,
        type=float,
        metavar=("Q1", "Q2", "Q3", "Q4"),
        help="Initial arm joint angles in radians (default: keyframe values)",
    )
    parser.add_argument("--save", type=Path, help="Save the initial pose as an image")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive window (useful with --save)",
    )
    return parser.parse_args()


def named_ids(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    names: tuple[str, ...],
) -> np.ndarray:
    ids = np.asarray(
        [mujoco.mj_name2id(model, object_type, name) for name in names],
        dtype=int,
    )
    missing = [name for name, object_id in zip(names, ids) if object_id < 0]
    if missing:
        raise ValueError(f"Model is missing required names: {missing}")
    return ids


def closest_segment_aabb(
    starts: np.ndarray,
    ends: np.ndarray,
    half_size: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact distances and closest points for segments and an AABB.

    Inputs and returned closest points are in the AABB local frame.  This is
    the same piecewise-quadratic minimization used by the locomani controller,
    extended here to retain the closest points for drawing distance lines.
    """
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    half_size = np.asarray(half_size, dtype=float)
    if starts.shape != ends.shape or starts.ndim != 2 or starts.shape[1] != 3:
        raise ValueError("starts and ends must both have shape (N, 3)")

    direction = ends - starts
    count = len(starts)
    boundaries = np.stack((-half_size, half_size), axis=0)
    numerator = boundaries[None, :, :] - starts[:, None, :]
    denominator = direction[:, None, :]
    crossings = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=np.abs(denominator) > 1e-12,
    ).reshape(count, 6)
    crossings = np.clip(crossings, 0.0, 1.0)

    knots = np.sort(
        np.concatenate(
            (np.zeros((count, 1)), crossings, np.ones((count, 1))), axis=1
        ),
        axis=1,
    )
    lower = knots[:, :-1]
    upper = knots[:, 1:]
    midpoint = 0.5 * (lower + upper)
    midpoint_position = (
        starts[:, None, :] + midpoint[:, :, None] * direction[:, None, :]
    )
    signs = np.where(
        midpoint_position > half_size,
        1.0,
        np.where(midpoint_position < -half_size, -1.0, 0.0),
    )
    active = signs != 0.0
    offset = starts[:, None, :] - signs * half_size
    derivative_offset = np.sum(
        active * direction[:, None, :] * offset, axis=2
    )
    derivative_scale = np.sum(
        active * direction[:, None, :] ** 2, axis=2
    )
    stationary = np.divide(
        -derivative_offset,
        derivative_scale,
        out=midpoint.copy(),
        where=derivative_scale > 1e-16,
    )
    stationary = np.minimum(np.maximum(stationary, lower), upper)

    segment_candidates = (
        starts[:, None, :] + stationary[:, :, None] * direction[:, None, :]
    )
    box_candidates = np.clip(segment_candidates, -half_size, half_size)
    delta = segment_candidates - box_candidates
    squared_distances = np.sum(delta * delta, axis=2)
    best = np.argmin(squared_distances, axis=1)
    rows = np.arange(count)
    return (
        np.sqrt(squared_distances[rows, best]),
        segment_candidates[rows, best],
        box_candidates[rows, best],
    )


def box_vertices(
    center: np.ndarray,
    rotation: np.ndarray,
    half_size: np.ndarray,
) -> np.ndarray:
    local = np.asarray(list(product((-1.0, 1.0), repeat=3))) * half_size
    return local @ rotation.T + center


BOX_FACE_INDICES = (
    (0, 1, 3, 2),
    (4, 5, 7, 6),
    (0, 1, 5, 4),
    (2, 3, 7, 6),
    (0, 2, 6, 4),
    (1, 3, 7, 5),
)


BOX_EDGE_INDICES = tuple(
    (first, second)
    for first in range(8)
    for second in range(first + 1, 8)
    if bin(first ^ second).count("1") == 1
)


def draw_box(
    axis,
    center: np.ndarray,
    rotation: np.ndarray,
    half_size: np.ndarray,
    color: str,
    alpha: float,
    wire_only: bool = False,
    linestyle: str = "-",
) -> None:
    vertices = box_vertices(center, rotation, half_size)
    if not wire_only:
        faces = [[vertices[index] for index in face] for face in BOX_FACE_INDICES]
        axis.add_collection3d(
            Poly3DCollection(
                faces,
                facecolors=color,
                edgecolors=color,
                linewidths=0.8,
                alpha=alpha,
            )
        )
    edges = [[vertices[first], vertices[second]] for first, second in BOX_EDGE_INDICES]
    axis.add_collection3d(
        Line3DCollection(
            edges,
            colors=color,
            linewidths=1.2,
            linestyles=linestyle,
            alpha=max(alpha, 0.65),
        )
    )


def orthogonal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unit = direction / np.linalg.norm(direction)
    reference = np.array((1.0, 0.0, 0.0))
    if abs(float(np.dot(unit, reference))) > 0.9:
        reference = np.array((0.0, 1.0, 0.0))
    first = np.cross(unit, reference)
    first /= np.linalg.norm(first)
    second = np.cross(unit, first)
    return first, second


def draw_capsule(
    axis,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    color: str,
) -> None:
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-12:
        return
    first, second = orthogonal_basis(direction)
    theta = np.linspace(0.0, 2.0 * np.pi, 18)
    rings = np.stack(
        (
            start[None, :] + radius * (
                np.cos(theta)[:, None] * first
                + np.sin(theta)[:, None] * second
            ),
            end[None, :] + radius * (
                np.cos(theta)[:, None] * first
                + np.sin(theta)[:, None] * second
            ),
        ),
        axis=0,
    )
    axis.plot_surface(
        rings[:, :, 0],
        rings[:, :, 1],
        rings[:, :, 2],
        color=color,
        alpha=0.45,
        linewidth=0.0,
        shade=True,
    )
    axis.plot(
        (start[0], end[0]),
        (start[1], end[1]),
        (start[2], end[2]),
        color=color,
        linewidth=3.0,
    )


class ClearanceVisualizer:
    def __init__(
        self,
        model_path: Path,
        config_path: Path,
        keyframe: str,
        arm_q: np.ndarray | None,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.arm_joint_ids = named_ids(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINT_NAMES
        )
        self.arm_qpos_addresses = self.model.jnt_qposadr[self.arm_joint_ids]
        self.arm_body_ids = named_ids(
            self.model, mujoco.mjtObj.mjOBJ_BODY, ARM_POINT_BODY_NAMES
        )
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME
        )
        if self.ee_site_id < 0:
            raise ValueError(f"Model is missing EE site: {EE_SITE_NAME}")

        geom_name = config.get("collision_body_geom", "base_collision")
        self.body_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
        )
        if self.body_geom_id < 0:
            raise ValueError(f"Model is missing collision geom: {geom_name}")
        if self.model.geom_type[self.body_geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError("collision_body_geom must be a box")

        self.body_half_size = self.model.geom_size[self.body_geom_id, :3].copy()
        self.capsule_radii = np.asarray(
            config.get("collision_capsule_radii", (0.030, 0.030, 0.0375)),
            dtype=float,
        )
        self.shoulder_exclusion = float(
            config.get("collision_shoulder_exclusion", 0.07)
        )
        self.safe_distance = float(config.get("collision_safe_distance", 0.05))
        self.hard_distance = float(config.get("collision_hard_distance", 0.005))

        keyframe_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe
        )
        if keyframe_id < 0:
            raise ValueError(f"Unknown keyframe: {keyframe}")
        mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe_id)
        self.base_qpos = self.data.qpos.copy()
        if arm_q is None:
            self.arm_q = self.base_qpos[self.arm_qpos_addresses].copy()
        else:
            self.arm_q = np.asarray(arm_q, dtype=float)
        lower = self.model.jnt_range[self.arm_joint_ids, 0]
        upper = self.model.jnt_range[self.arm_joint_ids, 1]
        if np.any(self.arm_q < lower) or np.any(self.arm_q > upper):
            raise ValueError(
                f"arm-q {self.arm_q} is outside joint limits "
                f"{np.column_stack((lower, upper))}"
            )

        self.figure = plt.figure(figsize=(12.5, 8.5))
        self.axis = self.figure.add_subplot(111, projection="3d")
        self.figure.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.24)
        self.sliders: list[Slider] = []
        self.last_clearances = np.zeros(3)
        self.draw(self.arm_q)
        self._add_sliders(lower, upper)

    def arm_geometry(
        self, arm_q: np.ndarray
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        self.data.qpos[:] = self.base_qpos
        self.data.qpos[self.arm_qpos_addresses] = arm_q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        points = np.empty((4, 3), dtype=float)
        points[:3] = self.data.xpos[self.arm_body_ids]
        points[3] = self.data.site_xpos[self.ee_site_id]
        starts = points[:3].copy()
        ends = points[1:].copy()

        upper_vector = ends[0] - starts[0]
        upper_length = np.linalg.norm(upper_vector)
        trim = min(self.shoulder_exclusion, upper_length)
        starts[0] += upper_vector * trim / max(upper_length, 1e-12)

        center = self.data.geom_xpos[self.body_geom_id].copy()
        rotation = self.data.geom_xmat[self.body_geom_id].reshape(3, 3).copy()
        starts_local = (starts - center) @ rotation
        ends_local = (ends - center) @ rotation
        distances, closest_segment_local, closest_box_local = closest_segment_aabb(
            starts_local, ends_local, self.body_half_size
        )
        clearances = distances - self.capsule_radii
        closest_segment_world = closest_segment_local @ rotation.T + center
        closest_box_world = closest_box_local @ rotation.T + center
        return (
            points,
            starts,
            ends,
            center,
            rotation,
            clearances,
            np.stack((closest_segment_world, closest_box_world), axis=1),
        )

    def clearance_color(self, clearance: float) -> str:
        if clearance < self.hard_distance:
            return HARD_COLOR
        if clearance < self.safe_distance:
            return SOFT_COLOR
        return SAFE_COLOR

    def draw(self, arm_q: np.ndarray) -> None:
        self.axis.clear()
        (
            points,
            starts,
            ends,
            center,
            rotation,
            clearances,
            closest_pairs,
        ) = self.arm_geometry(arm_q)
        self.last_clearances = clearances

        draw_box(
            self.axis,
            center,
            rotation,
            self.body_half_size,
            BODY_COLOR,
            alpha=0.28,
        )

        # This box is a conservative visual envelope. The exact safe region is
        # the rounded Minkowski sum of the torso box and capsule radius.
        max_radius = float(np.max(self.capsule_radii))
        draw_box(
            self.axis,
            center,
            rotation,
            self.body_half_size + max_radius + self.safe_distance,
            SOFT_COLOR,
            alpha=0.65,
            wire_only=True,
            linestyle="--",
        )
        draw_box(
            self.axis,
            center,
            rotation,
            self.body_half_size + max_radius + self.hard_distance,
            HARD_COLOR,
            alpha=0.65,
            wire_only=True,
            linestyle=":",
        )

        excluded_end = starts[0]
        self.axis.plot(
            (points[0, 0], excluded_end[0]),
            (points[0, 1], excluded_end[1]),
            (points[0, 2], excluded_end[2]),
            color="black",
            linestyle=":",
            linewidth=2.0,
            alpha=0.6,
        )

        for index, (start, end, radius, clearance) in enumerate(
            zip(starts, ends, self.capsule_radii, clearances)
        ):
            color = self.clearance_color(float(clearance))
            draw_capsule(self.axis, start, end, float(radius), color)
            segment_point, box_point = closest_pairs[index]
            self.axis.plot(
                (segment_point[0], box_point[0]),
                (segment_point[1], box_point[1]),
                (segment_point[2], box_point[2]),
                color=color,
                linestyle="--",
                linewidth=1.6,
            )
            label_position = 0.5 * (segment_point + box_point)
            self.axis.text(
                *label_position,
                f"L{index + 1}: {1000.0 * clearance:.1f} mm",
                color=color,
                fontsize=9,
                fontweight="bold",
            )

        self.axis.scatter(
            points[:, 0], points[:, 1], points[:, 2], color="black", s=18
        )
        self.axis.text(*points[-1], " EE", color="black", fontsize=9)

        all_points = np.vstack(
            (
                points,
                box_vertices(
                    center,
                    rotation,
                    self.body_half_size + max_radius + self.safe_distance,
                ),
            )
        )
        minimum = all_points.min(axis=0)
        maximum = all_points.max(axis=0)
        midpoint = 0.5 * (minimum + maximum)
        span = max(float(np.max(maximum - minimum)), 0.5) * 1.2
        self.axis.set_xlim(midpoint[0] - span / 2.0, midpoint[0] + span / 2.0)
        self.axis.set_ylim(midpoint[1] - span / 2.0, midpoint[1] + span / 2.0)
        self.axis.set_zlim(midpoint[2] - span / 2.0, midpoint[2] + span / 2.0)
        self.axis.set_box_aspect((1.0, 1.0, 1.0))
        self.axis.set_xlabel("world x [m]")
        self.axis.set_ylabel("world y [m]")
        self.axis.set_zlabel("world z [m]")
        self.axis.set_title(
            "Arm capsule-to-torso clearance\n"
            f"soft < {1000.0 * self.safe_distance:.1f} mm, "
            f"hard < {1000.0 * self.hard_distance:.1f} mm"
        )
        self.axis.view_init(elev=23.0, azim=-58.0)
        self.axis.legend(
            handles=(
                Patch(color=SAFE_COLOR, label="safe"),
                Patch(color=SOFT_COLOR, label="soft-cost region"),
                Patch(color=HARD_COLOR, label="hard violation"),
                Patch(color=BODY_COLOR, label="torso collision box"),
            ),
            loc="upper left",
        )
        self.figure.canvas.draw_idle()

    def _add_sliders(self, lower: np.ndarray, upper: np.ndarray) -> None:
        for index, name in enumerate(ARM_JOINT_NAMES):
            slider_axis = self.figure.add_axes(
                (0.20, 0.17 - 0.038 * index, 0.62, 0.022)
            )
            slider = Slider(
                slider_axis,
                name,
                float(lower[index]),
                float(upper[index]),
                valinit=float(self.arm_q[index]),
                valstep=0.002,
            )
            slider.on_changed(self._slider_changed)
            self.sliders.append(slider)

    def _slider_changed(self, _value: float) -> None:
        self.arm_q = np.asarray([slider.val for slider in self.sliders])
        self.draw(self.arm_q)

    def print_report(self) -> None:
        print(f"Arm q [rad]: {np.round(self.arm_q, 4)}")
        print(
            f"Thresholds: safe={self.safe_distance:.4f} m, "
            f"hard={self.hard_distance:.4f} m"
        )
        for name, radius, clearance in zip(
            LINK_NAMES, self.capsule_radii, self.last_clearances
        ):
            if clearance < self.hard_distance:
                status = "HARD VIOLATION"
            elif clearance < self.safe_distance:
                status = "soft-cost region"
            else:
                status = "safe"
            print(
                f"  {name:10s}: radius={radius:.4f} m, "
                f"clearance={clearance:.4f} m ({status})"
            )


def main() -> None:
    args = parse_args()
    visualizer = ClearanceVisualizer(
        args.model,
        args.config,
        args.keyframe,
        None if args.arm_q is None else np.asarray(args.arm_q, dtype=float),
    )
    visualizer.print_report()
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        visualizer.figure.savefig(args.save, dpi=180, bbox_inches="tight")
        print(f"Saved visualization: {args.save}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(visualizer.figure)


if __name__ == "__main__":
    main()
