"""Visualize the Locomani arm-to-torso hard collision constraint.

The model, task target, arm FK, and collision classification follow the current
``mppi_locomani.MPPI`` path. Move the four sliders to inspect whether any arm
capsule violates the configured hard clearance. Locomani excludes invalid
rollouts from the MPPI update.

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
import sys

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import yaml
from matplotlib.patches import Patch
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MANI_MPPI_ROOT = PACKAGE_ROOT / "mani_mppi"
CONTROLLER_ROOT = MANI_MPPI_ROOT / "control/controllers"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from mani_mppi.control.collision import ArmTorsoCollision  # noqa: E402
from mani_mppi.control.kinematics.arm_kinematics import (
    ARM_JOINT_NAMES,
    ArmKinematics,
)  # noqa: E402
from mani_mppi.utils.tasks import get_task  # noqa: E402


LOCOMANI_TASK = get_task("locomani")
DEFAULT_MODEL = MANI_MPPI_ROOT / LOCOMANI_TASK["model_path"]
DEFAULT_CONFIG = CONTROLLER_ROOT / LOCOMANI_TASK["config_path"]
DEFAULT_EE_SITE = LOCOMANI_TASK.get("ee_site", "end_effector")
LINK_NAMES = ("shoulder", "upper arm", "forearm", "wrist pitch")

VALID_COLOR = "#27ae60"
HARD_COLOR = "#c0392b"
BODY_COLOR = "#5d6d7e"
SAMPLE_COLOR = "#2980b9"
HARD_MARGIN_COLOR = "#f39c12"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--keyframe",
        help=(
            "MuJoCo keyframe used for the base pose "
            "(default: body_reference_keyframe from the Locomani config)"
        ),
    )
    parser.add_argument(
        "--goal-index",
        type=int,
        default=0,
        help=(
            "Locomani EE goal used to generate the initial arm pose "
            "(default: 0; ignored when --arm-q is given)"
        ),
    )
    parser.add_argument(
        "--arm-q",
        nargs=4,
        type=float,
        metavar=("ROTATION", "PITCH", "ELBOW", "WRIST_PITCH"),
        help=(
            "Initial arm joint angles in radians "
            "(default: Locomani IK solution for --goal-index)"
        ),
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Save the initial pose as an image",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive window (useful with --save)",
    )
    return parser.parse_args()


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
    *,
    alpha: float = 0.45,
    draw_centerline: bool = True,
    surface_linewidth: float = 0.0,
) -> None:
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-12:
        return
    unit = direction / length
    first, second = orthogonal_basis(direction)
    theta = np.linspace(0.0, 2.0 * np.pi, 24)
    radial = (
        np.cos(theta)[:, None] * first
        + np.sin(theta)[:, None] * second
    )
    rings = np.stack(
        (
            start[None, :] + radius * radial,
            end[None, :] + radius * radial,
        ),
        axis=0,
    )
    axis.plot_surface(
        rings[:, :, 0],
        rings[:, :, 1],
        rings[:, :, 2],
        color=color,
        alpha=alpha,
        linewidth=surface_linewidth,
        shade=True,
    )

    # Complete the swept-sphere geometry with one outward hemisphere at each
    # endpoint. This matches the segment-plus-radius capsule used for clearance.
    for center, phi in (
        (start, np.linspace(0.5 * np.pi, np.pi, 10)),
        (end, np.linspace(0.0, 0.5 * np.pi, 10)),
    ):
        cap = (
            center[None, None, :]
            + radius
            * (
                np.cos(phi)[:, None, None] * unit[None, None, :]
                + np.sin(phi)[:, None, None] * radial[None, :, :]
            )
        )
        axis.plot_surface(
            cap[:, :, 0],
            cap[:, :, 1],
            cap[:, :, 2],
            color=color,
            alpha=alpha,
            linewidth=surface_linewidth,
            shade=True,
        )
    if draw_centerline:
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
        keyframe: str | None,
        goal_index: int,
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
        self.arm_kinematics = ArmKinematics(
            self.model,
            config,
            ee_site_name=DEFAULT_EE_SITE,
        )
        self.collision = ArmTorsoCollision(self.model, config)
        self.arm_qpos_addresses = self.arm_kinematics.arm_qpos_indices
        self.body_geom_id = self.collision.body_geom_id
        self.body_half_size = self.collision.body_half_size
        self.capsule_radii = self.collision.capsule_radii
        self.shoulder_exclusion = self.collision.shoulder_exclusion
        self.hard_distance = self.collision.hard_distance

        keyframe = keyframe or config.get("body_reference_keyframe", "stand")
        keyframe_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe
        )
        if keyframe_id < 0:
            raise ValueError(f"Unknown keyframe: {keyframe}")
        mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe_id)
        self.base_qpos = self.data.qpos.copy()
        if arm_q is None:
            goals = np.asarray(LOCOMANI_TASK["ee_goal_pos"], dtype=float)
            if goal_index < 0 or goal_index >= len(goals):
                raise ValueError(
                    f"goal-index must be in [0, {len(goals) - 1}], "
                    f"got {goal_index}"
                )
            self.goal_index: int | None = goal_index
            self.goal_position = goals[goal_index].copy()
            self.arm_q = self.arm_kinematics.solve_ik(
                self.goal_position,
                self.base_qpos,
                strict=False,
            )
        else:
            self.goal_index = None
            self.goal_position = None
            self.arm_q = np.asarray(arm_q, dtype=float)
        lower = self.arm_kinematics.arm_joint_lower
        upper = self.arm_kinematics.arm_joint_upper
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
        self.last_segment_valid = np.ones(3, dtype=bool)
        self.last_exact_evaluations = 0
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
        np.ndarray,
    ]:
        self.data.qpos[:] = self.base_qpos
        self.data.qpos[self.arm_qpos_addresses] = arm_q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        points = self.arm_kinematics.batch_arm_fk(
            self.data.qpos[None, :]
        )[0][0]

        # This is the exact helper call made by Locomani after FK. It owns the
        # sampled bounds, ambiguous-case refinement, and hard-validity decision.
        collision_result = self.collision.evaluate(
            self.data.qpos[None, :],
            points[None, :, :],
        )

        # Reconstruct only the helper's trimmed centerlines and uniform sample
        # positions for display. Clearance and validity are never recomputed.
        starts = points[:3].copy()
        ends = points[1:].copy()

        upper_vector = ends[0] - starts[0]
        upper_length = np.linalg.norm(upper_vector)
        trim = min(self.shoulder_exclusion, upper_length)
        starts[0] += upper_vector * trim / max(upper_length, 1e-12)

        center = self.data.geom_xpos[self.body_geom_id].copy()
        rotation = self.data.geom_xmat[self.body_geom_id].reshape(3, 3).copy()
        samples_world = (
            starts[:, None, :]
            + self.collision.fast_alpha[None, :, None]
            * (ends - starts)[:, None, :]
        )
        self.last_exact_evaluations = collision_result.exact_evaluations
        return (
            points,
            starts,
            ends,
            center,
            rotation,
            collision_result.clearance[0],
            collision_result.segment_valid[0],
            samples_world,
        )

    @staticmethod
    def clearance_color(valid: bool) -> str:
        return VALID_COLOR if valid else HARD_COLOR

    def draw(self, arm_q: np.ndarray) -> None:
        self.axis.clear()
        (
            points,
            starts,
            ends,
            center,
            rotation,
            clearances,
            segment_valid,
            samples_world,
        ) = self.arm_geometry(arm_q)
        self.last_clearances = clearances
        self.last_segment_valid = segment_valid

        draw_box(
            self.axis,
            center,
            rotation,
            self.body_half_size,
            BODY_COLOR,
            alpha=0.28,
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

        for index, (
            start,
            end,
            radius,
            clearance,
            valid,
        ) in enumerate(
            zip(
                starts,
                ends,
                self.capsule_radii,
                clearances,
                segment_valid,
            )
        ):
            color = self.clearance_color(bool(valid))
            if self.hard_distance > 0.0:
                draw_capsule(
                    self.axis,
                    start,
                    end,
                    float(radius + self.hard_distance),
                    HARD_MARGIN_COLOR,
                    alpha=0.10,
                    draw_centerline=False,
                    surface_linewidth=0.25,
                )
            draw_capsule(self.axis, start, end, float(radius), color)
            label_position = 0.5 * (start + end)
            self.axis.text(
                *label_position,
                f"L{index + 1}: d={1000.0 * clearance:.1f} mm",
                color=color,
                fontsize=9,
                fontweight="bold",
            )

        flat_samples = samples_world.reshape(-1, 3)
        self.axis.scatter(
            flat_samples[:, 0],
            flat_samples[:, 1],
            flat_samples[:, 2],
            color=SAMPLE_COLOR,
            edgecolors="white",
            linewidths=0.35,
            s=18,
            depthshade=False,
        )
        self.axis.scatter(
            points[:, 0], points[:, 1], points[:, 2], color="black", s=18
        )
        self.axis.text(*points[-1], " EE", color="black", fontsize=9)

        all_points = np.vstack(
            (
                points,
                flat_samples,
                box_vertices(center, rotation, self.body_half_size),
            )
        )
        minimum = all_points.min(axis=0)
        maximum = all_points.max(axis=0)
        midpoint = 0.5 * (minimum + maximum)
        envelope_radius = float(
            np.max(self.capsule_radii) + max(self.hard_distance, 0.0)
        )
        span = max(
            float(np.max(maximum - minimum)) + 2.0 * envelope_radius,
            0.5,
        ) * 1.2
        self.axis.set_xlim(midpoint[0] - span / 2.0, midpoint[0] + span / 2.0)
        self.axis.set_ylim(midpoint[1] - span / 2.0, midpoint[1] + span / 2.0)
        self.axis.set_zlim(midpoint[2] - span / 2.0, midpoint[2] + span / 2.0)
        self.axis.set_box_aspect((1.0, 1.0, 1.0))
        self.axis.set_xlabel("world x [m]")
        self.axis.set_ylabel("world y [m]")
        self.axis.set_zlabel("world z [m]")
        self.axis.set_title(
            "Locomani arm-to-torso hard collision constraint\n"
            f"valid iff d >= {1000.0 * self.hard_distance:.1f} mm "
            f"| {self.collision.fast_samples} samples/segment, "
            f"exact refinements={self.last_exact_evaluations}"
        )
        self.axis.view_init(elev=23.0, azim=-58.0)
        legend_handles = [
            Patch(color=VALID_COLOR, label="hard-valid"),
            Patch(color=HARD_COLOR, label="hard violation"),
            Patch(color=BODY_COLOR, label="torso collision box"),
            Patch(color=SAMPLE_COLOR, label="fast centerline samples"),
        ]
        if self.hard_distance > 0.0:
            legend_handles.append(
                Patch(
                    color=HARD_MARGIN_COLOR,
                    alpha=0.35,
                    label=(
                        "hard margin: "
                        f"+{1000.0 * self.hard_distance:.1f} mm"
                    ),
                )
            )
        self.axis.legend(
            handles=legend_handles,
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
        if self.goal_index is not None:
            print(
                f"Locomani goal {self.goal_index}: "
                f"{np.round(self.goal_position, 4)}"
            )
        print(
            f"Hard constraint: enabled={self.collision.enabled}, "
            f"d >= {self.hard_distance:.4f} m, "
            f"fast_samples={self.collision.fast_samples}, "
            f"exact_evaluations={self.last_exact_evaluations}"
        )
        for name, radius, clearance, valid in zip(
            LINK_NAMES,
            self.capsule_radii,
            self.last_clearances,
            self.last_segment_valid,
        ):
            status = "VALID" if valid else "HARD VIOLATION"
            print(
                f"  {name:10s}: radius={radius:.4f} m, "
                f"clearance={clearance:.4f} m ({status})"
            )
        rollout_valid = bool(
            not self.collision.enabled or np.all(self.last_segment_valid)
        )
        print(f"Pose accepted by Locomani hard mask: {rollout_valid}")


def main() -> None:
    args = parse_args()
    visualizer = ClearanceVisualizer(
        args.model,
        args.config,
        args.keyframe,
        args.goal_index,
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
