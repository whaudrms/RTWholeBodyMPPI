"""Interactively inspect Locomani's height-conditioned gait interpolation.

The plotted result comes directly from
``HeightConditionedGaitScheduler.get_reference``. Dashed lines show the two
height-bank references that bracket the requested base height, and the solid
line shows the phase-aligned interpolated reference used by the controller.
For velocity rows, the result also includes the configured height-rate
morphing term.

Examples
--------
Inspect the first leg's position and velocity rows::

    python3 legged_mani/tests/visualize_gait_reference.py

Inspect selected rows from the walking gait::

    python3 legged_mani/tests/visualize_gait_reference.py \
        --gait walk_fast --height 0.40 --height-rate -0.10 \
        --rows 0 1 2 16 17 18

Save the initial view without opening a window::

    python3 legged_mani/tests/visualize_gait_reference.py \
        --save /tmp/gait_interpolation.png --no-show
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from mani_mppi.control.controllers.whole_body_arm_controller import (  # noqa: E402
    HEIGHT_GAIT_PATHS,
)
from mani_mppi.control.gait_scheduler.height_conditioned_scheduler import (  # noqa: E402
    HeightConditionedGaitScheduler,
)


LEG_NAMES = ("FR", "FL", "RR", "RL")
LEG_JOINTS = ("hip", "thigh", "calf")
ARM_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch")
DEFAULT_ROWS = (0, 1, 2, 16, 17, 18)

LOWER_COLOR = "#2980b9"
UPPER_COLOR = "#c0392b"
INTERPOLATED_COLOR = "#27ae60"


def row_labels() -> tuple[str, ...]:
    joint_names = tuple(
        f"{leg}_{joint}"
        for leg in LEG_NAMES
        for joint in LEG_JOINTS
    ) + ARM_JOINTS
    return tuple(f"q: {name}" for name in joint_names) + tuple(
        f"dq: {name}" for name in joint_names
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gait",
        choices=sorted(HEIGHT_GAIT_PATHS),
        default="stance_hold",
        help="Height-conditioned gait bank (default: stance_hold)",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=0.225,
        help="Initial commanded base height in meters (default: 0.225)",
    )
    parser.add_argument(
        "--height-rate",
        type=float,
        default=0.0,
        help="Initial commanded base-height rate in m/s (default: 0)",
    )
    parser.add_argument(
        "--height-rate-limit",
        type=float,
        default=0.30,
        help="Symmetric height-rate slider limit in m/s (default: 0.30)",
    )
    parser.add_argument(
        "--phase-time",
        type=int,
        default=0,
        help="First phase index shown in the plot (default: 0)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=list(DEFAULT_ROWS),
        help=(
            "Reference rows to plot (default: 0 1 2 16 17 18; "
            "rows 0:16 are q and 16:32 are dq)"
        ),
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Reference sample period in seconds (default: 0.01)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Save the initial interpolation view as a PNG",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive window",
    )
    return parser.parse_args()


class GaitInterpolationVisualizer:
    def __init__(
        self,
        gait_name: str,
        height: float,
        height_rate: float,
        height_rate_limit: float,
        phase_time: int,
        rows: list[int],
        dt: float,
    ) -> None:
        self.scheduler = HeightConditionedGaitScheduler(
            HEIGHT_GAIT_PATHS[gait_name],
            name=gait_name,
            phase_time=phase_time,
        )
        self.gait_name = gait_name
        self.height_rate_limit = float(height_rate_limit)
        self.dt = float(dt)
        self.rows = np.asarray(rows, dtype=int)
        self.labels = row_labels()

        if self.dt <= 0.0:
            raise ValueError("--dt must be positive")
        if (
            not np.isfinite(self.height_rate_limit)
            or self.height_rate_limit <= 0.0
        ):
            raise ValueError("--height-rate-limit must be finite and positive")
        invalid_rows = self.rows[
            (self.rows < 0) | (self.rows >= self.scheduler.gaits.shape[1])
        ]
        if len(invalid_rows):
            raise IndexError(
                f"Rows out of range: {invalid_rows.tolist()}; "
                f"reference has {self.scheduler.gaits.shape[1]} rows"
            )

        self.indices = self.scheduler.indices.copy()
        self.time = np.arange(self.scheduler.phase_length) * self.dt
        self.height = float(
            np.clip(height, self.scheduler.min_height, self.scheduler.max_height)
        )
        self.height_rate = float(
            np.clip(
                height_rate,
                -self.height_rate_limit,
                self.height_rate_limit,
            )
        )

        ncols = 2
        nrows = (len(self.rows) + ncols - 1) // ncols
        self.figure, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(14, max(6.5, 2.8 * nrows)),
            sharex=True,
            squeeze=False,
        )
        self.axes = axes.ravel()
        self.figure.subplots_adjust(
            left=0.07,
            right=0.98,
            top=0.89,
            bottom=0.15,
            hspace=0.34,
            wspace=0.22,
        )

        self.lower_lines = []
        self.upper_lines = []
        self.interpolated_lines = []
        initial = np.zeros(self.scheduler.phase_length)
        for axis, row in zip(self.axes, self.rows):
            lower_line, = axis.plot(
                self.time,
                initial,
                color=LOWER_COLOR,
                linestyle="--",
                linewidth=1.2,
            )
            upper_line, = axis.plot(
                self.time,
                initial,
                color=UPPER_COLOR,
                linestyle="--",
                linewidth=1.2,
            )
            interpolated_line, = axis.plot(
                self.time,
                initial,
                color=INTERPOLATED_COLOR,
                linewidth=2.0,
            )
            axis.set_title(f"row {row}: {self.labels[row]}")
            axis.set_ylabel("rad" if row < 16 else "rad/s")
            axis.grid(True, alpha=0.3)
            self.lower_lines.append(lower_line)
            self.upper_lines.append(upper_line)
            self.interpolated_lines.append(interpolated_line)

        for axis in self.axes[len(self.rows):]:
            axis.set_visible(False)
        for axis in self.axes[:len(self.rows)]:
            axis.set_xlabel("cycle time [s]")

        height_axis = self.figure.add_axes((0.20, 0.075, 0.65, 0.025))
        rate_axis = self.figure.add_axes((0.20, 0.035, 0.65, 0.025))
        self.height_slider = Slider(
            height_axis,
            "height [m]",
            self.scheduler.min_height,
            self.scheduler.max_height,
            valinit=self.height,
            valstep=0.001,
        )
        self.height_rate_slider = Slider(
            rate_axis,
            "height rate [m/s]",
            -self.height_rate_limit,
            self.height_rate_limit,
            valinit=self.height_rate,
            valstep=0.005,
        )
        self.height_slider.on_changed(self._slider_changed)
        self.height_rate_slider.on_changed(self._slider_changed)
        self.update_plot()

    def interpolation_data(
        self,
        height: float,
        height_rate: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float]:
        lower, upper, alpha, _span = self.scheduler._bracket(height)
        lower_reference = self.scheduler.gaits[lower][:, self.indices]
        upper_reference = self.scheduler.gaits[upper][:, self.indices]
        interpolated = self.scheduler.get_reference(
            height,
            height_rate=height_rate,
            indices=self.indices,
        )
        return (
            lower_reference,
            upper_reference,
            interpolated,
            lower,
            upper,
            alpha,
        )

    def update_plot(self) -> None:
        (
            lower_reference,
            upper_reference,
            interpolated,
            lower,
            upper,
            alpha,
        ) = self.interpolation_data(self.height, self.height_rate)
        lower_height = self.scheduler.heights[lower]
        upper_height = self.scheduler.heights[upper]

        for index, row in enumerate(self.rows):
            lower_line = self.lower_lines[index]
            upper_line = self.upper_lines[index]
            interpolated_line = self.interpolated_lines[index]
            lower_line.set_ydata(lower_reference[row])
            upper_line.set_ydata(upper_reference[row])
            interpolated_line.set_ydata(interpolated[row])
            lower_line.set_label(f"lower bank: {lower_height:.6g} m")
            upper_line.set_label(f"upper bank: {upper_height:.6g} m")
            interpolated_line.set_label(
                f"controller result: alpha={alpha:.3f}"
            )
            axis = self.axes[index]
            axis.relim()
            axis.autoscale_view(scalex=False, scaley=True)

        self.axes[0].legend(fontsize=8, loc="best")
        self.figure.suptitle(
            f"{self.gait_name}: HeightConditionedGaitScheduler.get_reference()\n"
            f"height={self.height:.3f} m, rate={self.height_rate:+.3f} m/s, "
            f"bracket=[{lower_height:.6g}, {upper_height:.6g}] m, "
            f"alpha={alpha:.3f}"
        )
        self.figure.canvas.draw_idle()

    def _slider_changed(self, _value: float) -> None:
        self.height = float(self.height_slider.val)
        self.height_rate = float(self.height_rate_slider.val)
        self.update_plot()

    def print_report(self) -> None:
        (
            lower_reference,
            upper_reference,
            interpolated,
            lower,
            upper,
            alpha,
        ) = self.interpolation_data(self.height, self.height_rate)
        lower_height = self.scheduler.heights[lower]
        upper_height = self.scheduler.heights[upper]
        linear = (1.0 - alpha) * lower_reference + alpha * upper_reference
        morph = interpolated - linear
        print(f"Gait: {self.gait_name}")
        print(f"Bank heights [m]: {self.scheduler.heights}")
        print(
            f"Requested: height={self.height:.4f} m, "
            f"height_rate={self.height_rate:+.4f} m/s"
        )
        print(
            f"Bracket: lower={lower_height:.6g} m, "
            f"upper={upper_height:.6g} m, alpha={alpha:.6f}"
        )
        print(
            f"Reference shape: {interpolated.shape}, "
            f"phase_time={self.scheduler.phase_time}"
        )
        print(
            "Maximum velocity morph term: "
            f"{np.max(np.abs(morph[self.scheduler.position_dim:])):.6g} rad/s"
        )
        print(f"Displayed rows: {self.rows.tolist()}")


def main() -> None:
    args = parse_args()
    visualizer = GaitInterpolationVisualizer(
        gait_name=args.gait,
        height=args.height,
        height_rate=args.height_rate,
        height_rate_limit=args.height_rate_limit,
        phase_time=args.phase_time,
        rows=args.rows,
        dt=args.dt,
    )
    visualizer.print_report()
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        visualizer.figure.savefig(args.save, dpi=180, bbox_inches="tight")
        print(f"Saved visualization: {args.save}")
    if args.no_show:
        plt.close(visualizer.figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
