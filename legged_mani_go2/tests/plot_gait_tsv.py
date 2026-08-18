"""Plot every row of a gait TSV as an individual matplotlib subplot.

Examples
--------
Plot the converted B2-Z1 in-place gait:

    python plot_gait_tsv.py \
        ../mani_mppi/control/gait_scheduler/gaits/FAST/b2_z1_in_place_FAST_0_0_10cm_100hz.tsv

Plot only selected rows and save the figures:

    python plot_gait_tsv.py gait.tsv --rows 0 1 2 18 19 20 --save-dir ./gait_plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LEG_NAMES = ("FR", "FL", "RR", "RL")
LEG_JOINTS = ("hip", "thigh", "calf")
ARM_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch")


def row_labels(n_rows: int) -> list[str]:
    """Return useful labels for common Go1/B2-Z1 gait layouts."""
    if n_rows == 32:
        position_labels = [
            f"{leg}_{joint}"
            for leg in LEG_NAMES
            for joint in LEG_JOINTS
        ] + list(ARM_JOINTS)
        return [f"q: {label}" for label in position_labels] + [
            f"dq: {label}" for label in position_labels
        ]

    if n_rows == 24:
        labels = [
            f"{leg}_{joint}"
            for leg in LEG_NAMES
            for joint in LEG_JOINTS
        ]
        # Legacy Go1 files contain positions and velocities in two 12-row blocks.
        return [f"q: {label}" for label in labels] + [
            f"dq: {label}" for label in labels
        ]

    return [f"row {row:02d}" for row in range(n_rows)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot each row of a tab-separated gait reference file."
    )
    parser.add_argument("tsv", type=Path, help="Path to the gait TSV file")
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        help="Optional zero-based row indices to plot; defaults to every row",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Sample period in seconds (default: 0.01 for 100 Hz)",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="Save PNG figures here instead of displaying them",
    )
    return parser.parse_args()


def plot_rows(
    data: np.ndarray,
    labels: list[str],
    rows: list[int],
    dt: float,
    title: str,
    output: Path | None = None,
) -> None:
    ncols = 2
    nrows = (len(rows) + ncols - 1) // ncols
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15, max(3.0, 2.1 * nrows)),
        sharex=True,
        squeeze=False,
    )
    axes = axes.ravel()
    time = np.arange(data.shape[1]) * dt

    for axis, row in zip(axes, rows):
        axis.plot(time, data[row], linewidth=1.2)
        axis.set_title(f"row {row}: {labels[row]}")
        axis.set_ylabel("value")
        axis.grid(True, alpha=0.3)

    for axis in axes[len(rows) :]:
        axis.set_visible(False)

    axes[0].figure.suptitle(title)
    axes[-1].set_xlabel("time [s]")
    figure.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved: {output}")
        plt.close(figure)
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    if args.dt <= 0.0:
        raise ValueError("--dt must be positive")
    if not args.tsv.is_file():
        raise FileNotFoundError(args.tsv)

    data = np.loadtxt(args.tsv, delimiter="\t", comments="#")
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected a 2-D TSV with at least two columns, got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError("The TSV contains NaN or infinite values")

    labels = row_labels(data.shape[0])
    rows = list(range(data.shape[0])) if args.rows is None else args.rows
    invalid = [row for row in rows if row < 0 or row >= data.shape[0]]
    if invalid:
        raise IndexError(f"Row indices out of range: {invalid}; file has {data.shape[0]} rows")

    print(f"Loaded {args.tsv}: rows={data.shape[0]}, samples={data.shape[1]}")
    print(f"Plotting rows: {rows}")

    if args.save_dir is not None:
        plot_rows(
            data,
            labels,
            rows,
            args.dt,
            f"Gait rows: {args.tsv.name}",
            args.save_dir / "gait_rows.png",
        )
        if data.shape[0] == 36 and args.rows is None:
            plot_rows(
                data,
                labels,
                list(range(18)),
                args.dt,
                f"Joint positions: {args.tsv.name}",
                args.save_dir / "gait_positions.png",
            )
            plot_rows(
                data,
                labels,
                list(range(18, 36)),
                args.dt,
                f"Joint velocities: {args.tsv.name}",
                args.save_dir / "gait_velocities.png",
            )
    else:
        plot_rows(data, labels, rows, args.dt, f"Gait rows: {args.tsv.name}")


if __name__ == "__main__":
    main()
