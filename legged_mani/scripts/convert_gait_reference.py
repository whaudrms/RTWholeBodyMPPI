"""Pad legacy Go1 gait TSVs to the B2-Z1 scheduler dimension.

No kinematic retargeting, IK, scaling, or angle transformation is performed.
The source leg rows are preserved exactly. Four fixed Z1 arm-position rows and
four zero arm-velocity rows are appended so the result has 32 rows::

    12 leg q + 4 arm q + 12 leg dq + 4 arm dq

Input files are expected below ``gaits/<speed>/go1``. The output keeps the
same filename and is written below ``gaits/<speed>/b2_z1``.

Examples
--------
    python convert_gait_reference.py
    python convert_gait_reference.py --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


GAITS_DIR = (
    Path(__file__).resolve().parents[1]
    / "mani_mppi"
    / "control"
    / "gait_scheduler"
    / "gaits"
)
DEFAULT_SOURCE = (
    GAITS_DIR
    / "FAST"
    / "go1"
    / "walking_gait_raibert_FAST_0_0_10cm_100hz.tsv"
)

# Used only to pad the four extra B2-Z1 arm position rows.
DEFAULT_ARM_POSITION = np.array([0.0, 1.0, -0.6, 0.0], dtype=float)


def default_output(source: Path) -> Path:
    """Map ``gaits/<speed>/go1/file.tsv`` to ``gaits/<speed>/b2_z1/file.tsv``."""
    source = source.resolve()
    if source.parent.name != "go1" or source.parent.parent.parent != GAITS_DIR:
        raise ValueError(
            "When --output is omitted, source must be inside "
            "gaits/<speed>/go1/<file>.tsv"
        )
    return source.parent.parent / "b2_z1" / source.name


def convert(
    source: Path,
    output: Path | None = None,
    arm_position: np.ndarray = DEFAULT_ARM_POSITION,
) -> np.ndarray:
    """Convert one 24-row Go1 gait by dimension padding only."""
    source = source.resolve()
    output = default_output(source) if output is None else output.resolve()

    source_data = np.loadtxt(source, delimiter="\t", comments="#")
    if source_data.ndim != 2 or source_data.shape[0] != 24:
        raise ValueError(
            f"Expected a 24-row Go1 gait [12 q + 12 dq], got {source_data.shape}"
        )
    if source_data.shape[1] < 2 or not np.isfinite(source_data).all():
        raise ValueError("Source gait must contain at least two finite samples")

    arm_position = np.asarray(arm_position, dtype=float)
    if arm_position.shape != (4,) or not np.isfinite(arm_position).all():
        raise ValueError("arm_position must contain four finite values")

    n_samples = source_data.shape[1]
    arm_q = np.repeat(arm_position[:, None], n_samples, axis=1)
    arm_dq = np.zeros((4, n_samples), dtype=float)

    # Keep the original leg positions and velocities unchanged.
    output_data = np.vstack(
        (
            source_data[:12],
            arm_q,
            source_data[12:24],
            arm_dq,
        )
    )
    if output_data.shape != (32, n_samples):
        raise RuntimeError(f"Unexpected output shape: {output_data.shape}")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output,
        output_data,
        delimiter="\t",
        # Round-trip precision keeps the copied source rows numerically exact.
        fmt="%.17g",
        header=(
            "B2-Z1 dimension-padded gait; source leg rows preserved; "
            "rows: 12 leg q, 4 arm q, 12 leg dq, 4 arm dq"
        ),
    )
    return output_data


def convert_all() -> list[Path]:
    """Convert every TSV below ``gaits/*/go1`` into a matching b2_z1 folder."""
    outputs = []
    for source in sorted(GAITS_DIR.glob("*/go1/*.tsv")):
        output = default_output(source)
        convert(source, output)
        outputs.append(output)
        print(f"wrote: {output}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="One source TSV; output defaults to the matching b2_z1 folder",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Explicit output path",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert every TSV in gaits/*/go1/",
    )
    args = parser.parse_args()

    if args.all and args.output is not None:
        parser.error("--all cannot be combined with --output")

    if args.all:
        outputs = convert_all()
        print(f"converted {len(outputs)} gait files")
        return

    converted = convert(args.source, args.output)
    output = default_output(args.source) if args.output is None else args.output.resolve()
    print(f"wrote: {output}")
    print(f"shape: {converted.shape}")
    print(f"leg q range: [{converted[:12].min():.4f}, {converted[:12].max():.4f}]")
    print(f"leg dq abs max: {np.abs(converted[16:28]).max():.4f}")


if __name__ == "__main__":
    main()
