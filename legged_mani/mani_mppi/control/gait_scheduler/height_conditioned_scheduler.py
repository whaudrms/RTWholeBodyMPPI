"""Phase-aligned interpolation over a bank of gait reference heights."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np


class HeightConditionedGaitScheduler:
    """Interpolate phase-compatible ``[q, dq]`` gait files by base height."""

    def __init__(
        self,
        gait_paths: Mapping[float, str | Path],
        name: str = "height_conditioned",
        phase_time: int = 0,
    ) -> None:
        if len(gait_paths) < 2:
            raise ValueError("A height-conditioned gait requires at least two levels")

        levels = sorted((float(height), Path(path)) for height, path in gait_paths.items())
        self.heights = np.asarray([height for height, _ in levels], dtype=float)
        if not np.isfinite(self.heights).all() or np.any(self.heights <= 0.0):
            raise ValueError("Gait-bank heights must be finite and positive")
        if np.any(np.diff(self.heights) <= 0.0):
            raise ValueError("Gait-bank heights must be unique")

        loaded = [np.loadtxt(path, delimiter="\t", comments="#") for _, path in levels]
        shape = loaded[0].shape
        if len(shape) != 2 or shape[0] % 2:
            raise ValueError(f"Gait references must contain matching q/dq rows, got {shape}")
        for gait in loaded:
            if gait.shape != shape:
                raise ValueError(
                    "All height levels must be phase compatible; "
                    f"expected {shape}, got {gait.shape}"
                )
            if not np.isfinite(gait).all():
                raise ValueError("Gait references must contain only finite values")

        self.gaits = np.stack(loaded, axis=0)
        self.position_dim = shape[0] // 2
        self.phase_length = shape[1]
        self.phase_time = int(phase_time) % self.phase_length
        self.indices = (
            self.phase_time + np.arange(self.phase_length)
        ) % self.phase_length
        self.type = name

    @property
    def min_height(self) -> float:
        return float(self.heights[0])

    @property
    def max_height(self) -> float:
        return float(self.heights[-1])

    def _bracket(self, height: float) -> tuple[int, int, float, float]:
        height = float(height)
        if not np.isfinite(height):
            raise ValueError("height must be finite")
        if height <= self.heights[0]:
            return 0, 0, 0.0, 0.0
        if height >= self.heights[-1]:
            last = len(self.heights) - 1
            return last, last, 0.0, 0.0

        upper = int(np.searchsorted(self.heights, height, side="right"))
        lower = upper - 1
        span = self.heights[upper] - self.heights[lower]
        alpha = (height - self.heights[lower]) / span
        return lower, upper, float(alpha), float(span)

    def get_reference(
        self,
        height: float,
        *,
        height_rate: float = 0.0,
        horizon: int | None = None,
        indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return one phase-aligned gait horizon at ``height``.

        The velocity reference includes the morphing term caused by a changing
        height, in addition to the phase velocities stored in the gait files.
        """
        height_rate = float(height_rate)
        if not np.isfinite(height_rate):
            raise ValueError("height_rate must be finite")
        if indices is None:
            if horizon is None:
                horizon = self.phase_length
            horizon = int(horizon)
            if horizon < 1:
                raise ValueError("horizon must be positive")
            indices = (
                self.phase_time + np.arange(horizon)
            ) % self.phase_length
        else:
            indices = np.asarray(indices, dtype=int)
            if indices.ndim != 1 or len(indices) < 1:
                raise ValueError("indices must be a non-empty vector")
            indices = indices % self.phase_length

        lower, upper, alpha, span = self._bracket(height)
        lower_ref = self.gaits[lower][:, indices]
        if lower == upper:
            return lower_ref.copy()

        upper_ref = self.gaits[upper][:, indices]
        reference = (1.0 - alpha) * lower_ref + alpha * upper_ref
        alpha_rate = height_rate / span
        reference[self.position_dim:] += (
            alpha_rate
            * (upper_ref[:self.position_dim] - lower_ref[:self.position_dim])
        )
        return reference

    def roll(self) -> None:
        self.phase_time = (self.phase_time + 1) % self.phase_length
        self.indices = (
            self.phase_time + np.arange(self.phase_length)
        ) % self.phase_length

    def get_current_ref(
        self,
        height: float,
        *,
        height_rate: float = 0.0,
    ) -> np.ndarray:
        return self.get_reference(
            height,
            height_rate=height_rate,
            horizon=1,
        )[:, 0]
