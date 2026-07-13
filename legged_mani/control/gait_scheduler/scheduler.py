"""Cyclic 32-row B2-Z1 joint reference scheduler."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


class JointReferenceScheduler:
    """Schedule ``[16 joint positions, 16 joint velocities]`` references."""

    POSITION_DIM = 16
    VELOCITY_DIM = 16
    REFERENCE_DIM = POSITION_DIM + VELOCITY_DIM

    def __init__(self, reference: np.ndarray, name: str = "in_place", phase: int = 0):
        reference = np.asarray(reference, dtype=float)
        if reference.ndim != 2 or reference.shape[0] != self.REFERENCE_DIM:
            raise ValueError(
                "Joint reference must have shape (32, cycle_steps), "
                f"got {reference.shape}"
            )
        if reference.shape[1] < 1 or not np.isfinite(reference).all():
            raise ValueError("Joint reference must be non-empty and finite")
        self.reference = reference.copy()
        self.name = name
        self.phase = int(phase) % self.cycle_steps

    @property
    def cycle_steps(self) -> int:
        return self.reference.shape[1]

    # Compatibility names used by the original mppi_locomotion.py.
    @property
    def gait(self) -> np.ndarray:
        return self.reference

    @property
    def indices(self) -> np.ndarray:
        return (self.phase + np.arange(self.cycle_steps)) % self.cycle_steps

    @classmethod
    def from_tsv(
        cls, path: str | Path, name: str = "gait", phase: int = 0
    ) -> "JointReferenceScheduler":
        reference = np.loadtxt(Path(path), delimiter="\t")
        return cls(reference, name=name, phase=phase)

    @classmethod
    def from_keyframe(
        cls,
        model: mujoco.MjModel,
        keyframe: str = "stand",
        cycle_steps: int = 100,
        name: str = "in_place",
    ) -> "JointReferenceScheduler":
        if cycle_steps < 1:
            raise ValueError("cycle_steps must be at least 1")
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if key_id < 0:
            raise ValueError(f"Unknown joint-reference keyframe: {keyframe}")
        joint_position = model.key_ctrl[key_id].copy()
        if joint_position.shape != (cls.POSITION_DIM,):
            raise ValueError(
                f"Expected 16 keyframe controls, got {joint_position.shape}"
            )
        one_step = np.concatenate((joint_position, np.zeros(cls.VELOCITY_DIM)))
        reference = np.repeat(one_step[:, None], cycle_steps, axis=1)
        return cls(reference, name=name)

    def horizon(self, length: int) -> np.ndarray:
        """Return a wrapped reference with shape ``(32, length)``."""
        if length < 1:
            raise ValueError("Reference horizon length must be at least 1")
        indices = (self.phase + np.arange(length)) % self.cycle_steps
        return self.reference[:, indices]

    def advance(self, steps: int = 1) -> None:
        if steps < 0:
            raise ValueError("Reference advance must be non-negative")
        self.phase = (self.phase + steps) % self.cycle_steps

    def roll(self) -> None:
        """Original scheduler API: advance one control step."""
        self.advance(1)

    def get_current_ref(self) -> np.ndarray:
        return self.reference[:, self.phase]

    def reset(self, phase: int = 0) -> None:
        self.phase = int(phase) % self.cycle_steps


# Original controller name retained for source-compatible imports.
GaitScheduler = JointReferenceScheduler


class Timer:
    """Small phase timer retained from the original gait scheduler."""

    def __init__(self, init_time: int = 0, end_time: int = 300):
        self.elapsed_time = init_time
        self.end_time = end_time
        self.done = False
        self.waiting = False

    def increment(self) -> None:
        if self.elapsed_time < self.end_time:
            self.elapsed_time += 1
        else:
            self.done = True

    def reset(self) -> None:
        self.elapsed_time = 0
        self.done = False
