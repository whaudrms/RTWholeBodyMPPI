"""Fast capsule-to-torso clearance for the B2-Z1 arm."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class CollisionResult:
    """Per-state arm clearance and hard-validity results."""

    clearance: np.ndarray
    segment_valid: np.ndarray
    exact_evaluations: int


class ArmTorsoCollision:
    """Evaluate three arm capsules against a torso-oriented box.

    Uniform centerline samples provide lower and upper clearance bounds. The
    soft cost uses their midpoint, while exact segment-to-box distance is only
    evaluated when the bounds straddle the hard-clearance threshold.
    """

    def __init__(self, model: mujoco.MjModel, config: dict) -> None:
        self.enabled = bool(config.get("collision_enabled", True))
        self.fast_samples = int(config.get("collision_fast_samples", 9))
        if self.fast_samples < 2:
            raise ValueError("collision_fast_samples must be at least 2")
        self.fast_alpha = np.linspace(
            0.0, 1.0, self.fast_samples, dtype=float
        )

        self.capsule_radii = np.asarray(
            config.get("collision_capsule_radii", [0.030, 0.030, 0.0375]),
            dtype=float,
        )
        if self.capsule_radii.shape != (3,):
            raise ValueError("collision_capsule_radii must contain 3 values")
        if np.any(self.capsule_radii <= 0.0):
            raise ValueError("collision capsule radii must be positive")

        self.shoulder_exclusion = float(
            config.get("collision_shoulder_exclusion", 0.07)
        )
        self.safe_distance = float(
            config.get("collision_safe_distance", 0.05)
        )
        self.hard_distance = float(
            config.get("collision_hard_distance", 0.005)
        )
        self.soft_weight = float(
            config.get("collision_soft_weight", 10000.0)
        )
        if self.shoulder_exclusion < 0.0:
            raise ValueError("collision_shoulder_exclusion must be non-negative")
        if self.hard_distance >= self.safe_distance:
            raise ValueError(
                "collision_hard_distance must be smaller than "
                "collision_safe_distance"
            )
        if self.soft_weight < 0.0:
            raise ValueError("collision_soft_weight must be non-negative")

        geom_name = config.get("collision_body_geom", "base_collision")
        self.body_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
        )
        if self.body_geom_id < 0:
            raise ValueError(f"Unknown collision body geom: {geom_name}")
        if model.geom_type[self.body_geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError("collision_body_geom must be a box geom")

        self.body_half_size = model.geom_size[self.body_geom_id, :3].copy()
        self.body_local_pos = model.geom_pos[self.body_geom_id].copy()
        local_mat = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(local_mat, model.geom_quat[self.body_geom_id])
        self.body_local_mat = local_mat.reshape(3, 3)

        self.rollout_contact_filter_enabled = bool(
            config.get("disable_arm_torso_rollout_contact", False)
        )
        self.rollout_arm_geom_ids = np.empty(0, dtype=int)
        if self.rollout_contact_filter_enabled:
            if not self.enabled:
                raise ValueError(
                    "Arm-to-torso rollout contact cannot be disabled when "
                    "the analytic collision constraint is disabled"
                )
            arm_root_body = config.get(
                "rollout_arm_root_body", "link00"
            )
            self.rollout_arm_geom_ids = self._disable_rollout_contact(
                model,
                arm_root_body=arm_root_body,
            )

    @staticmethod
    def _unused_collision_bits(model: mujoco.MjModel) -> tuple[int, int]:
        """Return two positive collision bits not used by the model."""
        used = 0
        for value in model.geom_contype:
            used |= int(value)
        for value in model.geom_conaffinity:
            used |= int(value)
        free = [
            1 << index
            for index in range(1, 30)
            if not used & (1 << index)
        ]
        if len(free) < 2:
            raise ValueError("Two free MuJoCo collision bits are required")
        return free[0], free[1]

    @staticmethod
    def _body_subtree_ids(
        model: mujoco.MjModel,
        root_body_id: int,
    ) -> np.ndarray:
        """Return the root body and all of its descendants."""
        in_subtree = np.zeros(model.nbody, dtype=bool)
        in_subtree[root_body_id] = True
        for body_id in range(root_body_id + 1, model.nbody):
            in_subtree[body_id] = in_subtree[
                int(model.body_parentid[body_id])
            ]
        return np.flatnonzero(in_subtree)

    def _disable_rollout_contact(
        self,
        model: mujoco.MjModel,
        arm_root_body: str,
    ) -> np.ndarray:
        """Disable only arm-to-analytic-torso contacts in this MjModel.

        The arm and torso receive separate collision type bits. Their original
        affinity is retained so both groups still collide with external geoms,
        including the push box and floor. Each group also accepts its own new
        type bit so arm self-contact and torso self-contact remain enabled.
        """
        arm_root_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, arm_root_body
        )
        if arm_root_id < 0:
            raise ValueError(f"Unknown rollout arm root body: {arm_root_body}")

        arm_body_ids = self._body_subtree_ids(model, arm_root_id)
        arm_geom_mask = np.isin(model.geom_bodyid, arm_body_ids)
        active_geom_mask = (
            (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        )
        arm_geom_ids = np.flatnonzero(arm_geom_mask & active_geom_mask)
        if not len(arm_geom_ids):
            raise ValueError(
                f"No collision geoms found below arm body '{arm_root_body}'"
            )

        arm_bit, torso_bit = self._unused_collision_bits(model)
        model.geom_contype[arm_geom_ids] = arm_bit
        model.geom_conaffinity[arm_geom_ids] |= arm_bit
        model.geom_contype[self.body_geom_id] = torso_bit
        model.geom_conaffinity[self.body_geom_id] |= torso_bit
        return arm_geom_ids

    @staticmethod
    def quaternion_rotation_matrices(quaternions: np.ndarray) -> np.ndarray:
        """Return world-from-local matrices for MuJoCo [w, x, y, z]."""
        quaternions = np.asarray(quaternions, dtype=float)
        norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
        q = quaternions / np.maximum(norms, 1e-12)
        w, x, y, z = q.T
        matrices = np.empty((len(q), 3, 3), dtype=float)
        matrices[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
        matrices[:, 0, 1] = 2.0 * (x * y - z * w)
        matrices[:, 0, 2] = 2.0 * (x * z + y * w)
        matrices[:, 1, 0] = 2.0 * (x * y + z * w)
        matrices[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
        matrices[:, 1, 2] = 2.0 * (y * z - x * w)
        matrices[:, 2, 0] = 2.0 * (x * z - y * w)
        matrices[:, 2, 1] = 2.0 * (y * z + x * w)
        matrices[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
        return matrices

    @staticmethod
    def segment_aabb_distance(
        starts: np.ndarray,
        ends: np.ndarray,
        half_size: np.ndarray,
    ) -> np.ndarray:
        """Return exact Euclidean distance between segments and an AABB."""
        starts = np.asarray(starts, dtype=float)
        ends = np.asarray(ends, dtype=float)
        half_size = np.asarray(half_size, dtype=float)
        original_shape = starts.shape[:-1]
        a = starts.reshape(-1, 3)
        direction = (ends - starts).reshape(-1, 3)
        count = len(a)

        boundaries = np.stack((-half_size, half_size), axis=0)
        numerator = boundaries[None, :, :] - a[:, None, :]
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
                (
                    np.zeros((count, 1)),
                    crossings,
                    np.ones((count, 1)),
                ),
                axis=1,
            ),
            axis=1,
        )
        lower = knots[:, :-1]
        upper = knots[:, 1:]
        midpoint = 0.5 * (lower + upper)
        midpoint_position = (
            a[:, None, :] + midpoint[:, :, None] * direction[:, None, :]
        )
        signs = np.where(
            midpoint_position > half_size,
            1.0,
            np.where(midpoint_position < -half_size, -1.0, 0.0),
        )
        active = signs != 0.0
        offset = a[:, None, :] - signs * half_size
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

        candidate_positions = (
            a[:, None, :] + stationary[:, :, None] * direction[:, None, :]
        )
        outside = np.maximum(np.abs(candidate_positions) - half_size, 0.0)
        minimum_squared_distance = np.min(
            np.sum(outside * outside, axis=2), axis=1
        )
        return np.sqrt(minimum_squared_distance).reshape(original_shape)

    def evaluate(
        self,
        flat_states: np.ndarray,
        arm_positions: np.ndarray,
    ) -> CollisionResult:
        """Evaluate soft clearance and exact hard validity for three links."""
        flat_states = np.asarray(flat_states, dtype=float)
        arm_positions = np.asarray(arm_positions, dtype=float)
        expected_shape = (len(flat_states), 4, 3)
        if arm_positions.shape != expected_shape:
            raise ValueError(
                f"arm_positions must have shape {expected_shape}, got "
                f"{arm_positions.shape}"
            )

        starts_world = arm_positions[:, :3].copy()
        ends_world = arm_positions[:, 1:]
        upper_vector = ends_world[:, 0] - starts_world[:, 0]
        upper_length = np.linalg.norm(upper_vector, axis=1)
        trim = np.minimum(self.shoulder_exclusion, upper_length)
        starts_world[:, 0] += (
            upper_vector
            * (trim / np.maximum(upper_length, 1e-12))[:, None]
        )

        base_position = flat_states[:, :3]
        base_rotation = self.quaternion_rotation_matrices(flat_states[:, 3:7])
        endpoints_world = np.concatenate((starts_world, ends_world), axis=1)
        relative_world = endpoints_world - base_position[:, None, :]
        endpoints_body = np.einsum(
            "mij,mni->mnj", base_rotation, relative_world
        )
        endpoints_geom = np.einsum(
            "mni,ij->mnj",
            endpoints_body - self.body_local_pos,
            self.body_local_mat,
        )
        starts_geom = endpoints_geom[:, :3]
        ends_geom = endpoints_geom[:, 3:]
        radii = np.broadcast_to(
            self.capsule_radii[None, :], starts_geom.shape[:2]
        )

        segment_vector = ends_geom - starts_geom
        samples_geom = (
            starts_geom[:, :, None, :]
            + self.fast_alpha[None, None, :, None]
            * segment_vector[:, :, None, :]
        )
        outside = np.maximum(
            np.abs(samples_geom) - self.body_half_size, 0.0
        )
        sampled_distance = np.min(
            np.linalg.norm(outside, axis=-1), axis=2
        )
        half_sample_spacing = np.linalg.norm(segment_vector, axis=2) / (
            2.0 * (self.fast_samples - 1)
        )
        sampled_clearance = sampled_distance - radii
        lower_clearance = (
            np.maximum(sampled_distance - half_sample_spacing, 0.0) - radii
        )

        definitely_invalid = sampled_clearance < self.hard_distance
        definitely_valid = lower_clearance >= self.hard_distance
        refine = ~(definitely_invalid | definitely_valid)

        clearance = 0.5 * (lower_clearance + sampled_clearance)
        segment_valid = ~definitely_invalid
        exact_evaluations = int(np.count_nonzero(refine))
        if exact_evaluations:
            exact_clearance = self.segment_aabb_distance(
                starts_geom[refine],
                ends_geom[refine],
                self.body_half_size,
            ) - radii[refine]
            clearance[refine] = exact_clearance
            segment_valid[refine] = exact_clearance >= self.hard_distance

        return CollisionResult(
            clearance=clearance,
            segment_valid=segment_valid,
            exact_evaluations=exact_evaluations,
        )
