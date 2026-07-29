"""Shared gait, arm-kinematics, and collision support for B2-Z1 MPPI."""

from __future__ import annotations

import os

import numpy as np

from mani_mppi.control.collision import ArmTorsoCollision
from mani_mppi.control.controllers.base_controller import BaseMPPI
from mani_mppi.control.kinematics import ArmKinematics


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GAIT_DIR = os.path.join(BASE_DIR, "../gait_scheduler/gaits/")
GAIT_PATHS = {
    "in_place": os.path.join(
        GAIT_DIR,
        "FAST/b2_retargeted/walking_gait_raibert_FAST_0_0_10cm_100hz.tsv",
    ),
    "trot": os.path.join(
        GAIT_DIR,
        "MED/b2_retargeted/walking_gait_raibert_MED_0_5_15cm_100hz.tsv",
    ),
    "walk": os.path.join(
        GAIT_DIR,
        "MED/b2_retargeted/walking_gait_raibert_MED_0_1_10cm_100hz.tsv",
    ),
    "walk_fast": os.path.join(
        GAIT_DIR,
        "FAST/b2_retargeted/walking_gait_raibert_FAST_0_1_10cm_100hz.tsv",
    ),
    "stance_hold": os.path.join(
        GAIT_DIR,
        "FAST/b2_height_conditioned/h_0p543542/stance_hold.tsv",
    ),
}
HEIGHT_GAIT_DIR = os.path.join(GAIT_DIR, "FAST/b2_height_conditioned")
HEIGHT_GAIT_LEVELS = {
    0.543542: "h_0p543542",
    0.45: "h_0p450",
    0.35: "h_0p350",
}
HEIGHT_GAIT_PATHS = {
    gait_name: {
        height: os.path.join(HEIGHT_GAIT_DIR, directory, f"{gait_name}.tsv")
        for height, directory in HEIGHT_GAIT_LEVELS.items()
    }
    for gait_name in ("in_place", "walk_fast", "stance_hold")
}


class WholeBodyArmMPPI(BaseMPPI):
    """Common mechanics for locomani and push-box task controllers.

    Task controllers intentionally retain their own ``update`` and cost
    functions so the MPPI flow remains visible where each task is defined.
    """

    def _configure_rollout_mode(self, rollout_mode) -> None:
        """Select one of the shared gait/action warm-start samplers."""
        rollout_modes = (
            None,
            "noise_spline",
            "original",
            "original_spline",
            "safe_spline",
        )
        if rollout_mode not in rollout_modes:
            raise ValueError(
                "rollout_mode must be None, 'noise_spline', 'original', "
                "'original_spline', or 'safe_spline'"
            )
        if rollout_mode == "noise_spline":
            self.nominal_from_gait = True
            self.sample_type = "cubic"
        elif rollout_mode == "original":
            self.nominal_from_gait = False
            self.sample_type = "cubic_original"
        elif rollout_mode == "original_spline":
            self.nominal_from_gait = True
            self.sample_type = "cubic_gait_absolute"
        elif rollout_mode == "safe_spline":
            self.nominal_from_gait = True
            self.sample_type = "cubic_gait_residual"
        self.rollout_mode = rollout_mode or "configured"

    def _rollout_spline_indices(self):
        """Return unique, horizon-aligned knot indices for spline modes."""
        indices = np.rint(
            np.linspace(0, self.horizon - 1, self.n_knots)
        ).astype(int)
        if len(indices) < 2:
            raise ValueError("n_knots must be at least 2 for spline sampling")
        if len(np.unique(indices)) != len(indices):
            raise ValueError(
                "n_knots must not produce duplicate horizon indices"
            )
        return indices

    def _sample_gait_absolute_spline(self):
        """Spline the complete gait/IK/residual warm start plus knot noise."""
        from scipy.interpolate import CubicSpline

        indices = self._rollout_spline_indices()
        noise = self.generate_noise(
            (self.n_samples, len(indices), self.act_dim)
        )
        knot_actions = self.trajectory[indices][None, :, :] + noise
        actions = CubicSpline(indices, knot_actions, axis=1)(
            np.arange(self.horizon)
        )
        return np.clip(actions, self.act_min, self.act_max)

    def _sample_gait_residual_spline(self):
        """Preserve gait/IK and spline only residuals plus knot noise."""
        from scipy.interpolate import CubicSpline

        indices = self._rollout_spline_indices()
        noise = self.generate_noise(
            (self.n_samples, len(indices), self.act_dim)
        )
        knot_corrections = self.gait_correction[indices][None, :, :] + noise
        smooth_corrections = CubicSpline(
            indices, knot_corrections, axis=1
        )(np.arange(self.horizon))
        actions = self.gait_nominal[None, :, :] + smooth_corrections
        return np.clip(actions, self.act_min, self.act_max)

    def perturb_action(self):
        """Generate candidates according to the selected rollout mode."""
        if getattr(self, "rollout_mode", None) == "original_spline":
            return self._sample_gait_absolute_spline()
        if getattr(self, "rollout_mode", None) == "safe_spline":
            return self._sample_gait_residual_spline()
        return super().perturb_action()

    def _noise_free_rollout_candidate(self):
        """Return a mode-consistent baseline for collision-safe sampling."""
        from scipy.interpolate import CubicSpline

        mode = getattr(self, "rollout_mode", None)
        if mode not in {"original_spline", "safe_spline"}:
            return np.clip(self.trajectory, self.act_min, self.act_max)

        indices = self._rollout_spline_indices()
        if mode == "original_spline":
            candidate = CubicSpline(
                indices, self.trajectory[indices], axis=0
            )(np.arange(self.horizon))
        else:
            smooth_correction = CubicSpline(
                indices, self.gait_correction[indices], axis=0
            )(np.arange(self.horizon))
            candidate = self.gait_nominal + smooth_correction
        return np.clip(candidate, self.act_min, self.act_max)

    def _compact_robot_state_error(self, states, references):
        """Return the 44-D robot error without a dummy quaternion slot."""
        raw_error = states - references
        body_rp = self._quat_to_roll_pitch(states[:, 3:7])
        body_rp_ref = self._quat_to_roll_pitch(references[:, 3:7])
        body_rpy_error = np.column_stack(
            (
                body_rp - body_rp_ref,
                np.zeros(len(states), dtype=float),
            )
        )
        return np.concatenate(
            (
                raw_error[:, :3],
                body_rpy_error,
                raw_error[:, 7:],
            ),
            axis=1,
        )

    def _configure_arm_system(
        self,
        params: dict,
        ee_site_name: str = "gripper_center",
    ) -> None:
        """Initialize shared arm kinematics, sensors, and torso collision."""
        self.ee_site_name = ee_site_name
        self.arm_kinematics = ArmKinematics(
            self.model, params, ee_site_name=ee_site_name
        )
        self.arm_collision = ArmTorsoCollision(self.model, params)
        self.enable_rollout_sensors()

        # Keep the established controller attributes available to task code,
        # diagnostics, and existing callers while ownership lives in helpers.
        self.ee_site_id = self.arm_kinematics.ee_site_id
        self.arm_qpos_indices = self.arm_kinematics.arm_qpos_indices
        self.arm_dof_indices = self.arm_kinematics.arm_dof_indices
        self.arm_fk_body_ids = self.arm_kinematics.arm_fk_body_ids
        self.arm_fk_sensor_adrs = self.arm_kinematics.arm_fk_sensor_adrs
        self.ee_pos_sensor_adr = self.arm_kinematics.ee_pos_sensor_adr
        self.ee_quat_sensor_adr = self.arm_kinematics.ee_quat_sensor_adr
        self.ik_data = self.arm_kinematics.ik_data
        self.ee_data = self.arm_kinematics.fk_data
        self.arm_ik_damping = self.arm_kinematics.damping
        self.arm_ik_step_size = self.arm_kinematics.step_size
        self.arm_ik_max_iterations = self.arm_kinematics.max_iterations
        self.arm_ik_tolerance = self.arm_kinematics.tolerance
        self.arm_ik_smoothing = self.arm_kinematics.smoothing
        self.arm_ik_max_step = self.arm_kinematics.max_step

        self.collision_enabled = self.arm_collision.enabled
        self.collision_fast_samples = self.arm_collision.fast_samples
        self.collision_fast_alpha = self.arm_collision.fast_alpha
        self.collision_capsule_radii = self.arm_collision.capsule_radii
        self.collision_shoulder_exclusion = (
            self.arm_collision.shoulder_exclusion
        )
        self.collision_safe_distance = self.arm_collision.safe_distance
        self.collision_hard_distance = self.arm_collision.hard_distance
        self.collision_soft_weight = self.arm_collision.soft_weight
        self.collision_body_geom_id = self.arm_collision.body_geom_id
        self.collision_body_half_size = self.arm_collision.body_half_size
        self.collision_body_local_pos = self.arm_collision.body_local_pos
        self.collision_body_local_mat = self.arm_collision.body_local_mat
        self.collision_valid_rollouts = np.ones(self.n_samples, dtype=bool)
        self.collision_min_clearance = np.full(self.n_samples, np.inf)
        self.collision_exact_evaluations = 0

    def _solve_arm_ik(
        self,
        target_pos,
        initial_qpos,
        damping=None,
        step_size=None,
        max_iterations=None,
        tolerance=None,
        strict=False,
    ):
        """Delegate position IK while retaining the controller API."""
        result = self.arm_kinematics.solve_ik(
            target_pos,
            initial_qpos,
            damping=damping,
            step_size=step_size,
            max_iterations=max_iterations,
            tolerance=tolerance,
            strict=strict,
        )
        self.last_ik_residual = self.arm_kinematics.last_ik_residual
        return result

    def _update_arm_reference_to(self, observation, target_pos):
        """Update the smoothed arm reference toward one task target."""
        self.arm_reference = self.arm_kinematics.update_reference(
            observation,
            target_pos,
            self.arm_reference,
        )
        self.last_ik_residual = self.arm_kinematics.last_ik_residual

    def _ee_pose(self, observation):
        return self.arm_kinematics.ee_pose(observation)

    def _batch_arm_fk(self, flat_states):
        return self.arm_kinematics.batch_arm_fk(flat_states)

    def _batch_ee_pose(self, flat_states):
        return self.arm_kinematics.batch_ee_pose(flat_states)

    def _rollout_arm_fk(self, states, rollout_sensors):
        return self.arm_kinematics.rollout_arm_fk(states, rollout_sensors)

    def _rollout_ee_pose(self, states, rollout_sensors):
        return self.arm_kinematics.rollout_ee_pose(states, rollout_sensors)

    @staticmethod
    def _quat_rotation_matrices(quaternions):
        return ArmTorsoCollision.quaternion_rotation_matrices(quaternions)

    @staticmethod
    def _segment_aabb_distance(starts, ends, half_size):
        return ArmTorsoCollision.segment_aabb_distance(
            starts, ends, half_size
        )

    def _arm_torso_clearance(self, flat_states, arm_positions):
        result = self.arm_collision.evaluate(flat_states, arm_positions)
        self.collision_exact_evaluations = result.exact_evaluations
        return result.clearance, result.segment_valid

    def _gait_blend(self):
        """Return one startup blend factor for each horizon point."""
        if self.gait_startup_blend_steps == 0:
            return np.ones(self.horizon)
        horizon_steps = self._gait_nominal_step + np.arange(self.horizon)
        return np.clip(
            horizon_steps / self.gait_startup_blend_steps, 0.0, 1.0
        )

    def _joint_reference(self):
        """Combine phase-aligned leg gait with the current arm IK prior."""
        if hasattr(self.gait_scheduler, "get_reference"):
            gait = self.gait_scheduler.get_reference(
                self.base_height_cmd,
                height_rate=self.base_height_rate_cmd,
                horizon=self.horizon,
            )
        else:
            indices = self.gait_scheduler.indices[:self.horizon]
            gait = self.gait_scheduler.gait[:, indices]
        arm_q = np.repeat(self.arm_reference[:, None], self.horizon, axis=1)
        arm_dq = np.zeros_like(arm_q)
        reference = np.vstack((gait[:12], arm_q, gait[16:28], arm_dq))
        expected_shape = (2 * self.act_dim, self.horizon)
        if reference.shape != expected_shape:
            raise ValueError(
                f"Joint reference must have shape {expected_shape}, got "
                f"{reference.shape}"
            )

        if not self.nominal_from_gait:
            return reference

        blend = self._gait_blend()
        target_q = reference[:self.act_dim].T
        source_q = self._gait_blend_source
        reference[:self.act_dim] = (
            source_q + blend[:, None] * (target_q - source_q)
        ).T
        reference[self.act_dim:] *= blend[None, :]
        return reference

    def _refresh_gait_nominal(self):
        """Apply the latest gait phase and arm IK posture to the prior."""
        self.joints_ref = self._joint_reference()
        self.gait_nominal = self.joints_ref[:self.act_dim].T.copy()
        self.trajectory = np.clip(
            self.gait_nominal + self.gait_correction,
            self.act_min,
            self.act_max,
        )

    def _reset_gait_nominal(self, blend_source=None):
        """Reset the whole-body gait prior and warm-started corrections."""
        self._gait_nominal_step = 0
        self.gait_correction = np.zeros((self.horizon, self.act_dim))
        if blend_source is None:
            blend_source = np.repeat(
                self.sampling_init[None, :], self.horizon, axis=0
            )
        blend_source = np.asarray(blend_source, dtype=float)
        if blend_source.shape != (self.horizon, self.act_dim):
            raise ValueError(
                "gait blend source must have shape "
                f"({self.horizon}, {self.act_dim}), got {blend_source.shape}"
            )
        self._gait_blend_source = blend_source.copy()

        if self.nominal_from_gait:
            self._refresh_gait_nominal()
            self.selected_trajectory = self.trajectory.copy()

    def _advance_gait_nominal(self, updated_actions, nominal_actions):
        """Shift optimized corrections and advance the gait phase."""
        selected_correction = updated_actions - nominal_actions
        self.gait_correction[:-1] = selected_correction[1:]
        self.gait_correction[-1] = 0.0

        self.gait_scheduler.roll()
        self._gait_nominal_step += 1
        self._gait_blend_source[:-1] = self._gait_blend_source[1:]
        self._gait_blend_source[-1] = self._gait_blend_source[-2]
        self._refresh_gait_nominal()

    @staticmethod
    def _quat_to_roll_pitch(quaternions):
        """Return roll and pitch from MuJoCo [w, x, y, z] quaternions."""
        w, x, y, z = quaternions.T
        roll = np.arctan2(
            2.0 * (w * x + y * z),
            1.0 - 2.0 * (x * x + y * y),
        )
        sin_pitch = 2.0 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
        return np.column_stack((roll, pitch))

    def eval_best_trajectory(self):
        """Return the cached cost of the latest best sampled trajectory."""
        return self.cached_best_cost
