"""CEM-guided whole-body MPPI controller for B2-Z1 box pushing."""

import os

import mujoco
import numpy as np
import yaml

from mani_mppi.control.controllers.cem_whole_body_controller import (
    CEMWholeBodyArmMPPI,
    EXECUTE_PLAN,
    PLANNING,
)
from mani_mppi.control.controllers.whole_body_arm_controller import (
    HEIGHT_GAIT_PATHS,
)
from mani_mppi.control.planning import BasePosePlan
from mani_mppi.utils.tasks import get_task
from mani_mppi.utils.transforms import batch_world_to_local_velocity


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLAN = PLANNING


class MPPI(CEMWholeBodyArmMPPI):
    """Move the base to a CEM-selected pose and push a free box with the EE.

    CEM searches only the low-dimensional robot base pose. The existing MPPI
    rollout is the sole dynamic optimizer and scores robot motion, a moving
    box-surface EE target, arm/torso clearance, and box position error.
    """

    controller_label = "Push"

    def __init__(
        self, task="push_box", rollout_mode="original_spline"
    ) -> None:
        print("Task: ", task)
        if task != "push_box":
            raise ValueError("MPPI push-box controller only supports task='push_box'")

        self.task = task
        self.task_data = get_task(task)
        config_path = os.path.join(BASE_DIR, self.task_data["config_path"])
        model_path = os.path.join(BASE_DIR, "../..", self.task_data["model_path"])
        with open(config_path, "r", encoding="utf-8") as stream:
            params = yaml.safe_load(stream)

        super().__init__(model_path, config_path)

        robot_state_dim = (self.model.nq - 7) + (self.model.nv - 6)
        robot_cost_dim = robot_state_dim - 1
        self.state_cost_weights = np.asarray(params["Q_diag"], dtype=float)
        self.box_cost_weights = np.asarray(params["Q_box"], dtype=float)
        self.control_cost_weights = np.asarray(params["R_diag"], dtype=float)
        self.box_orientation_weight = float(
            params["box_orientation_weight"]
        )
        self.box_max_tilt = float(params["box_max_tilt"])
        self.box_orientation_violation_penalty = float(
            params["box_orientation_violation_penalty"]
        )
        if self.state_cost_weights.shape != (robot_cost_dim,):
            raise ValueError(
                f"Q_diag must contain {robot_cost_dim} compact state weights"
            )
        if self.box_cost_weights.shape != (3,):
            raise ValueError("Q_box must contain three box-position weights")
        if self.control_cost_weights.shape != (self.act_dim,):
            raise ValueError(f"R_diag must contain {self.act_dim} weights")
        if any(
            np.any(weights < 0.0)
            for weights in (
                self.state_cost_weights,
                self.box_cost_weights,
                self.control_cost_weights,
            )
        ):
            raise ValueError("Cost weights must be non-negative")

        self.x_box_ref = np.asarray(self.task_data["box_pos_ref"], dtype=float)
        if self.x_box_ref.shape != (3,) or not np.isfinite(self.x_box_ref).all():
            raise ValueError("box_pos_ref must contain three finite values")
        self.ee_position_weight = float(params["ee_position_weight"])
        self.ee_orientation_weight = float(params["ee_orientation_weight"])
        self.ee_terminal_scale = float(params["ee_terminal_scale"])

        self.nominal_from_gait = bool(params.get("nominal_from_gait", True))
        self._configure_rollout_mode(rollout_mode)
        self.gait_startup_blend_steps = int(
            params.get("gait_startup_blend_steps", 50)
        )
        self.adaptive_gait_enabled = bool(
            params.get("adaptive_gait_enabled", True)
        )
        self.stand_base_height = float(params["stand_base_height"])
        self.min_base_height = float(params["min_base_height"])
        self.base_height_rate = float(params["base_height_rate"])
        self.base_height_tolerance = float(params["base_height_tolerance"])
        self.base_xy_tolerance = float(params["base_xy_tolerance"])
        self.base_height_reengage_tolerance = float(
            params["base_height_reengage_tolerance"]
        )
        self.base_xy_reengage_tolerance = float(
            params["base_xy_reengage_tolerance"]
        )
        self.approach_speed = float(params["approach_speed"])
        self.crouch_speed_scale_min = float(params["crouch_speed_scale_min"])
        self.max_fallback_replans = int(params["max_fallback_replans"])
        self.approach_gait = params["approach_gait"]
        self.tracking_gait = params["tracking_gait"]

        self.precontact_gap = float(params["precontact_gap"])
        self.push_penetration = float(params["push_penetration"])
        self.ee_contact_height_offset = float(
            params["ee_contact_height_offset"]
        )
        self.ee_contact_tolerance = float(params["ee_contact_tolerance"])
        self.box_goal_tolerance = float(params["box_goal_tolerance"])
        self.box_goal_hold_steps = int(params["box_goal_hold_steps"])
        self.cem_replan_distance = float(params["cem_replan_distance"])
        self.cem_target_stale_tolerance = float(
            params["cem_target_stale_tolerance"]
        )
        self.cem_replan_cooldown_steps = int(
            params["cem_replan_cooldown_steps"]
        )
        self.base_box_clearance = float(params["base_box_clearance"])
        if (
            self.gait_startup_blend_steps < 0
            or not 0.0 < self.min_base_height <= self.stand_base_height
            or self.base_height_rate <= 0.0
            or self.base_height_tolerance <= 0.0
            or self.base_xy_tolerance <= 0.0
            or self.base_height_reengage_tolerance
            <= self.base_height_tolerance
            or self.base_xy_reengage_tolerance <= self.base_xy_tolerance
            or self.approach_speed <= 0.0
            or not 0.0 < self.crouch_speed_scale_min <= 1.0
            or self.max_fallback_replans < 1
            or self.box_goal_hold_steps < 1
            or self.cem_replan_cooldown_steps < 0
            or min(
                self.ee_position_weight,
                self.ee_orientation_weight,
                self.ee_terminal_scale,
                self.precontact_gap,
                self.push_penetration,
                self.ee_contact_tolerance,
                self.box_goal_tolerance,
                self.cem_replan_distance,
                self.cem_target_stale_tolerance,
                self.base_box_clearance,
                self.box_orientation_weight,
                self.box_orientation_violation_penalty,
            )
            < 0.0
            or not 0.0 < self.box_max_tilt <= np.pi
        ):
            raise ValueError("Invalid push-box controller configuration")
        for gait_name in (self.approach_gait, self.tracking_gait):
            if gait_name not in HEIGHT_GAIT_PATHS:
                raise ValueError(f"Unknown height-conditioned gait: {gait_name}")

        box_joint_name = self.task_data.get("box_joint", "box_joint")
        self.box_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, box_joint_name
        )
        if (
            self.box_joint_id < 0
            or self.model.jnt_type[self.box_joint_id]
            != mujoco.mjtJoint.mjJNT_FREE
        ):
            raise ValueError(f"'{box_joint_name}' must be a free joint")
        self.box_qpos_adr = int(self.model.jnt_qposadr[self.box_joint_id])
        self.box_dof_adr = int(self.model.jnt_dofadr[self.box_joint_id])
        box_qpos_indices = np.arange(self.box_qpos_adr, self.box_qpos_adr + 7)
        box_dof_indices = np.arange(self.box_dof_adr, self.box_dof_adr + 6)
        self.robot_qpos_indices = np.setdiff1d(
            np.arange(self.model.nq), box_qpos_indices
        )
        self.robot_dof_indices = np.setdiff1d(
            np.arange(self.model.nv), box_dof_indices
        )
        if (
            len(self.robot_qpos_indices) != 23
            or len(self.robot_dof_indices) != 22
        ):
            raise ValueError("Expected B2-Z1 robot state dimensions 23 + 22")

        box_geom_name = self.task_data.get("box_geom", "box_geom")
        self.box_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, box_geom_name
        )
        if (
            self.box_geom_id < 0
            or self.model.geom_type[self.box_geom_id] != mujoco.mjtGeom.mjGEOM_BOX
        ):
            raise ValueError(f"'{box_geom_name}' must be a box geom")
        self.box_half_size = self.model.geom_size[self.box_geom_id, :3].copy()

        self.ee_site_name = self.task_data.get("ee_site", "gripper_center")
        self._configure_arm_system(params, self.ee_site_name)
        self._configure_cem_planner(
            model_path,
            params,
            thread_name_prefix="push-box-cem",
        )

        body_keyframe = params.get("body_reference_keyframe", "stand")
        body_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, body_keyframe
        )
        if body_key_id < 0:
            raise ValueError(f"Unknown body reference keyframe: {body_keyframe}")
        initial_qpos = self.model.key_qpos[body_key_id].copy()
        initial_observation = np.concatenate(
            (initial_qpos, np.zeros(self.model.nv, dtype=float))
        )

        self.body_ref = np.zeros(13, dtype=float)
        self.body_ref[:7] = initial_qpos[:7]
        self._configure_height_gaits(params)

        self.base_height_cmd = self.stand_base_height
        self.base_height_rate_cmd = 0.0
        self.base_xy_cmd = self.body_ref[:2].copy()
        self.base_xy_rate_cmd = np.zeros(2, dtype=float)
        self.planned_base_xy = self.base_xy_cmd.copy()
        self.planned_base_height = self.stand_base_height
        self.motion_phase = EXECUTE_PLAN
        self.plan_settled = False
        self.planar_gait_active = False
        self.plan_requires_replan = False
        self.fallback_replan_count = 0
        self._fallback_limit_reported = False
        self._has_active_plan = False
        self._planned_contact_target = None
        self._planned_box_xy = None
        self.last_cem_result = None

        self._planner_request_id = 0
        self._planner_future = None
        self._cem_steps_since_request = self.cem_replan_cooldown_steps
        self._force_cem_replan = False

        self.goal_index = 0
        self.task_success = False
        self._box_goal_hold_count = 0
        self.contact_engaged = False
        self._last_push_direction = np.array([1.0, 0.0], dtype=float)
        self.robot_state = np.zeros(robot_state_dim, dtype=float)
        self.box_state = np.zeros(13, dtype=float)
        self.ee_target_pos = np.zeros(3, dtype=float)
        self.ee_goal_pos = np.zeros((1, 3), dtype=float)
        self.ee_goal_quat = np.array([[1.0, 0.0, 0.0, 0.0]])
        self.obs = initial_observation
        self._update_task_references(
            initial_observation, allow_engagement=False
        )

        self.arm_reference = self._solve_arm_ik(
            self.ee_target_pos,
            initial_qpos,
            damping=self.arm_ik_damping,
            step_size=self.arm_ik_step_size,
            max_iterations=self.arm_ik_max_iterations,
            tolerance=self.arm_ik_tolerance,
            strict=False,
        )
        self.cached_best_cost = None
        self.exp_weights = np.ones(self.n_samples) / self.n_samples
        self.cost_func = self.calculate_total_cost

        self.reset_planner()
        self.set_noise_for_gait(self.default_gait)
        self._reset_gait_nominal(self.trajectory)
        self.last_safe_trajectory = self.trajectory.copy()
        print(f"Body reference keyframe: {body_keyframe}")
        print(f"Box target: {np.round(self.x_box_ref, 4)}")
        print(f"Initial EE contact target: {np.round(self.ee_target_pos, 4)}")
        print(f"IK arm reference: {np.round(self.arm_reference, 4)}")
        print(f"Rollout mode: {self.rollout_mode} ({self.sample_type})")

    def _split_observation(self, observation):
        """Split [qpos, qvel] into canonical robot and free-box states."""
        observation = np.asarray(observation, dtype=float)
        expected = self.model.nq + self.model.nv
        if observation.shape != (expected,):
            raise ValueError(
                f"Expected push-box observation shape {(expected,)}, "
                f"got {observation.shape}"
            )
        qpos = observation[:self.model.nq]
        qvel = observation[self.model.nq:]
        robot_state = np.concatenate(
            (qpos[self.robot_qpos_indices], qvel[self.robot_dof_indices])
        )
        box_state = np.concatenate(
            (
                qpos[self.box_qpos_adr:self.box_qpos_adr + 7],
                qvel[self.box_dof_adr:self.box_dof_adr + 6],
            )
        )
        return robot_state, box_state

    def _push_directions(self, box_positions):
        """Return horizontal unit vectors from each box center to the goal."""
        box_positions = np.asarray(box_positions, dtype=float)
        delta = self.x_box_ref[None, :2] - box_positions[:, :2]
        distance = np.linalg.norm(delta, axis=1, keepdims=True)
        fallback = np.broadcast_to(
            self._last_push_direction[None, :], delta.shape
        )
        return np.divide(
            delta, distance, out=fallback.copy(), where=distance > 1e-9
        )

    def _batch_contact_targets(self, box_states, penetration=None):
        """Return the ray/box surface target for every predicted box state.

        Positive ``penetration`` moves the target into the rear face along the
        push direction. A negative value places it outside the face.
        """
        box_states = np.asarray(box_states, dtype=float)
        if box_states.ndim != 2 or box_states.shape[1] < 7:
            raise ValueError("box_states must have shape (N, >=7)")
        if penetration is None:
            penetration = (
                self.push_penetration
                if self.contact_engaged
                else -self.precontact_gap
            )
        penetration = float(penetration)

        directions_xy = self._push_directions(box_states[:, :3])
        directions = np.column_stack(
            (directions_xy, np.zeros(len(directions_xy), dtype=float))
        )
        rotations = self._quat_rotation_matrices(box_states[:, 3:7])
        directions_local = np.einsum("mji,mj->mi", rotations, directions)
        ratios = np.full_like(directions_local, np.inf)
        nonzero = np.abs(directions_local) > 1e-12
        np.divide(
            self.box_half_size[None, :],
            np.abs(directions_local),
            out=ratios,
            where=nonzero,
        )
        surface_distance = np.min(ratios, axis=1)
        targets = (
            box_states[:, :3]
            - surface_distance[:, None] * directions
            + penetration * directions
        )
        targets[:, 2] = (
            box_states[:, 2] + self.ee_contact_height_offset
        )
        return targets

    def _box_position_cost(self, box_states):
        """Weighted L1 box-position cost from the reference controller."""
        box_states = np.asarray(box_states, dtype=float)
        position_error = box_states[:, :3] - self.x_box_ref[None, :]
        return np.sum(
            np.abs(position_error) * self.box_cost_weights[None, :], axis=1
        )

    def _box_orientation_cost(self, box_states):
        """Penalize roll/pitch tilt while leaving box yaw unconstrained."""
        box_states = np.asarray(box_states, dtype=float)
        if box_states.ndim != 2 or box_states.shape[1] < 7:
            raise ValueError("box_states must have shape (N, >=7)")
        rotations = self._quat_rotation_matrices(box_states[:, 3:7])
        # R[2, 2] is the world-z component of the box's local up axis. It is
        # invariant to pure yaw, so diagonal pushing does not incur a penalty.
        tilt = np.arccos(np.clip(rotations[:, 2, 2], -1.0, 1.0))
        violation = tilt > self.box_max_tilt
        return (
            self.box_orientation_weight * tilt * tilt
            + self.box_orientation_violation_penalty * violation
        )

    def _update_task_references(self, observation, *, allow_engagement=True):
        """Refresh observed box state and its moving EE target."""
        robot_state, box_state = self._split_observation(observation)
        self.robot_state[:] = robot_state
        self.box_state[:] = box_state
        direction = self._push_directions(box_state[None, :])[0]
        if np.linalg.norm(self.x_box_ref[:2] - box_state[:2]) > 1e-9:
            self._last_push_direction = direction.copy()

        precontact_target = self._batch_contact_targets(
            box_state[None, :], penetration=-self.precontact_gap
        )[0]
        if (
            allow_engagement
            and not self.task_success
            and not self.contact_engaged
        ):
            ee_position, _ = self._ee_pose(observation)
            if (
                np.linalg.norm(ee_position - precontact_target)
                <= self.ee_contact_tolerance
            ):
                self.contact_engaged = True
                self._force_cem_replan = True
                print("Push contact engaged; moving EE target into box face.")

        penetration = (
            self.push_penetration
            if self.contact_engaged
            else -self.precontact_gap
        )
        self.ee_target_pos = self._batch_contact_targets(
            box_state[None, :], penetration=penetration
        )[0]
        self.ee_goal_pos[0] = self.ee_target_pos

    def _arm_reference_target(self):
        return self.ee_target_pos

    @staticmethod
    def _run_cem_request(
        planner, request_id, observation, target, box_xy
    ):
        plan, elapsed = MPPI._run_timed_base_plan(
            planner,
            observation,
            target,
        )
        return request_id, target, box_xy, plan, elapsed

    def _submit_cem_plan(self, observation):
        """Submit one CEM request without invoking the MPPI rollout."""
        if self._planner_future is not None or self.task_success:
            return False
        self._update_task_references(observation, allow_engagement=False)
        request_id = self._planner_request_id
        target = self.ee_target_pos.copy()
        box_xy = self.box_state[:2].copy()
        self._planner_future = self.base_planner_executor.submit(
            self._run_cem_request,
            self.base_pose_planner,
            request_id,
            np.asarray(observation, dtype=float).copy(),
            target,
            box_xy,
        )
        self._cem_steps_since_request = 0
        if not self._has_active_plan:
            self._set_motion_phase(PLAN)
        print(
            "CEM planning started in background for push contact: "
            f"target={np.round(target, 3)}"
        )
        return True

    def _invalidate_cem_plan(self):
        """Invalidate a pending result while preserving the active plan."""
        self._planner_request_id += 1
        self._cancel_pending_cem()

    def _base_box_pose_valid(self, base_xy, box_state):
        """Reject a CEM base pose whose planar footprints overlap the box."""
        base_xy = np.asarray(base_xy, dtype=float)
        offset = np.asarray(box_state[:2], dtype=float) - base_xy
        distance = np.linalg.norm(offset)
        if distance <= 1e-12:
            return False
        direction = offset / distance
        box_rotation = self._quat_rotation_matrices(
            np.asarray(box_state[3:7], dtype=float)[None, :]
        )[0]
        box_direction_local = box_rotation.T[:2, :2] @ direction
        box_support = np.sum(
            np.abs(box_direction_local) * self.box_half_size[:2]
        )
        base_support = np.sum(
            np.abs(direction) * self.collision_body_half_size[:2]
        )
        return (
            distance
            >= box_support + base_support + self.base_box_clearance
        )

    def _apply_base_pose_plan(
        self, observation, target, box_xy, plan: BasePosePlan, elapsed
    ):
        """Atomically swap a completed CEM result into execute_plan."""
        result = plan.cem_result
        feasible = bool(result.metrics.get("feasible", False))
        collision_valid = bool(result.metrics.get("collision_valid", True))
        _, current_box_state = self._split_observation(observation)
        box_clear = self._base_box_pose_valid(plan.base_xy, current_box_state)
        accepted = collision_valid and box_clear
        first_plan = not self._has_active_plan

        if accepted:
            self.planned_base_xy = np.asarray(plan.base_xy, dtype=float).copy()
            self.planned_base_height = float(plan.base_height)
        elif first_plan:
            self.planned_base_xy = np.asarray(observation[:2], dtype=float).copy()
            self.planned_base_height = float(self.base_height_cmd)
        self.last_cem_result = result
        self._planned_contact_target = np.asarray(target, dtype=float).copy()
        self._planned_box_xy = np.asarray(box_xy, dtype=float).copy()
        self.plan_requires_replan = not (feasible and accepted)
        self._fallback_limit_reported = False
        self._has_active_plan = True
        self.plan_settled = False
        self.planar_gait_active = False
        # Contact may have switched from the outside target to penetration
        # while this request was running. Accept the still-useful base pose,
        # but retain the explicit refresh request for the new contact target.
        self._force_cem_replan = (
            self._force_cem_replan
            and np.linalg.norm(self.ee_target_pos - target) > 1e-9
        )
        if first_plan:
            self.base_xy_cmd = np.asarray(observation[:2], dtype=float).copy()
            self.base_xy_rate_cmd[:] = 0.0

        if feasible and accepted:
            status = "feasible"
        elif accepted:
            status = "fallback-intermediate"
        elif not collision_valid:
            status = "rejected-arm-collision"
        else:
            status = "rejected-base-box-clearance"
        print(
            "CEM decision completed: "
            f"status={status}, base_xy={np.round(self.planned_base_xy, 3)}, "
            f"height={self.planned_base_height:.3f}, "
            f"residual={result.metrics.get('residual', np.inf):.4f}, "
            f"cost={result.cost:.3f}, evaluations={result.evaluations}, "
            f"compute_time={1000.0 * elapsed:.1f} ms"
        )
        self._set_motion_phase(EXECUTE_PLAN)

    def _apply_completed_cem_plan(self, observation):
        """Apply a ready result, discarding one computed for a stale box."""
        completed = self._take_completed_cem_result()
        if completed is None:
            return False
        request_id, target, box_xy, plan, elapsed = completed
        self._update_task_references(observation, allow_engagement=False)
        target_drift = np.linalg.norm(self.ee_target_pos - target)
        box_drift = np.linalg.norm(self.box_state[:2] - box_xy)
        if (
            request_id != self._planner_request_id
            or target_drift > self.cem_target_stale_tolerance
            or box_drift > self.cem_target_stale_tolerance
        ):
            print(
                "Discarded stale push CEM plan: "
                f"request={request_id}, target_drift={target_drift:.3f}, "
                f"box_drift={box_drift:.3f}"
            )
            self._force_cem_replan = True
            self._cem_steps_since_request = self.cem_replan_cooldown_steps
            return False
        self._apply_base_pose_plan(
            observation, target, box_xy, plan, elapsed
        )
        return True

    def _maybe_submit_refresh(self, observation):
        if self.task_success or self._planner_future is not None:
            return
        self._cem_steps_since_request += 1
        if not self._has_active_plan:
            self._submit_cem_plan(observation)
            return

        target_drift = (
            np.inf
            if self._planned_contact_target is None
            else np.linalg.norm(
                self.ee_target_pos - self._planned_contact_target
            )
        )
        fallback_ready = self.plan_requires_replan and self.plan_settled
        should_refresh = (
            self._force_cem_replan
            or target_drift >= self.cem_replan_distance
            or fallback_ready
        )
        if (
            should_refresh
            and self._cem_steps_since_request
            >= self.cem_replan_cooldown_steps
        ):
            if fallback_ready:
                if self.fallback_replan_count >= self.max_fallback_replans:
                    if not self._fallback_limit_reported:
                        print(
                            "Push CEM fallback replan limit reached: "
                            f"attempts={self.fallback_replan_count}"
                        )
                        self._fallback_limit_reported = True
                    return
                self.fallback_replan_count += 1
            self._submit_cem_plan(observation)

    def _update_motion_reference(self, observation):
        self._update_task_references(observation)
        self._apply_completed_cem_plan(observation)
        self._maybe_submit_refresh(observation)

        if self.task_success:
            self._set_gait(self.tracking_gait)
            self._set_stationary_body_reference(observation)
            return

        if not self._has_active_plan:
            self._set_stationary_body_reference(observation)
            return

        self._track_planned_base_reference(observation)

    def next_goal(self):
        """Complete the single box target and hold the current posture."""
        if self.task_success:
            return
        self.task_success = True
        self.contact_engaged = False
        self._invalidate_cem_plan()
        self._set_gait(self.tracking_gait)
        print("Push-box task succeeded. Holding position.")

    def advance_task(self, observation):
        """Compatibility helper for callers outside Simulator."""
        if self.goal_reached(observation):
            self.next_goal()

    def goal_reached(self, observation):
        """Require the box to remain inside its XY tolerance for a short hold."""
        if self.task_success:
            return False
        self._update_task_references(
            np.asarray(observation, dtype=float), allow_engagement=False
        )
        error = np.linalg.norm(self.box_state[:2] - self.x_box_ref[:2])
        if error <= self.box_goal_tolerance:
            self._box_goal_hold_count += 1
        else:
            self._box_goal_hold_count = 0
        return self._box_goal_hold_count >= self.box_goal_hold_steps

    def update(self, obs):
        """Run one complete CEM-guided push-box MPPI update."""
        # 1. Update the box/contact target, base plan, and arm IK reference.
        self.obs = np.asarray(obs, dtype=float)
        self._update_motion_reference(self.obs)
        self._update_arm_reference(self.obs)
        if self.nominal_from_gait:
            self._refresh_gait_nominal()

        # 2. Sample controls and retain one noise-free safe baseline.
        actions = self.perturb_action()
        actions[0] = self._noise_free_rollout_candidate()

        # 3. Roll out robot/box dynamics and build the gait/IK reference.
        rollout_states = self.rollout_func(self.obs, actions)
        self.joints_ref = self._joint_reference()
        nominal_actions = self.joints_ref[:self.act_dim].T

        # 4. Evaluate robot, box, contact, and terminal EE costs.
        costs_sum = self.calculate_total_cost(
            rollout_states,
            actions,
            self.joints_ref,
            self.body_ref,
            rollout_sensors=self.sensor_rollouts,
        )
        updated_actions = self._select_updated_actions(
            actions,
            costs_sum,
        )

        # 5. Warm-start and advance the gait phase for the next update.
        self._advance_selected_trajectory(
            updated_actions,
            nominal_actions,
        )
        return updated_actions[0]

    def box_cost_np(self, x_box):
        """Backward-compatible alias for the weighted L1 box cost."""
        return self._box_position_cost(x_box)

    def calculate_total_cost(
        self,
        states,
        actions,
        joints_ref,
        body_ref,
        rollout_sensors=None,
    ):
        """Sum robot, box, and manipulator-contact costs."""
        states = np.asarray(states, dtype=float)
        actions = np.asarray(actions, dtype=float)
        num_samples, horizon = states.shape[:2]
        if states.shape[2] != self.model.nq + self.model.nv:
            raise ValueError(
                "Rollout state dimension must equal nq + nv; got "
                f"{states.shape[2]}"
            )

        qpos = states[:, :, :self.model.nq]
        qvel = states[:, :, self.model.nq:]
        robot_states = np.concatenate(
            (
                qpos[:, :, self.robot_qpos_indices],
                qvel[:, :, self.robot_dof_indices],
            ),
            axis=2,
        )
        box_states = np.concatenate(
            (
                qpos[:, :, self.box_qpos_adr:self.box_qpos_adr + 7],
                qvel[:, :, self.box_dof_adr:self.box_dof_adr + 6],
            ),
            axis=2,
        )
        flat_states = states.reshape(-1, states.shape[-1])
        flat_robot_states = robot_states.reshape(-1, robot_states.shape[-1]).copy()
        flat_box_states = box_states.reshape(-1, box_states.shape[-1])
        flat_actions = actions.reshape(-1, actions.shape[-1])

        # The desired EE point moves with the predicted box and stays on the
        # face opposite the box-to-goal direction.
        if rollout_sensors is None:
            arm_positions, ee_quaternions = self._batch_arm_fk(flat_states)
        else:
            arm_positions, ee_quaternions = self._rollout_arm_fk(
                states, rollout_sensors
            )
        ee_positions = arm_positions[:, -1]
        ee_targets = self._batch_contact_targets(flat_box_states)
        ee_position_error = ee_positions - ee_targets
        ee_position_cost = self.ee_position_weight * np.sum(
            ee_position_error * ee_position_error, axis=1
        )
        if self.ee_orientation_weight > 0.0:
            target_quaternion = self.ee_goal_quat[0]
            quaternion_dot = np.abs(
                np.sum(
                    ee_quaternions * target_quaternion[None, :], axis=1
                )
            )
            ee_orientation_error = 2.0 * np.arccos(
                np.clip(quaternion_dot, -1.0, 1.0)
            )
            ee_orientation_cost = (
                self.ee_orientation_weight * ee_orientation_error**2
            )
        else:
            ee_orientation_cost = np.zeros(
                len(flat_states), dtype=float
            )

        if self.collision_enabled:
            capsule_clearance, capsule_valid = self._arm_torso_clearance(
                flat_states, arm_positions
            )
            clearance_rollouts = capsule_clearance.reshape(
                num_samples, horizon, 3
            )
            self.collision_min_clearance = np.min(
                clearance_rollouts, axis=(1, 2)
            )
            valid_rollouts = capsule_valid.reshape(
                num_samples, horizon, 3
            )
            self.collision_valid_rollouts = np.all(
                valid_rollouts,
                axis=(1, 2),
            )
        else:
            self.collision_min_clearance = np.full(num_samples, np.inf)
            self.collision_valid_rollouts = np.ones(num_samples, dtype=bool)
            self.collision_exact_evaluations = 0

        body_refs = np.repeat(body_ref[None, :], len(flat_states), axis=0)
        tiled_joints = np.tile(
            joints_ref.T, (num_samples, 1, 1)
        ).reshape(-1, joints_ref.shape[0])
        base_velocity_ref = np.zeros((len(flat_states), 6), dtype=float)
        base_velocity_ref[:, :2] = body_refs[:, 7:9]
        robot_ref = np.concatenate(
            (
                body_refs[:, :7],
                tiled_joints[:, :self.act_dim],
                base_velocity_ref,
                tiled_joints[:, self.act_dim:],
            ),
            axis=1,
        )
        flat_robot_states[:, 23:26] = batch_world_to_local_velocity(
            flat_robot_states[:, 3:7], flat_robot_states[:, 23:26]
        )

        robot_cost = self.quadruped_cost_np(
            flat_robot_states, flat_actions, robot_ref
        )
        box_cost = self.box_cost_np(flat_box_states)
        box_orientation_cost = self._box_orientation_cost(
            flat_box_states
        )
        costs = (
            robot_cost
            + box_cost
            + box_orientation_cost
            + ee_position_cost
            + ee_orientation_cost
        ).reshape(num_samples, horizon)

        # Keep the terminal EE emphasis from the B2-Z1 locomani controller.
        terminal_ee = (
            ee_position_cost + ee_orientation_cost
        ).reshape(num_samples, horizon)
        costs[:, -1] += self.ee_terminal_scale * terminal_ee[:, -1]
        return costs.sum(axis=1)

if __name__ == "__main__":
    MPPI()
