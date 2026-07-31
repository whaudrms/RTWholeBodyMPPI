"""Shared CEM-guided base motion for whole-body arm MPPI tasks."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from mani_mppi.control.controllers.whole_body_arm_controller import (
    HEIGHT_GAIT_PATHS,
    WholeBodyArmMPPI,
)
from mani_mppi.control.gait_scheduler.height_conditioned_scheduler import (
    HeightConditionedGaitScheduler,
)
from mani_mppi.control.planning import KinematicBasePoseCEMPlanner


PLANNING = "planning"
EXECUTE_PLAN = "execute_plan"


class CEMWholeBodyArmMPPI(WholeBodyArmMPPI):
    """Common asynchronous CEM, adaptive gait, and MPPI update mechanics."""

    controller_label = "Whole-body"

    def _configure_cem_planner(
        self,
        model_path,
        params,
        *,
        thread_name_prefix,
    ):
        """Create the private kinematic planner and its single worker."""
        self.base_pose_planner = KinematicBasePoseCEMPlanner(
            model_path,
            params,
            ee_site_name=self.ee_site_name,
        )
        self.base_planner_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=thread_name_prefix,
        )
        self._base_planner_closed = False

    def _configure_height_gaits(self, params):
        """Initialize phase-compatible height-conditioned gait banks."""
        self.height_gaits = {
            name: HeightConditionedGaitScheduler(paths, name=name)
            for name, paths in HEIGHT_GAIT_PATHS.items()
        }
        self.default_gait = params.get("default_gait", self.tracking_gait)
        if self.default_gait not in self.height_gaits:
            raise ValueError(f"Unknown default gait: {self.default_gait}")
        self.gait_scheduler = self.height_gaits[self.default_gait]

    @staticmethod
    def _run_timed_base_plan(planner, observation, target):
        """Run one planner-owned CEM solve and return its elapsed time."""
        start_time = time.perf_counter()
        plan = planner.plan(observation, target)
        return plan, time.perf_counter() - start_time

    def _take_completed_cem_result(self):
        """Return and clear a completed CEM future, or None while pending."""
        if self._planner_future is None or not self._planner_future.done():
            return None
        future = self._planner_future
        self._planner_future = None
        return future.result()

    def _cancel_pending_cem(self):
        """Cancel and clear a CEM future when it has not started running."""
        if (
            self._planner_future is not None
            and self._planner_future.cancel()
        ):
            self._planner_future = None
            return True
        return False

    def _set_gait(self, gait_name):
        """Switch height-conditioned gait banks without resetting phase."""
        if gait_name == self.default_gait:
            return
        previous_gait = self.default_gait
        transition_source = self.trajectory.copy()
        old_phase = self.gait_scheduler.phase_time
        scheduler = self.height_gaits[gait_name]
        scheduler.phase_time = old_phase % scheduler.phase_length
        scheduler.indices = (
            scheduler.phase_time + np.arange(scheduler.phase_length)
        ) % scheduler.phase_length
        self.gait_scheduler = scheduler
        self.default_gait = gait_name
        self.set_noise_for_gait(gait_name)
        if self.nominal_from_gait:
            self._reset_gait_nominal(transition_source)
            self.last_safe_trajectory = self.trajectory.copy()
        print(
            f"{self.controller_label} gait: "
            f"{previous_gait} -> {gait_name}"
        )

    def _set_motion_phase(self, phase):
        if phase == self.motion_phase:
            return
        self.motion_phase = phase
        if phase == PLANNING and not getattr(self, "_has_active_plan", False):
            self._set_gait(self.tracking_gait)
        print(
            f"{self.controller_label} phase: {phase}, "
            f"gait={self.default_gait}, "
            f"height={self.base_height_cmd:.3f}"
        )

    def _ramp_height(self, target_height):
        dt = float(self.model.opt.timestep)
        delta = np.clip(
            target_height - self.base_height_cmd,
            -self.base_height_rate * dt,
            self.base_height_rate * dt,
        )
        self.base_height_cmd += delta
        self.base_height_rate_cmd = delta / dt

    def _effective_approach_speed(self):
        """Reduce commanded XY speed as the gait approaches deep crouch."""
        height_span = self.stand_base_height - self.min_base_height
        motion_height = min(
            self.base_height_cmd,
            self.planned_base_height,
        )
        height_ratio = np.clip(
            (motion_height - self.min_base_height) / height_span,
            0.0,
            1.0,
        )
        speed_scale = (
            self.crouch_speed_scale_min
            + (1.0 - self.crouch_speed_scale_min) * height_ratio
        )
        return self.approach_speed * speed_scale

    def _ramp_xy(self, target_xy):
        """Rate-limit the world-frame planar body command."""
        dt = float(self.model.opt.timestep)
        error = np.asarray(target_xy, dtype=float) - self.base_xy_cmd
        error_norm = np.linalg.norm(error)
        max_step = self._effective_approach_speed() * dt
        if error_norm <= max_step:
            delta = error
        elif error_norm > 1e-12:
            delta = max_step * error / error_norm
        else:
            delta = np.zeros(2, dtype=float)
        self.base_xy_cmd += delta
        self.base_xy_rate_cmd = delta / dt

    def _update_plan_settled(self, observation):
        """Update base-plan convergence with hysteresis."""
        current_xy = np.asarray(observation[:2], dtype=float)
        xy_error = np.linalg.norm(self.planned_base_xy - current_xy)
        height_error = abs(
            float(observation[2]) - self.planned_base_height
        )
        xy_command_error = np.linalg.norm(
            self.planned_base_xy - self.base_xy_cmd
        )
        height_command_error = abs(
            self.planned_base_height - self.base_height_cmd
        )
        commands_finished = (
            xy_command_error <= 1e-9
            and height_command_error <= 1e-9
        )

        if self.plan_settled:
            if (
                not commands_finished
                or xy_error > self.base_xy_reengage_tolerance
                or height_error > self.base_height_reengage_tolerance
            ):
                self.plan_settled = False
                print(
                    f"{self.controller_label} plan tracking re-engaged: "
                    f"xy_error={xy_error:.3f}, "
                    f"height_error={height_error:.3f}"
                )
        elif (
            commands_finished
            and xy_error <= self.base_xy_tolerance
            and height_error <= self.base_height_tolerance
        ):
            self.plan_settled = True
            print(
                f"{self.controller_label} plan settled: "
                f"base_xy={np.round(current_xy, 3)}, "
                f"height={float(observation[2]):.3f}"
            )

        planar_motion_needed = (
            xy_command_error > 1e-9
            or xy_error > self.base_xy_tolerance
        )
        if self.planar_gait_active:
            if not planar_motion_needed:
                self.planar_gait_active = False
        elif (
            xy_command_error > 1e-9
            or xy_error > self.base_xy_reengage_tolerance
        ):
            self.planar_gait_active = True
        return self.planar_gait_active

    def _set_stationary_body_reference(self, observation):
        """Hold the current XY pose while planning or after completion."""
        self.base_xy_cmd = np.asarray(observation[:2], dtype=float).copy()
        self.base_xy_rate_cmd[:] = 0.0
        self.base_height_rate_cmd = 0.0
        self.body_ref[:2] = self.base_xy_cmd
        self.body_ref[2] = self.base_height_cmd
        self.body_ref[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.body_ref[7:13] = 0.0

    def _track_planned_base_reference(self, observation):
        """Advance commands toward the active CEM base plan."""
        self._set_motion_phase(EXECUTE_PLAN)
        self._ramp_xy(self.planned_base_xy)
        self._ramp_height(self.planned_base_height)
        planar_gait_active = self._update_plan_settled(observation)
        desired_gait = (
            self.approach_gait
            if planar_gait_active and not self.plan_settled
            else self.tracking_gait
        )
        self._set_gait(desired_gait)

        self.body_ref[:2] = self.base_xy_cmd
        self.body_ref[2] = self.base_height_cmd
        self.body_ref[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.body_ref[7:13] = 0.0
        self.body_ref[7:9] = self.base_xy_rate_cmd

    def _arm_reference_target(self):
        """Return the task-specific EE position used by online IK."""
        raise NotImplementedError

    def _update_arm_reference(self, observation):
        """Re-solve IK toward the current task target."""
        self._update_arm_reference_to(
            observation,
            self._arm_reference_target(),
        )

    def _select_updated_actions(self, actions, costs_sum):
        """Apply hard validity and MPPI weights to sampled trajectories."""
        valid = (
            np.asarray(self.collision_valid_rollouts, dtype=bool)
            & np.isfinite(costs_sum)
        )
        if valid.shape != (self.n_samples,):
            raise ValueError(
                "collision_valid_rollouts must contain one flag per sample"
            )

        if np.any(valid):
            valid_costs = costs_sum[valid]
            min_cost = np.min(valid_costs)
            self.cached_best_cost = float(min_cost)
            cost_range = np.max(valid_costs) - min_cost
            self.exp_weights = np.zeros(self.n_samples, dtype=float)
            if cost_range < 1e-12:
                self.exp_weights[valid] = 1.0
            else:
                self.exp_weights[valid] = np.exp(
                    -((valid_costs - min_cost) / cost_range) / self.temperature
                )
            updated_actions = np.sum(
                self.exp_weights[:, None, None] * actions,
                axis=0,
            ) / (np.sum(self.exp_weights) + 1e-10)
            updated_actions = np.clip(
                updated_actions,
                self.act_min,
                self.act_max,
            )
            self.last_safe_trajectory = updated_actions.copy()
        else:
            self.cached_best_cost = float("inf")
            self.exp_weights = np.zeros(self.n_samples, dtype=float)
            updated_actions = np.empty_like(self.last_safe_trajectory)
            updated_actions[:-1] = self.last_safe_trajectory[1:]
            updated_actions[-1] = self.last_safe_trajectory[-1]
            self.last_safe_trajectory = updated_actions.copy()
        return updated_actions

    def _advance_selected_trajectory(
        self,
        updated_actions,
        nominal_actions,
    ):
        """Warm-start the next horizon after selecting an MPPI trajectory."""
        self.selected_trajectory = updated_actions
        if self.nominal_from_gait:
            self._advance_gait_nominal(updated_actions, nominal_actions)
        else:
            self.gait_scheduler.roll()
            self.trajectory = np.roll(updated_actions, shift=-1, axis=0)
            self.trajectory[-1] = updated_actions[-1]

    def quadruped_cost_np(self, x, u, x_ref):
        """Compute robot-state and actuator-consistent control cost."""
        kp = np.asarray(self.model.actuator_gainprm[:, 0], dtype=float)
        kd = -np.asarray(self.model.actuator_biasprm[:, 2], dtype=float)
        x_error = self._compact_robot_state_error(x, x_ref)
        x_joint = x[:, 7:23]
        v_joint = x[:, 29:45]
        u_error = kp * (u - x_joint) - kd * v_joint
        return (
            np.sum(
                x_error * x_error * self.state_cost_weights[None, :],
                axis=1,
            )
            + np.sum(
                u_error * u_error * self.control_cost_weights[None, :],
                axis=1,
            )
        )

    def close(self):
        """Shut down the private CEM worker and MPPI rollout workers."""
        if (
            hasattr(self, "base_planner_executor")
            and not getattr(self, "_base_planner_closed", True)
        ):
            self.base_planner_executor.shutdown(
                wait=True,
                cancel_futures=True,
            )
            self._base_planner_closed = True
        super().close()
