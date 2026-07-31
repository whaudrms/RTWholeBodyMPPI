"""Whole-body MPPI controller for B2-Z1 locomani tasks."""

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
from mani_mppi.control.gait_scheduler.scheduler import Timer
from mani_mppi.control.planning import BasePosePlan
from mani_mppi.utils.tasks import get_task
from mani_mppi.utils.transforms import batch_world_to_local_velocity


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class MPPI(CEMWholeBodyArmMPPI):
    """MPPI controller for locomotion with an end-effector pose task.

    The arm IK reference provides a nominal joint posture, while the rollout
    cost directly evaluates the MuJoCo end-effector pose.
    """

    controller_label = "Locomani"

    def __init__(
        self, task="locomani", rollout_mode="original_spline"
    ) -> None:
        print("Task: ", task)

        # Load task targets and task-specific EE cost parameters.
        self.task = task
        self.task_data = get_task(task)

        self.ee_goal_pos = np.asarray(self.task_data["ee_goal_pos"], dtype=float)
        self.ee_goal_quat = np.asarray(self.task_data["ee_goal_quat"], dtype=float)
        if self.ee_goal_pos.ndim != 2 or self.ee_goal_pos.shape[1] != 3:
            raise ValueError("ee_goal_pos must have shape (N, 3)")
        if self.ee_goal_quat.shape != (len(self.ee_goal_pos), 4):
            raise ValueError("ee_goal_quat must have shape (N, 4)")

        self.ee_site_name = self.task_data.get("ee_site", "gripper_center")
        self.ee_pos_thresh = float(self.task_data.get("ee_pos_thresh", 0.03))
        self.ee_ori_thresh = float(self.task_data.get("ee_ori_thresh", 0.1))

        config_path = os.path.join(BASE_DIR, self.task_data["config_path"])
        model_path = os.path.join(BASE_DIR, "../..", self.task_data["model_path"])

        # Initialize the MuJoCo model, MPPI sampler, and rollout workers.
        super().__init__(model_path, config_path)

        # Q/R and EE rollout costs are controller tuning parameters.
        with open(config_path, "r", encoding="utf-8") as stream:
            params = yaml.safe_load(stream)
        self.state_cost_weights = np.asarray(params["Q_diag"], dtype=float)
        self.control_cost_weights = np.asarray(params["R_diag"], dtype=float)
        state_cost_dim = self.model.nq + self.model.nv - 1
        if self.state_cost_weights.shape != (state_cost_dim,):
            raise ValueError(
                f"Q_diag must contain {state_cost_dim} compact state weights"
            )
        self.Q = np.diag(self.state_cost_weights)
        self.R = np.diag(self.control_cost_weights)
        self.ee_position_weight = float(params["ee_position_weight"])
        self.ee_orientation_weight = float(params["ee_orientation_weight"])
        self.ee_terminal_scale = float(params["ee_terminal_scale"])
        if self.ee_position_weight < 0.0 or self.ee_orientation_weight < 0.0:
            raise ValueError("EE cost weights must be non-negative")
        if self.ee_terminal_scale < 0.0:
            raise ValueError("ee_terminal_scale must be non-negative")
        self.cost_func = self.calculate_total_cost
        self.nominal_from_gait = bool(params.get("nominal_from_gait", True))
        self._configure_rollout_mode(rollout_mode)
        self.gait_startup_blend_steps = int(
            params.get("gait_startup_blend_steps", 50)
        )
        if self.gait_startup_blend_steps < 0:
            raise ValueError("gait_startup_blend_steps must be non-negative")

        # Reachability-aware base motion and height-conditioned gait settings.
        self.adaptive_gait_enabled = bool(
            params.get("adaptive_gait_enabled", True)
        )
        self.stand_base_height = float(
            params.get("stand_base_height", 0.543542)
        )
        self.min_base_height = float(params.get("min_base_height", 0.35))
        self.base_height_rate = float(params.get("base_height_rate", 0.10))
        self.base_height_tolerance = float(
            params.get("base_height_tolerance", 0.015)
        )
        self.base_xy_tolerance = float(
            params.get("base_xy_tolerance", 0.05)
        )
        self.base_xy_reengage_tolerance = float(
            params.get("base_xy_reengage_tolerance", 0.08)
        )
        self.base_height_reengage_tolerance = float(
            params.get("base_height_reengage_tolerance", 0.025)
        )
        self.approach_speed = float(params.get("approach_speed", 0.15))
        self.crouch_speed_scale_min = float(
            params.get("crouch_speed_scale_min", 0.40)
        )
        self.max_fallback_replans = int(
            params.get("max_fallback_replans", 8)
        )
        self.approach_gait = params.get("approach_gait", "walk_fast")
        self.tracking_gait = params.get("tracking_gait", "stance_hold")
        if self.approach_gait not in HEIGHT_GAIT_PATHS:
            raise ValueError(f"Unknown approach gait: {self.approach_gait}")
        if self.tracking_gait not in HEIGHT_GAIT_PATHS:
            raise ValueError(f"Unknown tracking gait: {self.tracking_gait}")
        if (
            not 0.0 < self.min_base_height <= self.stand_base_height
            or self.base_height_rate <= 0.0
            or self.base_height_tolerance <= 0.0
            or self.base_xy_tolerance <= 0.0
            or self.approach_speed <= 0.0
            or self.base_xy_reengage_tolerance <= self.base_xy_tolerance
            or self.base_height_reengage_tolerance
            <= self.base_height_tolerance
            or not 0.0 < self.crouch_speed_scale_min <= 1.0
            or self.max_fallback_replans < 1
        ):
            raise ValueError("Invalid adaptive base-height configuration")

        # Shared arm IK/FK, rollout sensors, and capsule-to-torso collision.
        self._configure_arm_system(params, self.ee_site_name)
        # The background planner owns a separate model/data stack. It never
        # touches the main controller model or the MPPI rollout worker pool.
        self._configure_cem_planner(
            model_path,
            params,
            thread_name_prefix="locomani-cem",
        )

        # The body reference pose comes from the selected MuJoCo keyframe.
        body_keyframe = params.get("body_reference_keyframe", "stand")
        body_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, body_keyframe
        )
        if body_key_id < 0:
            raise ValueError(f"Unknown body reference keyframe: {body_keyframe}")

        self.body_ref = np.zeros(13, dtype=float)
        self.body_ref[:7] = self.model.key_qpos[body_key_id][:7]

        # Initialize phase-compatible gait banks for walking and crouching.
        self._configure_height_gaits(params)
        self.base_height_cmd = self.stand_base_height
        self.base_height_rate_cmd = 0.0
        self.base_xy_cmd = self.body_ref[:2].copy()
        self.base_xy_rate_cmd = np.zeros(2, dtype=float)
        self.motion_phase = EXECUTE_PLAN
        self.plan_settled = False
        self.planar_gait_active = False
        self.plan_requires_replan = False
        self.fallback_replan_count = 0
        self._fallback_limit_reported = False
        self.planned_base_xy = self.body_ref[:2].copy()
        self.planned_base_height = self.stand_base_height
        self.last_cem_result = None
        self._planned_goal_index = -1
        self._planner_request_id = 0
        self._planner_future = None
        self._planner_future_request_id = None
        self._planner_future_goal_index = None

        # Compute the initial arm posture that reaches the first EE goal.
        self.arm_reference = self._solve_arm_ik(
            self.ee_goal_pos[0],
            self.model.key_qpos[body_key_id],
            damping=self.arm_ik_damping,
            step_size=self.arm_ik_step_size,
            max_iterations=self.arm_ik_max_iterations,
            tolerance=self.arm_ik_tolerance,
            strict=False,
        )

        self.internal_ref = True
        self.obs = None
        self.cached_best_cost = None
        self.goal_index = 0
        self.task_success = False
        self.waiting_times = list(
            self.task_data.get("waiting_times", [0] * len(self.ee_goal_pos))
        )
        if len(self.waiting_times) != len(self.ee_goal_pos):
            raise ValueError("waiting_times must match the number of EE goals")
        self.timer = Timer(end_time=self.waiting_times[0])
        self.exp_weights = np.ones(self.n_samples) / self.n_samples

        # Initialize the nominal action trajectory and exploration noise.
        self.reset_planner()
        self.set_noise_for_gait(self.default_gait)
        self._reset_gait_nominal(self.trajectory)
        self.last_safe_trajectory = self.trajectory.copy()
        print(f"Body reference keyframe: {body_keyframe}")
        print(f"Initial EE goal: {self.ee_goal_pos[0]}")
        print(f"IK arm reference: {np.round(self.arm_reference, 4)}")
        print(f"Rollout mode: {self.rollout_mode} ({self.sample_type})")

    @staticmethod
    def _run_base_plan_request(
        planner,
        request_id,
        goal_index,
        observation,
        target,
    ):
        """Execute one request entirely on the planner-owned model/data."""
        plan, planning_seconds = MPPI._run_timed_base_plan(
            planner,
            observation,
            target,
        )
        return request_id, goal_index, plan, planning_seconds

    def _start_background_plan(self, observation):
        """Submit one non-blocking CEM request for the current EE goal."""
        if self._planner_future is not None:
            return
        request_id = self._planner_request_id
        goal_index = self.goal_index
        self._planner_future_request_id = request_id
        self._planner_future_goal_index = goal_index
        self._planner_future = self.base_planner_executor.submit(
            self._run_base_plan_request,
            self.base_pose_planner,
            request_id,
            goal_index,
            np.asarray(observation, dtype=float).copy(),
            self.ee_goal_pos[goal_index].copy(),
        )
        self._set_motion_phase(PLANNING)
        print(
            f"CEM planning started in background for EE goal {goal_index}"
        )

    def _invalidate_base_plan(self):
        """Invalidate pending/completed results after an EE goal change."""
        self._planner_request_id += 1
        self._planned_goal_index = -1
        self.plan_settled = False
        self.planar_gait_active = False
        self.plan_requires_replan = False
        if self._cancel_pending_cem():
            self._planner_future_request_id = None
            self._planner_future_goal_index = None

    def _apply_base_pose_plan(
        self,
        observation,
        plan: BasePosePlan,
        planning_seconds=None,
    ):
        """Atomically expose a completed plan to the main control thread."""
        result = plan.cem_result
        self.last_cem_result = result
        feasible = bool(result.metrics.get("feasible", False))
        collision_valid = bool(
            result.metrics.get("collision_valid", True)
        )
        if collision_valid:
            self.planned_base_xy = np.asarray(
                plan.base_xy,
                dtype=float,
            ).copy()
            self.planned_base_height = float(plan.base_height)
        else:
            # A finite CEM cost can still be returned when every candidate
            # collides. Never expose that pose as a body command.
            self.planned_base_xy = np.asarray(
                observation[:2],
                dtype=float,
            ).copy()
            self.planned_base_height = float(self.base_height_cmd)
        self._planned_goal_index = self.goal_index
        self.plan_requires_replan = not feasible
        self._fallback_limit_reported = False
        if feasible:
            status = "feasible"
        elif collision_valid:
            status = "fallback-intermediate"
        else:
            status = "fallback-rejected-collision"
        timing = (
            ""
            if planning_seconds is None
            else f", compute_time={1000.0 * planning_seconds:.1f} ms"
        )
        print(
            f"CEM decision completed: goal={self.goal_index}, "
            f"status={status}, "
            f"base_xy={np.round(self.planned_base_xy, 3)}, "
            f"height={self.planned_base_height:.3f}, "
            f"residual={result.metrics.get('residual', np.inf):.4f}, "
            f"cost={result.cost:.3f}, evaluations={result.evaluations}"
            f"{timing}"
        )
        self.base_xy_cmd = np.asarray(observation[:2], dtype=float).copy()
        self.base_xy_rate_cmd[:] = 0.0
        self.plan_settled = False
        self.planar_gait_active = False
        self._set_motion_phase(EXECUTE_PLAN)

    def _ee_goal_satisfied(self, observation):
        """Return actual EE success without coupling it to base convergence."""
        position, quat = self._ee_pose(observation)
        target_pos = self.ee_goal_pos[self.goal_index]
        target_quat = self.ee_goal_quat[self.goal_index]
        return (
            np.linalg.norm(position - target_pos) <= self.ee_pos_thresh
            and 1.0 - abs(float(np.dot(quat, target_quat)))
            <= self.ee_ori_thresh
        )

    def _replan_settled_fallback(self, observation):
        """Chain a safe fallback waypoint into a new same-goal CEM plan."""
        if (
            not self.adaptive_gait_enabled
            or self._planned_goal_index != self.goal_index
            or not self.plan_requires_replan
            or not self.plan_settled
            or self._ee_goal_satisfied(observation)
        ):
            return
        if self.fallback_replan_count >= self.max_fallback_replans:
            if not self._fallback_limit_reported:
                print(
                    "CEM fallback replan limit reached: "
                    f"goal={self.goal_index}, "
                    f"attempts={self.fallback_replan_count}"
                )
                self._fallback_limit_reported = True
            return

        self.fallback_replan_count += 1
        print(
            "CEM intermediate plan settled; replanning same EE goal: "
            f"goal={self.goal_index}, "
            f"attempt={self.fallback_replan_count}/"
            f"{self.max_fallback_replans}"
        )
        self._invalidate_base_plan()

    def _poll_background_plan(self, observation):
        """Apply a ready current-goal result without waiting for the worker."""
        if self._planned_goal_index == self.goal_index:
            return True
        if self._planner_future is None:
            self._start_background_plan(observation)
            return False
        completed = self._take_completed_cem_result()
        if completed is None:
            return False

        self._planner_future_request_id = None
        self._planner_future_goal_index = None
        request_id, goal_index, plan, planning_seconds = completed
        if (
            request_id != self._planner_request_id
            or goal_index != self.goal_index
        ):
            print(
                "Discarded stale CEM plan: "
                f"request={request_id}, goal={goal_index}"
            )
            self._start_background_plan(observation)
            return False

        self._apply_base_pose_plan(
            observation,
            plan,
            planning_seconds=planning_seconds,
        )
        return True

    def _update_motion_reference(self, observation):
        # A fallback at the bounded CEM search radius is an intermediate body
        # waypoint. Give the simulator one control interval to accept an EE hit,
        # then continue planning from the newly reached body pose if necessary.
        self._replan_settled_fallback(observation)
        if (
            self.adaptive_gait_enabled
            and self._planned_goal_index != self.goal_index
        ):
            plan_ready = self._poll_background_plan(observation)
            if not plan_ready:
                # CEM is running on private data. Keep a stationary reference
                # so the main MPPI loop remains active and safe meanwhile.
                self.plan_settled = False
                self.planar_gait_active = False
                self._set_stationary_body_reference(observation)
                return

        self._track_planned_base_reference(observation)

    def _arm_reference_target(self):
        return self.ee_goal_pos[self.goal_index]

    def next_goal(self):
        """Advance to the next EE goal after its waiting time expires."""
        self.timer.increment()
        if self.goal_index < len(self.ee_goal_pos) - 1 and self.timer.done:
            transition_source = self.trajectory.copy()
            self.goal_index += 1
            self._invalidate_base_plan()
            self.fallback_replan_count = 0
            self._fallback_limit_reported = False
            initial_qpos = (
                self.obs[:self.model.nq]
                if self.obs is not None
                else self.ik_data.qpos.copy()
            )
            self.arm_reference = self._solve_arm_ik(
                self.ee_goal_pos[self.goal_index],
                initial_qpos,
                damping=self.arm_ik_damping,
                step_size=self.arm_ik_step_size,
                max_iterations=self.arm_ik_max_iterations,
                tolerance=self.arm_ik_tolerance,
                strict=False,
            )
            if self.nominal_from_gait:
                # A distant EE waypoint can move the IK solution sharply.
                # Blend from the currently optimized whole-body trajectory so
                # the new arm nominal cannot create an action discontinuity.
                self._reset_gait_nominal(transition_source)
            self.timer = Timer(end_time=self.waiting_times[self.goal_index])
            print(f"Moved to next EE goal {self.goal_index}: {self.ee_goal_pos[self.goal_index]}")
        elif self.goal_index == len(self.ee_goal_pos) - 1 and not self.task_success and self.timer.done:
            print("Task succeeded.")
            self.task_success = True
        else:
            self.timer.waiting = True

        if not self.task_success:
            self.set_noise_for_gait(self.default_gait)

    def goal_reached(self, observation):
        """Check whether the current EE goal satisfies its thresholds."""
        if self.adaptive_gait_enabled and (
            self._planned_goal_index != self.goal_index
            or self.motion_phase != EXECUTE_PLAN
        ):
            return False
        return self._ee_goal_satisfied(observation)

    def update(self, obs):
        """Run one complete CEM-guided whole-body MPPI update."""
        # 1. Update the planned base motion and current arm IK reference.
        self.obs = np.asarray(obs, dtype=float)
        self._update_motion_reference(self.obs)
        self._update_arm_reference(self.obs)
        if self.nominal_from_gait:
            self._refresh_gait_nominal()

        # 2. Sample controls and retain one noise-free safe baseline.
        actions = self.perturb_action()
        actions[0] = self._noise_free_rollout_candidate()

        # 3. Roll out dynamics and build the phase-aligned gait/IK reference.
        rollout_states = self.rollout_func(self.obs, actions)
        self.joints_ref = self._joint_reference()
        nominal_actions = self.joints_ref[:self.act_dim].T

        # 4. Evaluate the task cost, excluding hard-collision rollouts.
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

    def calculate_total_cost(
        self,
        states,
        actions,
        joints_ref,
        body_ref,
        rollout_sensors=None,
    ):
        """Compute summed cost for all sampled trajectories."""
        num_samples, horizon = states.shape[:2]
        flat_states = states.reshape(-1, states.shape[-1])
        flat_actions = actions.reshape(-1, actions.shape[-1])

        # Direct EE task cost; IK remains a nominal arm-posture reference.
        # MuJoCo already evaluates these frame sensors during mj_step. Reading
        # the rollout sensor buffer avoids repeating 1,200 Python-dispatched
        # forward-kinematics calls for the default 30 x 40 MPPI batch.
        if rollout_sensors is None:
            arm_positions, ee_quaternions = self._batch_arm_fk(flat_states)
        else:
            arm_positions, ee_quaternions = self._rollout_arm_fk(
                states, rollout_sensors
            )
        ee_positions = arm_positions[:, -1]
        target_position = self.ee_goal_pos[self.goal_index]
        target_quaternion = self.ee_goal_quat[self.goal_index]
        ee_position_error = ee_positions - target_position
        ee_position_cost = self.ee_position_weight * np.sum(
            ee_position_error**2, axis=1
        )
        if self.ee_orientation_weight > 0.0:
            quaternion_dot = np.abs(
                np.sum(ee_quaternions * target_quaternion[None, :], axis=1)
            )
            quaternion_dot = np.clip(quaternion_dot, -1.0, 1.0)
            ee_orientation_error = 2.0 * np.arccos(quaternion_dot)
            ee_orientation_cost = (
                self.ee_orientation_weight * ee_orientation_error**2
            )
        else:
            ee_orientation_cost = np.zeros(len(flat_states), dtype=float)

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
        tiled_joints = np.tile(joints_ref.T, (num_samples, 1, 1)).reshape(
            -1, joints_ref.shape[0]
        )
        base_velocity_ref = np.zeros((len(flat_states), 6), dtype=float)
        base_velocity_ref[:, :2] = body_refs[:, 7:9]
        x_ref = np.concatenate(
            [body_refs[:, :7], tiled_joints[:, :16], base_velocity_ref,
             tiled_joints[:, 16:]], axis=1
        )
        flat_states[:, 23:26] = batch_world_to_local_velocity(
            flat_states[:, 3:7], flat_states[:, 23:26]
        )
        costs = self.quadruped_cost_np(flat_states, flat_actions, x_ref)
        costs += ee_position_cost + ee_orientation_cost
        costs = costs.reshape(num_samples, horizon)

        # Increase the importance of the final EE pose.
        ee_terminal_cost = (
            ee_position_cost + ee_orientation_cost
        ).reshape(num_samples, horizon)
        costs[:, -1] += self.ee_terminal_scale * ee_terminal_cost[:, -1]
        return costs.sum(axis=1)

if __name__ == "__main__":
    MPPI()
