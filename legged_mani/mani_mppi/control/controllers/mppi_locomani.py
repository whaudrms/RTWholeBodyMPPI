"""Whole-body MPPI controller for B2-Z1 locomani tasks."""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
import yaml

from mani_mppi.control.controllers.whole_body_arm_controller import (
    HEIGHT_GAIT_PATHS,
    WholeBodyArmMPPI,
)
from mani_mppi.control.gait_scheduler.height_conditioned_scheduler import (
    HeightConditionedGaitScheduler,
)
from mani_mppi.control.gait_scheduler.scheduler import Timer
from mani_mppi.control.planning import (
    BasePosePlan,
    KinematicBasePoseCEMPlanner,
)
from mani_mppi.utils.tasks import get_task
from mani_mppi.utils.transforms import batch_world_to_local_velocity


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APPROACH = "approach"
LOWER = "lower"
TRACK = "track"
RECOVER = "recover"
PLANNING = "planning"


class MPPI(WholeBodyArmMPPI):
    """MPPI controller for locomotion with an end-effector pose task.

    The arm IK reference provides a nominal joint posture, while the rollout
    cost directly evaluates the MuJoCo end-effector pose.
    """

    def __init__(self, task="locomani") -> None:
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
        self.approach_speed = float(params.get("approach_speed", 0.15))
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
        ):
            raise ValueError("Invalid adaptive base-height configuration")

        # Shared arm IK/FK, rollout sensors, and capsule-to-torso collision.
        self._configure_arm_system(params, self.ee_site_name)
        # The background planner owns a separate model/data stack. It never
        # touches the main controller model or the MPPI rollout worker pool.
        self.base_pose_planner = KinematicBasePoseCEMPlanner(
            model_path,
            params,
            ee_site_name=self.ee_site_name,
        )
        self.base_planner_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="locomani-cem",
        )
        self._base_planner_closed = False

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
        self.height_gaits = {
            name: HeightConditionedGaitScheduler(paths, name=name)
            for name, paths in HEIGHT_GAIT_PATHS.items()
        }
        self.default_gait = params.get("default_gait", self.tracking_gait)
        if self.default_gait not in self.height_gaits:
            raise ValueError(f"Unknown default gait: {self.default_gait}")
        self.gait_scheduler = self.height_gaits[self.default_gait]
        self.base_height_cmd = self.stand_base_height
        self.base_height_rate_cmd = 0.0
        self.motion_phase = TRACK
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

    def _set_gait(self, gait_name):
        """Switch gait banks without resetting the shared phase."""
        if gait_name == self.default_gait:
            return
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

    def _set_motion_phase(self, phase):
        if phase == self.motion_phase:
            return
        self.motion_phase = phase
        gait_name = self.approach_gait if phase == APPROACH else self.tracking_gait
        self._set_gait(gait_name)
        print(
            f"Locomani phase: {phase}, gait={gait_name}, "
            f"height={self.base_height_cmd:.3f}"
        )

    @staticmethod
    def _run_base_plan_request(
        planner,
        request_id,
        goal_index,
        observation,
        target,
    ):
        """Execute one request entirely on the planner-owned model/data."""
        start_time = time.perf_counter()
        plan = planner.plan(observation, target)
        planning_seconds = time.perf_counter() - start_time
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
        if (
            self._planner_future is not None
            and self._planner_future.cancel()
        ):
            self._planner_future = None
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
        self.planned_base_xy = np.asarray(plan.base_xy, dtype=float).copy()
        self.planned_base_height = float(plan.base_height)
        self._planned_goal_index = self.goal_index
        status = (
            "feasible"
            if result.metrics.get("feasible", False)
            else "fallback"
        )
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
        xy_error = np.linalg.norm(
            self.planned_base_xy - np.asarray(observation[:2])
        )
        if (
            self.base_height_cmd < self.stand_base_height
            - self.base_height_tolerance
            and (
                xy_error > self.base_xy_tolerance
                or self.planned_base_height > self.base_height_cmd
                + self.base_height_tolerance
            )
        ):
            self._set_motion_phase(RECOVER)
        elif xy_error > self.base_xy_tolerance:
            self._set_motion_phase(APPROACH)
        elif (
            self.planned_base_height
            < self.base_height_cmd - self.base_height_tolerance
        ):
            self._set_motion_phase(LOWER)
        else:
            self.base_height_cmd = self.planned_base_height
            self._set_motion_phase(TRACK)

    def _poll_background_plan(self, observation):
        """Apply a ready current-goal result without waiting for the worker."""
        if self._planned_goal_index == self.goal_index:
            return True
        if self._planner_future is None:
            self._start_background_plan(observation)
            return False
        if not self._planner_future.done():
            return False

        future = self._planner_future
        self._planner_future = None
        self._planner_future_request_id = None
        self._planner_future_goal_index = None
        request_id, goal_index, plan, planning_seconds = future.result()
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

    def _ramp_height(self, target_height):
        dt = float(self.model.opt.timestep)
        delta = np.clip(
            target_height - self.base_height_cmd,
            -self.base_height_rate * dt,
            self.base_height_rate * dt,
        )
        self.base_height_cmd += delta
        self.base_height_rate_cmd = delta / dt

    def _update_motion_reference(self, observation):
        if (
            self.adaptive_gait_enabled
            and self._planned_goal_index != self.goal_index
        ):
            plan_ready = self._poll_background_plan(observation)
            if not plan_ready:
                # CEM is running on private data. Keep a stationary reference
                # so the main MPPI loop remains active and safe meanwhile.
                self.body_ref[:2] = np.asarray(observation[:2], dtype=float)
                self.body_ref[2] = self.base_height_cmd
                self.body_ref[3:7] = np.array(
                    [1.0, 0.0, 0.0, 0.0]
                )
                self.body_ref[7:13] = 0.0
                self.base_height_rate_cmd = 0.0
                return

        current_xy = np.asarray(observation[:2], dtype=float)
        if self.motion_phase == RECOVER:
            self.body_ref[:2] = current_xy
            self._ramp_height(self.stand_base_height)
            if (
                abs(observation[2] - self.stand_base_height)
                <= self.base_height_tolerance
                and abs(self.base_height_cmd - self.stand_base_height)
                <= 1e-9
            ):
                xy_error = np.linalg.norm(self.planned_base_xy - current_xy)
                if xy_error > self.base_xy_tolerance:
                    self._set_motion_phase(APPROACH)
                elif (
                    self.planned_base_height
                    < self.stand_base_height - self.base_height_tolerance
                ):
                    self._set_motion_phase(LOWER)
                else:
                    self._set_motion_phase(TRACK)
        elif self.motion_phase == APPROACH:
            self.base_height_cmd = self.stand_base_height
            self.base_height_rate_cmd = 0.0
            self.body_ref[:2] = self.planned_base_xy
            if (
                np.linalg.norm(self.planned_base_xy - current_xy)
                <= self.base_xy_tolerance
            ):
                if (
                    self.planned_base_height
                    < self.stand_base_height - self.base_height_tolerance
                ):
                    self._set_motion_phase(LOWER)
                else:
                    self._set_motion_phase(TRACK)
        elif self.motion_phase == LOWER:
            self.body_ref[:2] = self.planned_base_xy
            self._ramp_height(self.planned_base_height)
            if (
                abs(observation[2] - self.planned_base_height)
                <= self.base_height_tolerance
                and abs(self.base_height_cmd - self.planned_base_height)
                <= 1e-9
            ):
                self._set_motion_phase(TRACK)
        else:
            self.body_ref[:2] = self.planned_base_xy
            self.base_height_cmd = self.planned_base_height
            self.base_height_rate_cmd = 0.0

        self.body_ref[2] = self.base_height_cmd
        self.body_ref[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.body_ref[7:13] = 0.0
        if self.motion_phase == APPROACH:
            planar_error = self.planned_base_xy - current_xy
            error_norm = np.linalg.norm(planar_error)
            if error_norm > 1e-9:
                self.body_ref[7:9] = (
                    self.approach_speed * planar_error / error_norm
                )

    def _update_arm_reference(self, observation):
        """Re-solve IK from the current body pose once per MPPI update."""
        self._update_arm_reference_to(
            observation,
            self.ee_goal_pos[self.goal_index],
        )

    def next_goal(self):
        """Advance to the next EE goal after its waiting time expires."""
        self.timer.increment()
        if self.goal_index < len(self.ee_goal_pos) - 1 and self.timer.done:
            transition_source = self.trajectory.copy()
            self.goal_index += 1
            self._invalidate_base_plan()
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
        if self.adaptive_gait_enabled and self.motion_phase != TRACK:
            return False
        position, quat = self._ee_pose(observation)
        target_pos = self.ee_goal_pos[self.goal_index]
        target_quat = self.ee_goal_quat[self.goal_index]
        return (
            np.linalg.norm(position - target_pos) <= self.ee_pos_thresh
            and 1.0 - abs(float(np.dot(quat, target_quat))) <= self.ee_ori_thresh
        )

    def close(self):
        """Shut down the private CEM worker and the MPPI rollout workers."""
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

    def update(self, obs):
        """Sample, rollout, score, and update the MPPI action trajectory."""
        self.obs = np.asarray(obs, dtype=float)
        self._update_motion_reference(self.obs)
        self._update_arm_reference(self.obs)
        if self.nominal_from_gait:
            # Arm IK changes every update, so refresh the whole-body prior
            # before drawing candidate controls around it.
            self._refresh_gait_nominal()
        actions = self.perturb_action()
        # Keep one noise-free gait/IK candidate so rejection cannot eliminate
        # the nominal merely because every sampled perturbation is unsafe.
        actions[0] = np.clip(self.trajectory, self.act_min, self.act_max)
        rollout_states = self.rollout_func(self.obs, actions)
        self.joints_ref = self._joint_reference()
        nominal_actions = self.joints_ref[:self.act_dim].T
        costs_sum = self.cost_func(
            rollout_states,
            actions,
            self.joints_ref,
            self.body_ref,
            rollout_sensors=self.sensor_rollouts,
        )

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
            # Reuse the cost already computed by the main MPPI rollout for
            # trajectory logging instead of launching another rollout.
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
                self.exp_weights[:, None, None] * actions, axis=0
            ) / (np.sum(self.exp_weights) + 1e-10)
            updated_actions = np.clip(
                updated_actions, self.act_min, self.act_max
            )
            self.last_safe_trajectory = updated_actions.copy()
        else:
            # Every new candidate violated the hard clearance. Continue the
            # previously accepted plan, shifted by one step, instead of
            # averaging invalid controls or producing a zero action.
            self.cached_best_cost = float("inf")
            self.exp_weights = np.zeros(self.n_samples, dtype=float)
            updated_actions = np.empty_like(self.last_safe_trajectory)
            updated_actions[:-1] = self.last_safe_trajectory[1:]
            updated_actions[-1] = self.last_safe_trajectory[-1]
            self.last_safe_trajectory = updated_actions.copy()

        self.selected_trajectory = updated_actions
        if self.nominal_from_gait:
            self._advance_gait_nominal(updated_actions, nominal_actions)
        else:
            self.gait_scheduler.roll()
            self.trajectory = np.roll(updated_actions, shift=-1, axis=0)
            self.trajectory[-1] = updated_actions[-1]
        return updated_actions[0]

    def quaternion_distance_np(self, q1, q2):
        return 1.0 - np.abs(np.einsum("ij,ij->i", q1, q2))

    def quadruped_cost_np(self, x, u, x_ref):
        """Compute joint/base state cost and actuator-consistent control cost."""
        kp = np.asarray(self.model.actuator_gainprm[:, 0], dtype=float)
        kd = -np.asarray(self.model.actuator_biasprm[:, 2], dtype=float)
        x_error = x - x_ref
        x_joint = x[:, 7:23]
        v_joint = x[:, 29:45]
        v_ref = x_ref[:, 29:45]
        u_error = kp * (u - x_joint) - kd * (v_joint)

        # Q slots 0:7 represent [base_x, base_y, base_z, roll, pitch, yaw, unused].
        # Keep base x/y active when configured so an in-place gait cannot
        # reduce joint/EE cost by drifting away from the standing location.
        body_rp = self._quat_to_roll_pitch(x[:, 3:7])
        body_rp_ref = self._quat_to_roll_pitch(x_ref[:, 3:7])
        body_rp_error = body_rp - body_rp_ref
        x_error[:, 3] = body_rp_error[:, 0]
        x_error[:, 4] = body_rp_error[:, 1]
        x_error[:, 5:7] = 0.0

        return (
            np.sum(x_error * x_error * self.state_cost_weights[None, :], axis=1)
            + np.sum(
                u_error * u_error * self.control_cost_weights[None, :], axis=1
            )
        )

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
            clearance_deficit = np.maximum(
                self.collision_safe_distance - capsule_clearance, 0.0
            )
            collision_cost = self.collision_soft_weight * np.sum(
                clearance_deficit * clearance_deficit, axis=1
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
            collision_cost = np.zeros(len(flat_states), dtype=float)
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
        costs += ee_position_cost + ee_orientation_cost + collision_cost
        costs = costs.reshape(num_samples, horizon)

        # Increase the importance of the final EE pose.
        ee_terminal_cost = (
            ee_position_cost + ee_orientation_cost
        ).reshape(num_samples, horizon)
        costs[:, -1] += self.ee_terminal_scale * ee_terminal_cost[:, -1]
        return costs.sum(axis=1)

if __name__ == "__main__":
    MPPI()
