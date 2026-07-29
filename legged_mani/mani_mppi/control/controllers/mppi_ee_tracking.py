"""Whole-body MPPI controller for B2-Z1 locomani tasks."""

import os

import mujoco
import numpy as np
import yaml

from mani_mppi.control.controllers.whole_body_arm_controller import (
    GAIT_PATHS,
    WholeBodyArmMPPI,
)
from mani_mppi.control.gait_scheduler.scheduler import GaitScheduler, Timer
from mani_mppi.utils.tasks import get_task
from mani_mppi.utils.transforms import batch_world_to_local_velocity


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class MPPI(WholeBodyArmMPPI):
    """MPPI controller for locomotion with an end-effector pose task.

    The arm IK reference provides a nominal joint posture, while the rollout
    cost directly evaluates the MuJoCo end-effector pose.
    """

    def __init__(
        self, task="ee_tracking", rollout_mode="original_spline"
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

        # Shared arm IK/FK, rollout sensors, and capsule-to-torso collision.
        self._configure_arm_system(params, self.ee_site_name)

        # The body reference pose comes from the selected MuJoCo keyframe.
        body_keyframe = params.get("body_reference_keyframe", "stand")
        body_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, body_keyframe
        )
        if body_key_id < 0:
            raise ValueError(f"Unknown body reference keyframe: {body_keyframe}")

        self.body_ref = np.zeros(13, dtype=float)
        self.body_ref[:7] = self.model.key_qpos[body_key_id][:7]

        # Initialize the gait reference used for the leg joints.
        self.default_gait = params.get("default_gait", "walk_fast")
        if self.default_gait not in GAIT_PATHS:
            raise ValueError(f"Unknown default gait: {self.default_gait}")
        self.gait_scheduler = GaitScheduler(
            gait_path=GAIT_PATHS[self.default_gait], name=self.default_gait
        )

        # Compute the initial arm posture that reaches the first EE goal.
        self.arm_reference = self._solve_arm_ik(
            self.ee_goal_pos[0],
            self.model.key_qpos[body_key_id],
            damping=self.arm_ik_damping,
            step_size=self.arm_ik_step_size,
            max_iterations=self.arm_ik_max_iterations,
            tolerance=self.arm_ik_tolerance,
            strict=True,
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
            self.arm_reference = self._solve_arm_ik(
                self.ee_goal_pos[self.goal_index],
                self.ik_data.qpos.copy(),
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
        position, quat = self._ee_pose(observation)
        target_pos = self.ee_goal_pos[self.goal_index]
        target_quat = self.ee_goal_quat[self.goal_index]
        return (
            np.linalg.norm(position - target_pos) <= self.ee_pos_thresh
            and 1.0 - abs(float(np.dot(quat, target_quat))) <= self.ee_ori_thresh
        )

    def update(self, obs):
        """Sample, rollout, score, and update the MPPI action trajectory."""
        self.obs = np.asarray(obs, dtype=float)
        self._update_arm_reference(self.obs)
        if self.nominal_from_gait:
            # Arm IK changes every update, so refresh the whole-body prior
            # before drawing candidate controls around it.
            self._refresh_gait_nominal()
        actions = self.perturb_action()
        # Keep one noise-free gait/IK candidate so rejection cannot eliminate
        # the nominal merely because every sampled perturbation is unsafe.
        actions[0] = self._noise_free_rollout_candidate()
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

    def quadruped_cost_np(self, x, u, x_ref):
        """Compute joint/base state cost and actuator-consistent control cost."""
        kp = np.asarray(self.model.actuator_gainprm[:, 0], dtype=float)
        kd = -np.asarray(self.model.actuator_biasprm[:, 2], dtype=float)
        x_error = self._compact_robot_state_error(x, x_ref)
        x_joint = x[:, 7:23]
        v_joint = x[:, 29:45]
        v_ref = x_ref[:, 29:45]
        u_error = kp * (u - x_joint) - kd * (v_joint)

        # Keep base x/y active when configured so an in-place gait cannot
        # reduce joint/EE cost by drifting away from the standing location.
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
