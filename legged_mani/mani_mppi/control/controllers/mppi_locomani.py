"""Whole-body MPPI controller for B2-Z1 locomani tasks."""

import os

import mujoco
import numpy as np
import yaml

from mani_mppi.control.controllers.base_controller import BaseMPPI
from mani_mppi.control.gait_scheduler.scheduler import GaitScheduler, Timer
from mani_mppi.utils.tasks import get_task
from mani_mppi.utils.transforms import batch_world_to_local_velocity


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GAIT_DIR = os.path.join(BASE_DIR, "../gait_scheduler/gaits/")
GAIT_PATHS = {
    "in_place": os.path.join(
        GAIT_DIR, "FAST/b2_z1/walking_gait_raibert_FAST_0_0_10cm_100hz.tsv"
    ),
    "trot": os.path.join(
        GAIT_DIR, "MED/b2_z1/walking_gait_raibert_MED_0_5_15cm_100hz.tsv"
    ),
    "walk": os.path.join(
        GAIT_DIR, "MED/b2_z1/walking_gait_raibert_MED_0_1_10cm_100hz.tsv"
    ),
    "walk_fast": os.path.join(
        GAIT_DIR, "FAST/b2_z1/walking_gait_raibert_FAST_0_1_10cm_100hz.tsv"
    ),
}


class MPPI(BaseMPPI):
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

        # Q/R contain state and control costs. EE task weights remain in
        # tasks.py because EE pose is not part of the generalized state.
        with open(config_path, "r", encoding="utf-8") as stream:
            params = yaml.safe_load(stream)
        self.Q = np.diag(np.asarray(params["Q_diag"], dtype=float))
        self.R = np.diag(np.asarray(params["R_diag"], dtype=float))
        self.ee_position_weight = float(self.task_data.get("ee_position_weight", 0.0))
        self.ee_orientation_weight = float(self.task_data.get("ee_orientation_weight", 0.0))
        self.ee_terminal_scale = float(self.task_data.get("ee_terminal_scale", 1.0))
        self.cost_func = self.calculate_total_cost

        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name
        )
        if self.ee_site_id < 0:
            raise ValueError(f"Unknown EE site: {self.ee_site_name}")

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
        self.default_gait = params.get("default_gait", "in_place")
        if self.default_gait not in GAIT_PATHS:
            raise ValueError(f"Unknown default gait: {self.default_gait}")
        self.gait_scheduler = GaitScheduler(
            gait_path=GAIT_PATHS[self.default_gait], name=self.default_gait
        )

        # Cache arm joint indices and initialize the IK workspace.
        self.arm_qpos_indices = self._joint_indices("qpos")
        self.arm_dof_indices = self._joint_indices("dof")
        self.ik_data = mujoco.MjData(self.model)
        self.arm_ik_damping = float(params.get("arm_ik_damping", 0.05))
        self.arm_ik_step_size = float(params.get("arm_ik_step_size", 0.7))
        self.arm_ik_max_iterations = int(params.get("arm_ik_max_iterations", 100))
        self.arm_ik_tolerance = float(params.get("arm_ik_tolerance", 1e-4))
        self.arm_ik_smoothing = float(params.get("arm_ik_smoothing", 0.3))
        self.arm_ik_max_step = float(params.get("arm_ik_max_step", 0.04))
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
        self.ee_data = mujoco.MjData(self.model)
        self.exp_weights = np.ones(self.n_samples) / self.n_samples

        # Initialize the nominal action trajectory and exploration noise.
        self.reset_planner()
        self.set_noise_for_gait(self.default_gait)
        print(f"Body reference keyframe: {body_keyframe}")
        print(f"Initial EE goal: {self.ee_goal_pos[0]}")
        print(f"IK arm reference: {np.round(self.arm_reference, 4)}")

    def _joint_indices(self, kind):
        """Return qpos or DoF addresses for the four arm joints."""
        names = ("joint1", "joint2", "joint3", "joint4")
        if kind == "qpos":
            addresses = self.model.jnt_qposadr
        else:
            addresses = self.model.jnt_dofadr
        return np.asarray(
            [addresses[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
             for name in names],
            dtype=int,
        )

    def _solve_arm_ik(
        self, target_pos, initial_qpos, damping, step_size, max_iterations, tolerance,
        strict=False,
    ):
        """Solve a position-only damped-least-squares IK once per task goal."""
        # IK provides a nominal arm posture; the actual EE pose is evaluated
        # separately in the rollout cost.
        data = self.ik_data
        data.qpos[:] = initial_qpos
        data.qvel[:] = 0.0
        joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ("joint1", "joint2", "joint3", "joint4")
        ]
        lower = self.model.jnt_range[joint_ids, 0]
        upper = self.model.jnt_range[joint_ids, 1]
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        for _ in range(max_iterations):
            mujoco.mj_forward(self.model, data)
            error = np.asarray(target_pos) - data.site_xpos[self.ee_site_id]
            if np.linalg.norm(error) <= tolerance:
                break
            mujoco.mj_jacSite(self.model, data, jacp, jacr, self.ee_site_id)
            jac = jacp[:, self.arm_dof_indices]
            lhs = jac @ jac.T + (damping ** 2) * np.eye(3)
            dq = jac.T @ np.linalg.solve(lhs, error)
            data.qpos[self.arm_qpos_indices] = np.clip(
                data.qpos[self.arm_qpos_indices] + step_size * dq,
                lower,
                upper,
            )

        mujoco.mj_forward(self.model, data)
        residual = np.linalg.norm(data.site_xpos[self.ee_site_id] - target_pos)
        self.last_ik_residual = residual
        if strict and residual > max(0.03, 10.0 * tolerance):
            raise ValueError(
                f"EE target is not reachable by the 4-DoF arm; residual={residual:.4f} m"
            )
        return data.qpos[self.arm_qpos_indices].copy()

    def _update_arm_reference(self, observation):
        """Re-solve IK from the current body pose once per MPPI update."""
        # Smooth and limit the IK update to avoid abrupt changes in the prior.
        current_qpos = np.asarray(observation[:self.model.nq], dtype=float)
        ik_reference = self._solve_arm_ik(
            self.ee_goal_pos[self.goal_index],
            current_qpos,
            damping=self.arm_ik_damping,
            step_size=self.arm_ik_step_size,
            max_iterations=self.arm_ik_max_iterations,
            tolerance=self.arm_ik_tolerance,
            strict=False,
        )
        smoothed = self.arm_reference + self.arm_ik_smoothing * (
            ik_reference - self.arm_reference
        )
        delta = np.clip(
            smoothed - self.arm_reference,
            -self.arm_ik_max_step,
            self.arm_ik_max_step,
        )
        self.arm_reference = self.arm_reference + delta

    def _joint_reference(self):
        """Combine gait joint references with the current arm posture prior."""
        indices = self.gait_scheduler.indices[:self.horizon]
        gait = self.gait_scheduler.gait[:, indices]
        arm_q = np.repeat(self.arm_reference[:, None], self.horizon, axis=1)
        arm_dq = np.zeros_like(arm_q)
        return np.vstack((gait[:12], arm_q, gait[16:28], arm_dq))

    def next_goal(self):
        """Advance to the next EE goal after its waiting time expires."""
        self.timer.increment()
        if self.goal_index < len(self.ee_goal_pos) - 1 and self.timer.done:
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
            self.timer = Timer(end_time=self.waiting_times[self.goal_index])
            print(f"Moved to next EE goal {self.goal_index}: {self.ee_goal_pos[self.goal_index]}")
        elif self.goal_index == len(self.ee_goal_pos) - 1 and not self.task_success and self.timer.done:
            print("Task succeeded.")
            self.task_success = True
        else:
            self.timer.waiting = True

        if not self.task_success:
            self.set_noise_for_gait(self.default_gait)

    def _ee_pose(self, observation):
        """Return the current EE position and quaternion for goal checks."""
        observation = np.asarray(observation, dtype=float)
        self.ee_data.qpos[:] = observation[:self.model.nq]
        self.ee_data.qvel[:] = observation[self.model.nq:]
        mujoco.mj_forward(self.model, self.ee_data)
        quat = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(quat, self.ee_data.site_xmat[self.ee_site_id])
        return self.ee_data.site_xpos[self.ee_site_id].copy(), quat

    def _batch_ee_pose(self, flat_states):
        """Evaluate EE position and orientation for flattened rollout states."""
        # Kinematics is sufficient for pose evaluation; full forward dynamics
        # for every rollout state would unnecessarily slow down MPPI.
        positions = np.empty((len(flat_states), 3), dtype=float)
        quaternions = np.empty((len(flat_states), 4), dtype=float)

        for index, state in enumerate(flat_states):
            self.ee_data.qpos[:] = state[:self.model.nq]
            mujoco.mj_kinematics(self.model, self.ee_data)

            positions[index] = self.ee_data.site_xpos[self.ee_site_id]
            mujoco.mju_mat2Quat(
                quaternions[index],
                self.ee_data.site_xmat[self.ee_site_id],
            )

        return positions, quaternions

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
        actions = self.perturb_action()
        self.obs = np.asarray(obs, dtype=float)
        self._update_arm_reference(self.obs)
        rollout_states = self.rollout_func(self.obs, actions)
        self.joints_ref = self._joint_reference()
        costs_sum = self.cost_func(
            rollout_states, actions, self.joints_ref, self.body_ref
        )
        self.gait_scheduler.roll()

        min_cost = np.min(costs_sum)
        # Reuse the cost already computed by the main MPPI rollout for
        # trajectory logging instead of launching an additional rollout from
        # Simulator.store_trajectory() on every simulation step.
        self.cached_best_cost = float(min_cost)
        cost_range = np.max(costs_sum) - min_cost
        if cost_range < 1e-12:
            self.exp_weights = np.ones(self.n_samples)
        else:
            self.exp_weights = np.exp(
                -((costs_sum - min_cost) / cost_range) / self.temperature
            )
        updated_actions = np.sum(
            self.exp_weights[:, None, None] * actions, axis=0
        ) / (np.sum(self.exp_weights) + 1e-10)
        updated_actions = np.clip(updated_actions, self.act_min, self.act_max)
        self.selected_trajectory = updated_actions
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
        u_error = kp * (u - x_joint) - kd * (v_joint - v_ref)

        # Q slots 0:7 represent [base_x, base_y, base_z, roll, pitch, yaw, unused].
        # Base x/y and yaw are disabled through zero entries in Q_diag.
        x_error[:, 0:2] = 0.0
        body_rp = self._quat_to_roll_pitch(x[:, 3:7])
        body_rp_ref = self._quat_to_roll_pitch(x_ref[:, 3:7])
        body_rp_error = body_rp - body_rp_ref
        x_error[:, 3] = body_rp_error[:, 0]
        x_error[:, 4] = body_rp_error[:, 1]
        x_error[:, 5:7] = 0.0

        return (
            np.einsum("ij,ik,jk->i", x_error, x_error, self.Q)
            + np.einsum("ij,ik,jk->i", u_error, u_error, self.R)
        )

    def calculate_total_cost(self, states, actions, joints_ref, body_ref):
        """Compute summed cost for all sampled trajectories."""
        num_samples, horizon = states.shape[:2]
        flat_states = states.reshape(-1, states.shape[-1])
        flat_actions = actions.reshape(-1, actions.shape[-1])

        # Direct EE task cost; IK remains a nominal arm-posture reference.
        ee_positions, ee_quaternions = self._batch_ee_pose(flat_states)
        target_position = self.ee_goal_pos[self.goal_index]
        target_quaternion = self.ee_goal_quat[self.goal_index]
        ee_position_error = ee_positions - target_position
        ee_position_cost = self.ee_position_weight * np.sum(
            ee_position_error**2, axis=1
        )
        quaternion_dot = np.abs(
            np.sum(ee_quaternions * target_quaternion[None, :], axis=1)
        )
        quaternion_dot = np.clip(quaternion_dot, -1.0, 1.0)
        ee_orientation_error = 2.0 * np.arccos(quaternion_dot)
        ee_orientation_cost = self.ee_orientation_weight * ee_orientation_error**2
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

    def eval_best_trajectory(self):
        """Return the cached cost of the best sample from the latest update."""
        return self.cached_best_cost

    def __del__(self):
        if hasattr(self, "_closed") and hasattr(self, "executor"):
            self.close()


if __name__ == "__main__":
    MPPI()
