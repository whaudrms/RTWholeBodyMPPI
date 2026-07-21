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
        GAIT_DIR, "FAST/b2_retargeted/walking_gait_raibert_FAST_0_0_10cm_100hz.tsv"
    ),
    "trot": os.path.join(
        GAIT_DIR, "MED/b2_retargeted/walking_gait_raibert_MED_0_5_15cm_100hz.tsv"
    ),
    "walk": os.path.join(
        GAIT_DIR, "MED/b2_retargeted/walking_gait_raibert_MED_0_1_10cm_100hz.tsv"
    ),
    "walk_fast": os.path.join(
        GAIT_DIR, "FAST/b2_retargeted/walking_gait_raibert_FAST_0_1_10cm_100hz.tsv"
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

        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name
        )
        if self.ee_site_id < 0:
            raise ValueError(f"Unknown EE site: {self.ee_site_name}")
        self.arm_fk_point_names = (
            "arm_shoulder_pos",
            "arm_elbow_pos",
            "arm_wrist_pos",
            "ee_pos",
        )
        self.arm_fk_sensor_adrs = np.asarray(
            [self._sensor_address(name, expected_dim=3)
             for name in self.arm_fk_point_names],
            dtype=int,
        )
        self.arm_fk_body_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
                )
                for body_name in ("link02", "link03", "link04")
            ],
            dtype=int,
        )
        if np.any(self.arm_fk_body_ids < 0):
            raise ValueError("Arm FK bodies link02/link03/link04 must exist")
        self.ee_pos_sensor_adr = self._sensor_address("ee_pos", expected_dim=3)
        self.ee_quat_sensor_adr = self._sensor_address("ee_quat", expected_dim=4)
        self.enable_rollout_sensors()

        # Approximate the three arm links as capsules against the torso OBB.
        # The four FK frames are segment endpoints, not four equal-radius
        # collision points. The shoulder endpoint itself is never penalized.
        self.collision_enabled = bool(params.get("collision_enabled", True))
        self.collision_fast_samples = 9
        self.collision_fast_alpha = np.linspace(
            0.0, 1.0, self.collision_fast_samples, dtype=float
        )
        self.collision_capsule_radii = np.asarray(
            params.get("collision_capsule_radii", [0.030, 0.030, 0.0375]),
            dtype=float,
        )
        if self.collision_capsule_radii.shape != (3,):
            raise ValueError("collision_capsule_radii must contain 3 values")
        if np.any(self.collision_capsule_radii <= 0.0):
            raise ValueError("collision capsule radii must be positive")
        self.collision_shoulder_exclusion = float(
            params.get("collision_shoulder_exclusion", 0.07)
        )
        self.collision_safe_distance = float(
            params.get("collision_safe_distance", 0.05)
        )
        self.collision_hard_distance = float(
            params.get("collision_hard_distance", 0.005)
        )
        self.collision_soft_weight = float(
            params.get("collision_soft_weight", 10000.0)
        )
        if self.collision_shoulder_exclusion < 0.0:
            raise ValueError("collision_shoulder_exclusion must be non-negative")
        if self.collision_hard_distance >= self.collision_safe_distance:
            raise ValueError(
                "collision_hard_distance must be smaller than "
                "collision_safe_distance"
            )
        if self.collision_soft_weight < 0.0:
            raise ValueError("collision_soft_weight must be non-negative")

        collision_geom_name = params.get(
            "collision_body_geom", "base_collision"
        )
        self.collision_body_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, collision_geom_name
        )
        if self.collision_body_geom_id < 0:
            raise ValueError(f"Unknown collision body geom: {collision_geom_name}")
        if self.model.geom_type[self.collision_body_geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError("collision_body_geom must be a box geom")
        self.collision_body_half_size = self.model.geom_size[
            self.collision_body_geom_id, :3
        ].copy()
        self.collision_body_local_pos = self.model.geom_pos[
            self.collision_body_geom_id
        ].copy()
        self.collision_body_local_mat = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(
            self.collision_body_local_mat,
            self.model.geom_quat[self.collision_body_geom_id],
        )
        self.collision_body_local_mat = self.collision_body_local_mat.reshape(3, 3)

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
        self.collision_valid_rollouts = np.ones(self.n_samples, dtype=bool)
        self.collision_min_clearance = np.full(self.n_samples, np.inf)
        self.collision_exact_evaluations = 0

        # Initialize the nominal action trajectory and exploration noise.
        self.reset_planner()
        self.set_noise_for_gait(self.default_gait)
        self._reset_gait_nominal(self.trajectory)
        self.last_safe_trajectory = self.trajectory.copy()
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

    def _sensor_address(self, name, expected_dim):
        """Return the data address of a required rollout sensor."""
        sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, name
        )
        if sensor_id < 0:
            raise ValueError(f"Unknown rollout sensor: {name}")
        sensor_dim = int(self.model.sensor_dim[sensor_id])
        if sensor_dim != expected_dim:
            raise ValueError(
                f"Rollout sensor '{name}' must have dimension {expected_dim}, "
                f"got {sensor_dim}"
            )
        return int(self.model.sensor_adr[sensor_id])

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

    def _gait_blend(self):
        """Return one startup blend factor for each point in the horizon."""
        if self.gait_startup_blend_steps == 0:
            return np.ones(self.horizon)
        horizon_steps = self._gait_nominal_step + np.arange(self.horizon)
        return np.clip(
            horizon_steps / self.gait_startup_blend_steps, 0.0, 1.0
        )

    def _joint_reference(self):
        """Combine phase-aligned leg gait with the current arm IK prior."""
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
        """Apply the latest leg phase and arm IK posture to the action prior."""
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
        """Shift optimized corrections and seed the tail from future gait/IK."""
        selected_correction = updated_actions - nominal_actions
        self.gait_correction[:-1] = selected_correction[1:]
        self.gait_correction[-1] = 0.0

        self.gait_scheduler.roll()
        self._gait_nominal_step += 1
        self._gait_blend_source[:-1] = self._gait_blend_source[1:]
        self._gait_blend_source[-1] = self._gait_blend_source[-2]
        self._refresh_gait_nominal()

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

    def _ee_pose(self, observation):
        """Return the current EE position and quaternion for goal checks."""
        observation = np.asarray(observation, dtype=float)
        self.ee_data.qpos[:] = observation[:self.model.nq]
        self.ee_data.qvel[:] = observation[self.model.nq:]
        mujoco.mj_forward(self.model, self.ee_data)
        quat = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(quat, self.ee_data.site_xmat[self.ee_site_id])
        return self.ee_data.site_xpos[self.ee_site_id].copy(), quat

    def _batch_arm_fk(self, flat_states):
        """Evaluate four arm points with one kinematics call per input state."""
        # One mj_kinematics call updates every body/site frame.  Reading four
        # points from that result is substantially cheaper than four separate
        # FK evaluations.
        positions = np.empty((len(flat_states), 4, 3), dtype=float)
        quaternions = np.empty((len(flat_states), 4), dtype=float)

        for index, state in enumerate(flat_states):
            self.ee_data.qpos[:] = state[:self.model.nq]
            mujoco.mj_kinematics(self.model, self.ee_data)

            positions[index, :3] = self.ee_data.xpos[self.arm_fk_body_ids]
            positions[index, 3] = self.ee_data.site_xpos[self.ee_site_id]
            mujoco.mju_mat2Quat(
                quaternions[index],
                self.ee_data.site_xmat[self.ee_site_id],
            )

        return positions, quaternions

    def _batch_ee_pose(self, flat_states):
        """Evaluate EE pose while retaining the four-point FK implementation."""
        arm_positions, quaternions = self._batch_arm_fk(flat_states)
        return arm_positions[:, -1], quaternions

    def _rollout_arm_fk(self, states, rollout_sensors):
        """Return four arm points aligned with post-step rollout states.

        Intermediate points come from MuJoCo frame sensors already evaluated
        during rollout.  The final state has no following sensor sample, so a
        single ``mj_kinematics`` call per rollout produces all four points.
        """
        num_samples, horizon = states.shape[:2]
        expected_shape = (
            num_samples,
            horizon,
            self.model.nsensordata,
        )
        rollout_sensors = np.asarray(rollout_sensors, dtype=float)
        if rollout_sensors.shape != expected_shape:
            raise ValueError(
                f"rollout_sensors must have shape {expected_shape}, got "
                f"{rollout_sensors.shape}"
            )

        positions = np.empty((num_samples, horizon, 4, 3), dtype=float)
        for point_index, sensor_adr in enumerate(self.arm_fk_sensor_adrs):
            positions[:, :-1, point_index] = rollout_sensors[
                :, 1:, sensor_adr:sensor_adr + 3
            ]

        final_positions, final_quaternions = self._batch_arm_fk(
            states[:, -1, :]
        )
        positions[:, -1] = final_positions

        quaternions = np.empty((num_samples, horizon, 4), dtype=float)
        quaternions[:, :-1] = rollout_sensors[
            :, 1:, self.ee_quat_sensor_adr:self.ee_quat_sensor_adr + 4
        ]
        quaternions[:, -1] = final_quaternions
        return positions.reshape(-1, 4, 3), quaternions.reshape(-1, 4)

    def _rollout_ee_pose(self, states, rollout_sensors):
        """Return EE poses aligned with post-step rollout states.

        MuJoCo records frame sensors before the corresponding post-step state:
        sensor[t + 1] therefore matches state[t].  Only the final state has no
        following sensor sample, so it needs one FK evaluation per rollout.
        """
        arm_positions, quaternions = self._rollout_arm_fk(
            states, rollout_sensors
        )
        return arm_positions[:, -1], quaternions

    @staticmethod
    def _quat_rotation_matrices(quaternions):
        """Return world-from-local rotation matrices for [w, x, y, z]."""
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
    def _segment_aabb_distance(starts, ends, half_size):
        """Return exact Euclidean distance between segments and an AABB.

        Squared distance to a box is piecewise quadratic along a segment. Its
        minimum is therefore at an interval stationary point or at one of the
        six box-plane crossings. Evaluating those candidates avoids temporal
        Python loops and dense centerline sampling.
        """
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

        # The clipped stationary point is the minimum of each quadratic
        # interval, including either boundary when the unconstrained minimum
        # lies outside. Separate knot evaluation would therefore be redundant.
        candidate_positions = (
            a[:, None, :] + stationary[:, :, None] * direction[:, None, :]
        )
        outside = np.maximum(
            np.abs(candidate_positions) - half_size, 0.0
        )
        minimum_squared_distance = np.min(
            np.sum(outside * outside, axis=2), axis=1
        )
        return np.sqrt(minimum_squared_distance).reshape(original_shape)

    def _arm_torso_clearance(self, flat_states, arm_positions):
        """Return signed torso clearance for three arm capsules.

        Clearance is positive outside the torso, zero at contact, and negative
        in penetration. The proximal shoulder region is trimmed from only the
        first capsule; the shoulder FK point itself receives no point cost.
        """
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
        trim = np.minimum(self.collision_shoulder_exclusion, upper_length)
        starts_world[:, 0] += (
            upper_vector
            * (trim / np.maximum(upper_length, 1e-12))[:, None]
        )

        base_position = flat_states[:, :3]
        base_rotation = self._quat_rotation_matrices(flat_states[:, 3:7])
        endpoints_world = np.concatenate(
            (starts_world, ends_world), axis=1
        )
        relative_world = endpoints_world - base_position[:, None, :]
        # R maps base-local vectors to world vectors, so R.T maps world to base.
        endpoints_body = np.einsum(
            "mij,mni->mnj", base_rotation, relative_world
        )
        endpoints_geom = np.einsum(
            "mni,ij->mnj",
            endpoints_body - self.collision_body_local_pos,
            self.collision_body_local_mat,
        )
        starts_geom = endpoints_geom[:, :3]
        ends_geom = endpoints_geom[:, 3:]
        radii = np.broadcast_to(
            self.collision_capsule_radii[None, :], starts_geom.shape[:2]
        )

        # Fast distance upper/lower bounds. Distance to a closed box is
        # 1-Lipschitz, so the true segment distance can be at most half one
        # sample interval below the minimum sampled point distance.
        segment_vector = ends_geom - starts_geom
        samples_geom = (
            starts_geom[:, :, None, :]
            + self.collision_fast_alpha[None, None, :, None]
            * segment_vector[:, :, None, :]
        )
        outside = np.maximum(
            np.abs(samples_geom) - self.collision_body_half_size, 0.0
        )
        sampled_distance = np.min(
            np.linalg.norm(outside, axis=-1), axis=2
        )
        half_sample_spacing = np.linalg.norm(segment_vector, axis=2) / (
            2.0 * (self.collision_fast_samples - 1)
        )
        lower_clearance = (
            np.maximum(sampled_distance - half_sample_spacing, 0.0)
            - radii
        )
        clearance = sampled_distance - radii

        # Refine only segments that could enter the soft-cost region. This
        # superset also contains every possible hard-threshold violation, so
        # valid/invalid rollout decisions remain identical to exact mode.
        refine = lower_clearance < self.collision_safe_distance
        self.collision_exact_evaluations = int(np.count_nonzero(refine))
        if self.collision_exact_evaluations:
            exact_distance = self._segment_aabb_distance(
                starts_geom[refine],
                ends_geom[refine],
                self.collision_body_half_size,
            )
            clearance[refine] = exact_distance - radii[refine]
        return clearance

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
        self.obs = np.asarray(obs, dtype=float)
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
            capsule_clearance = self._arm_torso_clearance(
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
            self.collision_valid_rollouts = np.all(
                clearance_rollouts >= self.collision_hard_distance,
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

    def eval_best_trajectory(self):
        """Return the cached cost of the best sample from the latest update."""
        return self.cached_best_cost

    def __del__(self):
        if hasattr(self, "_closed") and hasattr(self, "executor"):
            self.close()


if __name__ == "__main__":
    MPPI()
