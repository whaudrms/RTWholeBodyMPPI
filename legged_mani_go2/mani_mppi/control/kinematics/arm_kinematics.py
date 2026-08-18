"""Go2 + SO-ARM100 IK, FK, and rollout-sensor access."""

from __future__ import annotations

import mujoco
import numpy as np


ARM_JOINT_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch")
ARM_BODY_NAMES = (
    "Rotation_Pitch", "Upper_Arm", "Lower_Arm", "Wrist_Pitch_Roll"
)
ARM_POINT_SENSOR_NAMES = (
    "arm_shoulder_pos",
    "arm_upper_pos",
    "arm_elbow_pos",
    "arm_wrist_pos",
    "ee_pos",
)


class ArmKinematics:
    """Own reusable MuJoCo data and indices for four controlled arm joints."""

    def __init__(
        self,
        model: mujoco.MjModel,
        config: dict,
        ee_site_name: str = "end_effector",
    ) -> None:
        self.model = model
        self.ee_site_name = ee_site_name
        self.ee_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name
        )
        if self.ee_site_id < 0:
            raise ValueError(f"Unknown EE site: {ee_site_name}")

        self.arm_joint_ids = self._named_ids(
            mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINT_NAMES
        )
        self.arm_qpos_indices = model.jnt_qposadr[self.arm_joint_ids].copy()
        self.arm_dof_indices = model.jnt_dofadr[self.arm_joint_ids].copy()
        self.arm_joint_lower = model.jnt_range[self.arm_joint_ids, 0].copy()
        self.arm_joint_upper = model.jnt_range[self.arm_joint_ids, 1].copy()
        self.arm_fk_body_ids = self._named_ids(
            mujoco.mjtObj.mjOBJ_BODY, ARM_BODY_NAMES
        )
        self.arm_fk_sensor_adrs = np.asarray(
            [
                self._sensor_address(name, expected_dim=3)
                for name in ARM_POINT_SENSOR_NAMES
            ],
            dtype=int,
        )
        self.ee_pos_sensor_adr = self._sensor_address(
            "ee_pos", expected_dim=3
        )
        self.ee_quat_sensor_adr = self._sensor_address(
            "ee_quat", expected_dim=4
        )

        self.damping = float(config.get("arm_ik_damping", 0.05))
        self.step_size = float(config.get("arm_ik_step_size", 0.7))
        self.max_iterations = int(config.get("arm_ik_max_iterations", 100))
        self.tolerance = float(config.get("arm_ik_tolerance", 1e-4))
        self.smoothing = float(config.get("arm_ik_smoothing", 0.3))
        self.max_step = float(config.get("arm_ik_max_step", 0.04))
        if self.max_iterations < 0:
            raise ValueError("arm_ik_max_iterations must be non-negative")
        if self.damping < 0.0 or self.step_size < 0.0:
            raise ValueError("arm IK damping and step size must be non-negative")
        if self.tolerance < 0.0 or self.smoothing < 0.0 or self.max_step < 0.0:
            raise ValueError(
                "arm IK tolerance, smoothing, and max step must be non-negative"
            )

        self.ik_data = mujoco.MjData(model)
        self.fk_data = mujoco.MjData(model)
        self.jacp = np.zeros((3, model.nv), dtype=float)
        self.jacr = np.zeros((3, model.nv), dtype=float)
        self.damping_identity = np.eye(3, dtype=float)
        self.last_ik_residual = np.inf

    def _named_ids(
        self,
        object_type: mujoco.mjtObj,
        names: tuple[str, ...],
    ) -> np.ndarray:
        ids = np.asarray(
            [mujoco.mj_name2id(self.model, object_type, name) for name in names],
            dtype=int,
        )
        missing = [name for name, object_id in zip(names, ids) if object_id < 0]
        if missing:
            raise ValueError(f"Model is missing required names: {missing}")
        return ids

    def _sensor_address(self, name: str, expected_dim: int) -> int:
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

    def solve_ik(
        self,
        target_pos: np.ndarray,
        initial_qpos: np.ndarray,
        *,
        damping: float | None = None,
        step_size: float | None = None,
        max_iterations: int | None = None,
        tolerance: float | None = None,
        strict: bool = False,
    ) -> np.ndarray:
        """Solve position-only damped-least-squares IK."""
        damping = self.damping if damping is None else float(damping)
        step_size = self.step_size if step_size is None else float(step_size)
        max_iterations = (
            self.max_iterations
            if max_iterations is None
            else int(max_iterations)
        )
        tolerance = self.tolerance if tolerance is None else float(tolerance)

        data = self.ik_data
        data.qpos[:] = initial_qpos
        data.qvel[:] = 0.0
        target_pos = np.asarray(target_pos, dtype=float)
        for _ in range(max_iterations):
            mujoco.mj_forward(self.model, data)
            error = target_pos - data.site_xpos[self.ee_site_id]
            if np.linalg.norm(error) <= tolerance:
                break
            mujoco.mj_jacSite(
                self.model, data, self.jacp, self.jacr, self.ee_site_id
            )
            jac = self.jacp[:, self.arm_dof_indices]
            lhs = jac @ jac.T + (damping ** 2) * self.damping_identity
            dq = jac.T @ np.linalg.solve(lhs, error)
            data.qpos[self.arm_qpos_indices] = np.clip(
                data.qpos[self.arm_qpos_indices] + step_size * dq,
                self.arm_joint_lower,
                self.arm_joint_upper,
            )

        mujoco.mj_forward(self.model, data)
        self.last_ik_residual = float(
            np.linalg.norm(data.site_xpos[self.ee_site_id] - target_pos)
        )
        if strict and self.last_ik_residual > max(0.03, 10.0 * tolerance):
            raise ValueError(
                "EE target is not reachable by the 4-DoF arm; "
                f"residual={self.last_ik_residual:.4f} m"
            )
        return data.qpos[self.arm_qpos_indices].copy()

    def update_reference(
        self,
        observation: np.ndarray,
        target_pos: np.ndarray,
        previous_reference: np.ndarray,
    ) -> np.ndarray:
        """Smooth one IK update from the current whole-body pose."""
        current_qpos = np.asarray(
            observation[:self.model.nq], dtype=float
        )
        ik_reference = self.solve_ik(target_pos, current_qpos, strict=False)
        previous_reference = np.asarray(previous_reference, dtype=float)
        smoothed = previous_reference + self.smoothing * (
            ik_reference - previous_reference
        )
        delta = np.clip(
            smoothed - previous_reference,
            -self.max_step,
            self.max_step,
        )
        return previous_reference + delta

    def ee_pose(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return EE position and quaternion for one observation."""
        observation = np.asarray(observation, dtype=float)
        self.fk_data.qpos[:] = observation[:self.model.nq]
        self.fk_data.qvel[:] = observation[self.model.nq:]
        mujoco.mj_forward(self.model, self.fk_data)
        quat = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(
            quat, self.fk_data.site_xmat[self.ee_site_id]
        )
        return self.fk_data.site_xpos[self.ee_site_id].copy(), quat

    def batch_arm_fk(
        self, flat_states: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the configured arm link frames and EE for each state."""
        point_count = len(self.arm_fk_body_ids) + 1
        positions = np.empty((len(flat_states), point_count, 3), dtype=float)
        quaternions = np.empty((len(flat_states), 4), dtype=float)
        for index, state in enumerate(flat_states):
            self.fk_data.qpos[:] = state[:self.model.nq]
            mujoco.mj_kinematics(self.model, self.fk_data)
            positions[index, :-1] = self.fk_data.xpos[self.arm_fk_body_ids]
            positions[index, -1] = self.fk_data.site_xpos[self.ee_site_id]
            mujoco.mju_mat2Quat(
                quaternions[index],
                self.fk_data.site_xmat[self.ee_site_id],
            )
        return positions, quaternions

    def batch_ee_pose(
        self, flat_states: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate EE poses through the shared arm-chain FK path."""
        arm_positions, quaternions = self.batch_arm_fk(flat_states)
        return arm_positions[:, -1], quaternions

    def rollout_arm_fk(
        self,
        states: np.ndarray,
        rollout_sensors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read rollout FK sensors aligned with post-step rollout states."""
        num_samples, horizon = states.shape[:2]
        expected_shape = (num_samples, horizon, self.model.nsensordata)
        rollout_sensors = np.asarray(rollout_sensors, dtype=float)
        if rollout_sensors.shape != expected_shape:
            raise ValueError(
                f"rollout_sensors must have shape {expected_shape}, got "
                f"{rollout_sensors.shape}"
            )

        point_count = len(self.arm_fk_sensor_adrs)
        positions = np.empty((num_samples, horizon, point_count, 3), dtype=float)
        for point_index, sensor_adr in enumerate(self.arm_fk_sensor_adrs):
            positions[:, :-1, point_index] = rollout_sensors[
                :, 1:, sensor_adr:sensor_adr + 3
            ]

        final_positions, final_quaternions = self.batch_arm_fk(
            states[:, -1, :]
        )
        positions[:, -1] = final_positions

        quaternions = np.empty((num_samples, horizon, 4), dtype=float)
        quaternions[:, :-1] = rollout_sensors[
            :, 1:, self.ee_quat_sensor_adr:self.ee_quat_sensor_adr + 4
        ]
        quaternions[:, -1] = final_quaternions
        return positions.reshape(-1, point_count, 3), quaternions.reshape(-1, 4)

    def rollout_ee_pose(
        self,
        states: np.ndarray,
        rollout_sensors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return only EE poses from rollout FK sensors."""
        arm_positions, quaternions = self.rollout_arm_fk(
            states, rollout_sensors
        )
        return arm_positions[:, -1], quaternions
