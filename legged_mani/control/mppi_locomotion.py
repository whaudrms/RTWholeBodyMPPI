"""Original RTWholeBodyMPPI locomotion controller adapted for B2-Z1."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from mujoco import rollout

from legged_mani.control.base_controller import BaseMPPI
from legged_mani.control.gait_scheduler.scheduler import GaitScheduler, Timer
from legged_mani.interface.environment import MODEL_PATH
from legged_mani.utils.state_layout import B2Z1StateLayout
from legged_mani.utils.tasks import get_task
from legged_mani.utils.transforms import (
    batch_world_to_local_velocity,
    calculate_orientation_quaternion,
)


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "mppi_locomotion_in_place.yml"
)


class MPPI(BaseMPPI):
    """MPPI locomotion controller preserving the original source structure.

    The original controller API and cost pipeline are retained. Robot-specific
    changes are limited to the 45-D B2-Z1 state layout, 16-D controls, and the
    32-row joint reference scheduler.
    """

    def __init__(
        self,
        task: str = "in_place",
        model_path: str | Path = MODEL_PATH,
        config_path: str | Path | None = None,
    ) -> None:
        self.task = task
        self.task_data = get_task(task)
        config_path = self.task_data.config_path if config_path is None else config_path
        super().__init__(model_path, config_path)

        self.layout = B2Z1StateLayout()
        if (self.model.nq, self.model.nv, self.model.nu) != (23, 22, 16):
            raise ValueError("B2-Z1 locomotion requires model dimensions 23/22/16")

        q_diag = np.asarray(self.params["Q_diag"], dtype=float)
        r_diag = np.asarray(self.params["R_diag"], dtype=float)
        if q_diag.shape != (self.layout.state_dim,):
            raise ValueError(f"Q_diag must have 45 entries, got {q_diag.shape}")
        if r_diag.shape != (self.act_dim,):
            raise ValueError(f"R_diag must have 16 entries, got {r_diag.shape}")
        self.Q = np.diag(q_diag)
        self.R = np.diag(r_diag)

        gains = self.params.get("virtual_pd_gains", {})
        self.cost_kp = float(gains.get("kp", 50.0))
        self.cost_kd = float(gains.get("kd", 3.0))

        body_reference_keyframe = self.params.get(
            "body_reference_keyframe", "stand"
        )
        body_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, body_reference_keyframe
        )
        if body_key_id < 0:
            raise ValueError(
                f"Unknown body-reference keyframe: {body_reference_keyframe}"
            )
        nominal_qpos = self.model.key_qpos[body_key_id]
        desired_linear_velocity = np.asarray(
            self.params.get("desired_linear_velocity", [0.0, 0.0, 0.0]),
            dtype=float,
        )
        desired_angular_velocity = np.asarray(
            self.params.get("desired_angular_velocity", [0.0, 0.0, 0.0]),
            dtype=float,
        )
        if desired_linear_velocity.shape != (3,):
            raise ValueError("desired_linear_velocity must have 3 entries")
        if desired_angular_velocity.shape != (3,):
            raise ValueError("desired_angular_velocity must have 3 entries")

        # Preserve the original task-level sequence of body waypoints,
        # commanded velocities, and gait transitions.
        if self.task_data.goal_positions:
            self.goal_pos = np.asarray(self.task_data.goal_positions, dtype=float)
            self.cmd_vel = np.asarray(
                self.task_data.desired_linear_velocities, dtype=float
            )
            self.desired_gait = list(self.task_data.desired_gaits)
            self.goal_thresh = np.asarray(
                self.task_data.goal_thresholds, dtype=float
            )
            self.waiting_times = list(self.task_data.waiting_times)
        else:
            self.goal_pos = np.asarray([nominal_qpos[:3]], dtype=float)
            self.cmd_vel = np.asarray([desired_linear_velocity], dtype=float)
            self.desired_gait = ["in_place"]
            self.goal_thresh = np.asarray([0.2], dtype=float)
            self.waiting_times = [0]

        sequence_length = len(self.goal_pos)
        sequence_shapes = {
            "cmd_vel": len(self.cmd_vel),
            "desired_gait": len(self.desired_gait),
            "goal_thresh": len(self.goal_thresh),
            "waiting_times": len(self.waiting_times),
        }
        invalid = {
            name: length
            for name, length in sequence_shapes.items()
            if length != sequence_length
        }
        if self.goal_pos.shape != (sequence_length, 3) or self.cmd_vel.shape != (
            sequence_length,
            3,
        ) or invalid:
            raise ValueError(
                "Locomotion task sequence fields must have matching lengths "
                f"and 3-D positions/velocities: {invalid}"
            )

        self.goal_ori = nominal_qpos[3:7].copy()

        self.obs = None
        self.internal_ref = True
        self.exp_weights = np.ones(self.n_samples) / self.n_samples
        self.timer = Timer(end_time=self.waiting_times[0])
        reference_configs = self.params.get("joint_references")
        if reference_configs is None:
            reference_config = self.params.get("joint_reference", {})
            reference_configs = {
                reference_config.get("name", "in_place"): reference_config
            }
        self.gaits = {
            name: self._create_gait_scheduler(reference_config)
            for name, reference_config in reference_configs.items()
        }
        missing_gaits = set(self.desired_gait) - set(self.gaits)
        if missing_gaits:
            raise ValueError(
                f"Task '{task}' references unavailable gaits: "
                f"{sorted(missing_gaits)}"
            )
        self.gait_scheduler = self.gaits[self.desired_gait[0]]
        self.joint_reference = self.gait_scheduler
        self.reference_step = 0

        self.reset_planner()
        self.goal_index = 0
        self.body_ref = np.concatenate(
            (
                self.goal_pos[self.goal_index],
                self.goal_ori,
                self.cmd_vel[self.goal_index],
                desired_angular_velocity,
            )
        )
        self.joints_ref = self.gait_scheduler.horizon(self.horizon)
        self.reference = None
        self.last_costs = np.full(self.n_samples, np.nan)
        self.task_success = False
        self.cost_func = self.calculate_total_cost

    def _create_gait_scheduler(self, reference_config: dict) -> GaitScheduler:
        reference_frequency = float(reference_config.get("frequency_hz", 100.0))
        if reference_frequency <= 0.0:
            raise ValueError("joint_reference.frequency_hz must be positive")
        reference_timestep = 1.0 / reference_frequency
        if not np.isclose(reference_timestep, self.model.opt.timestep):
            raise ValueError(
                "Gait reference period and MuJoCo timestep differ: "
                f"reference={reference_timestep}, model={self.model.opt.timestep}"
            )
        reference_type = reference_config.get("type", "keyframe")
        name = reference_config.get("name", "in_place")
        if reference_type == "keyframe":
            return GaitScheduler.from_keyframe(
                self.model,
                keyframe=reference_config.get("keyframe", "stand"),
                cycle_steps=int(reference_config.get("cycle_steps", 100)),
                name=name,
            )
        if reference_type == "tsv":
            path = Path(reference_config["path"])
            if not path.is_absolute():
                path = Path(self.task_data.config_path).resolve().parent / path
            return GaitScheduler.from_tsv(path, name=name)
        raise ValueError(f"Unsupported joint reference type: {reference_type}")

    def next_goal(self) -> None:
        """Retain the original task progression API for future B2 gait tasks."""
        self.timer.increment()
        if self.goal_index < len(self.goal_pos) - 1 and self.timer.done:
            self.goal_index += 1
            self.body_ref[:3] = self.goal_pos[self.goal_index]
            self.body_ref[7:10] = self.cmd_vel[self.goal_index]
            self.gait_scheduler = self.gaits[self.desired_gait[self.goal_index]]
            self.joint_reference = self.gait_scheduler
            self.timer.reset()
            self.timer.end_time = self.waiting_times[self.goal_index]
            self.timer.waiting = False
        elif self.goal_index == len(self.goal_pos) - 1 and self.timer.done:
            self.task_success = True
        else:
            self.timer.waiting = True

        # Keep the original gait-dependent leg exploration policy, with four
        # zero-noise arm dimensions appended for B2-Z1.
        if not self.task_success:
            gait = self.desired_gait[self.goal_index]
            if gait in ("in_place", "walk", "walk_fast"):
                self.noise_sigma = np.array([0.06, 0.1, 0.1] * 4 + [0.0] * 4)
            elif gait == "trot":
                self.noise_sigma = np.array([0.06, 0.2, 0.2] * 4 + [0.0] * 4)

    def update(self, obs: np.ndarray) -> np.ndarray:
        """Run one original-style MPPI update for a 45-D B2-Z1 observation."""
        obs = np.asarray(obs, dtype=float)
        if obs.shape != (self.layout.state_dim,):
            raise ValueError(f"Expected observation shape (45,), got {obs.shape}")

        actions = self.perturb_action()
        self.obs = obs.copy()

        direction = self.body_ref[:3] - obs[:3]
        goal_delta = np.linalg.norm(direction)
        if goal_delta > 0.1 and not self.timer.waiting:
            self.goal_ori = calculate_orientation_quaternion(obs[:3], self.body_ref[:3])
        else:
            self.goal_ori = np.array([1.0, 0.0, 0.0, 0.0])
        self.body_ref[3:7] = self.goal_ori

        states = self.rollout_func(obs, actions)
        if self.internal_ref:
            self.joints_ref = self.gait_scheduler.gait[
                :, self.gait_scheduler.indices[: self.horizon]
            ]
        costs_sum = self.cost_func(states, actions, self.joints_ref, self.body_ref)
        if not np.isfinite(costs_sum).all():
            raise FloatingPointError("Non-finite locomotion MPPI cost")

        # Original scheduler advances once per update. The simulator advances
        # the remaining decimated physics steps through advance_reference().
        self.gait_scheduler.roll()

        min_cost = np.min(costs_sum)
        cost_range = np.max(costs_sum) - min_cost
        if cost_range <= 1e-12:
            self.exp_weights = np.ones(self.n_samples)
        else:
            self.exp_weights = np.exp(
                -1.0 / self.temperature * ((costs_sum - min_cost) / cost_range)
            )

        weighted_actions = self.exp_weights.reshape(self.n_samples, 1, 1) * actions
        updated_actions = np.sum(weighted_actions, axis=0) / (
            np.sum(self.exp_weights) + 1e-10
        )
        updated_actions = np.clip(updated_actions, self.act_min, self.act_max)

        self.last_costs = costs_sum
        self.selected_trajectory = updated_actions.copy()
        self.trajectory = np.roll(updated_actions, shift=-1, axis=0)
        self.trajectory[-1] = updated_actions[-1]
        return updated_actions[0].copy()

    @staticmethod
    def quaternion_distance_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        dot_products = np.einsum("ij,ij->i", q1, q2)
        return 1.0 - np.abs(dot_products)

    def quadruped_cost_np(
        self, x: np.ndarray, u: np.ndarray, x_ref: np.ndarray
    ) -> np.ndarray:
        """Original locomotion stage cost with B2-Z1 joint slices."""
        x_error = x - x_ref
        q_dist = self.quaternion_distance_np(x[:, 3:7], x_ref[:, 3:7])
        x_error[:, 3:7] = q_dist[:, None]

        x_joint = x[:, self.layout.joint_position]
        v_joint = x[:, self.layout.joint_velocity]
        u_error = self.cost_kp * (u - x_joint) - self.cost_kd * v_joint

        x_error[:, :3] = 0.0
        x_pos_error = x[:, :3] - x_ref[:, :3]
        position_cost = np.abs(np.dot(x_pos_error, self.Q[:3, :3])).sum(axis=1)
        return (
            np.einsum("ij,ik,jk->i", x_error, x_error, self.Q)
            + np.einsum("ij,ik,jk->i", u_error, u_error, self.R)
            + position_cost
        )

    def calculate_total_cost(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        joints_ref: np.ndarray,
        body_ref: np.ndarray,
    ) -> np.ndarray:
        """Assemble the original body + gait reference in the 45-D layout."""
        num_samples, num_pairs, state_dim = states.shape
        if joints_ref.shape != (32, num_pairs):
            raise ValueError(
                f"Expected joints_ref shape {(32, num_pairs)}, got {joints_ref.shape}"
            )
        if body_ref.shape != (13,):
            raise ValueError(f"Expected body_ref shape (13,), got {body_ref.shape}")

        traj_body_ref = np.repeat(
            body_ref[np.newaxis, :], num_samples * num_pairs, axis=0
        )
        flat_states = states.reshape(-1, state_dim).copy()
        flat_actions = actions.reshape(-1, actions.shape[2])
        tiled_joints_ref = np.tile(joints_ref.T, (num_samples, 1))
        x_ref = np.concatenate(
            (
                traj_body_ref[:, :7],
                tiled_joints_ref[:, :16],
                traj_body_ref[:, 7:],
                tiled_joints_ref[:, 16:],
            ),
            axis=1,
        )
        # Keep the package-level full-reference inspection contract while the
        # cost assembly above remains identical to the original source flow.
        self.reference = x_ref[:num_pairs].copy()

        flat_states[:, self.layout.base_linear_velocity] = (
            batch_world_to_local_velocity(
                flat_states[:, self.layout.base_quaternion],
                flat_states[:, self.layout.base_linear_velocity],
            )
        )
        costs = self.quadruped_cost_np(flat_states, flat_actions, x_ref)
        return costs.reshape(num_samples, num_pairs).sum(axis=1)

    def eval_best_trajectory(self) -> float | None:
        if self.obs is None:
            return None
        full_state_dim = mujoco.mj_stateSize(
            self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value
        )
        best_rollout = np.empty((1, self.horizon, full_state_dim), dtype=float)
        initial_state = np.concatenate(([0.0], self.obs))[None, :]
        rollout.rollout(
            self.model,
            mujoco.MjData(self.model),
            skip_checks=False,
            initial_state=initial_state,
            control=self.selected_trajectory[None, :, :],
            state=best_rollout,
        )
        return float(
            self.cost_func(
                best_rollout[:, :, 1:],
                self.selected_trajectory[None, :, :],
                self.joints_ref,
                self.body_ref,
            )[0]
        )

    def advance_reference(self, steps: int) -> None:
        """Align gait phase and warm-start with the executed physics steps.

        ``update()`` already advances both by one step, matching the original
        100 Hz controller. This method advances only the remaining steps when
        a simulator deliberately executes more than one control per update.
        """
        executed_steps = int(steps)
        if executed_steps != steps or executed_steps < 0:
            raise ValueError("Executed control steps must be a non-negative integer")

        if executed_steps == 0:
            # No selected action reached the plant: undo update()'s one-step
            # scheduler/warm-start assumption.
            self.gait_scheduler.phase = (
                self.gait_scheduler.phase - 1
            ) % self.gait_scheduler.cycle_steps
            self.trajectory = self.selected_trajectory.copy()
            return

        extra_steps = executed_steps - 1
        self.gait_scheduler.advance(extra_steps)
        if extra_steps >= self.horizon:
            self.trajectory[:] = self.trajectory[-1]
        elif extra_steps > 0:
            terminal_action = self.trajectory[-1].copy()
            self.trajectory[:-extra_steps] = self.trajectory[extra_steps:]
            self.trajectory[-extra_steps:] = terminal_action
        self.reference_step += executed_steps


# B2-Z1 package name plus the original source class name are both supported.
B2Z1LocomotionMPPI = MPPI


if __name__ == "__main__":
    controller = MPPI()
    controller.close()
