"""B2-Z1 hold MPPI using the original locomotion-controller cost form."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from legged_mani.control.base_controller import BaseMPPI
from legged_mani.interface.environment import MODEL_PATH
from legged_mani.utils.state_layout import B2Z1StateLayout


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "mppi_sit_hold.yml"


class B2Z1PoseHoldMPPI(BaseMPPI):
    """Hold the state observed at the beginning of each MPPI update.

    This keeps the state and virtual PD-effort terms from the original
    ``mppi_locomotion.py`` cost, but removes gait files, gait schedulers, goal
    switching, and generated joint references. The complete 45-D observation
    is repeated as the reference over the prediction horizon.
    """

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        config_path: str | Path = DEFAULT_CONFIG,
    ):
        super().__init__(model_path, config_path)
        if (self.model.nq, self.model.nv, self.model.nu) != (23, 22, 16):
            raise ValueError("B2Z1PoseHoldMPPI requires the 23/22/16 B2-Z1 model")

        self.layout = B2Z1StateLayout()
        q_diag = np.asarray(self.params["Q_diag"], dtype=float)
        r_diag = np.asarray(self.params["R_diag"], dtype=float)
        if q_diag.shape != (self.layout.state_dim,):
            raise ValueError(
                f"Q_diag must have {self.layout.state_dim} entries, got {q_diag.shape}"
            )
        if r_diag.shape != (self.act_dim,):
            raise ValueError(
                f"R_diag must have {self.act_dim} entries, got {r_diag.shape}"
            )
        self.Q = np.diag(q_diag)
        self.R = np.diag(r_diag)

        gains = self.params.get("virtual_pd_gains", {})
        self.cost_kp = float(gains.get("kp", 50.0))
        self.cost_kd = float(gains.get("kd", 3.0))
        self.reference = None
        self.obs = None
        self.last_costs = np.full(self.n_samples, np.nan)
        self.exp_weights = np.ones(self.n_samples) / self.n_samples
        self.cost_func = self.calculate_total_cost

    @staticmethod
    def quaternion_distance_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Quaternion distance used by the original locomotion cost."""
        return 1.0 - np.abs(np.einsum("bi,bi->b", q1, q2))

    def quadruped_cost_np(
        self, x: np.ndarray, u: np.ndarray, x_ref: np.ndarray
    ) -> np.ndarray:
        """Locomotion-style state, position-L1, and virtual PD-effort cost."""
        x_error = x - x_ref

        quaternion_slice = self.layout.base_quaternion
        quaternion_distance = self.quaternion_distance_np(
            x[:, quaternion_slice], x_ref[:, quaternion_slice]
        )
        x_error[:, quaternion_slice] = quaternion_distance[:, None]

        joint_position = x[:, self.layout.joint_position]
        joint_velocity = x[:, self.layout.joint_velocity]
        u_error = (
            self.cost_kp * (u - joint_position)
            - self.cost_kd * joint_velocity
        )

        position_error = x[:, self.layout.base_position] - x_ref[:, self.layout.base_position]
        position_l1_cost = np.abs(position_error @ self.Q[:3, :3]).sum(axis=1)
        x_error[:, self.layout.base_position] = 0.0

        state_cost = np.einsum("bi,ij,bj->b", x_error, self.Q, x_error)
        effort_cost = np.einsum("bi,ij,bj->b", u_error, self.R, u_error)
        return state_cost + effort_cost + position_l1_cost

    def calculate_total_cost(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        reference: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sum locomotion-style stage costs over every sampled horizon."""
        if reference is None:
            reference = self.reference
        if reference is None:
            raise RuntimeError("The current-state reference is not initialized")

        sample_count, horizon, state_dim = states.shape
        flat_states = states.reshape(-1, state_dim)
        flat_actions = actions.reshape(-1, self.act_dim)
        flat_reference = np.broadcast_to(
            np.asarray(reference, dtype=float),
            (sample_count * horizon, state_dim),
        )
        stage_cost = self.quadruped_cost_np(
            flat_states, flat_actions, flat_reference
        )
        return stage_cost.reshape(sample_count, horizon).sum(axis=1)

    def update(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=float)
        if observation.shape != (self.layout.state_dim,):
            raise ValueError(
                f"Expected observation shape {(self.layout.state_dim,)}, "
                f"got {observation.shape}"
            )

        # Save the latest observation, but initialize the hold reference only once.
        self.obs = observation.copy()

        if self.reference is None:
            self.reference = observation.copy()

            # Static hold: keep the initial pose but target zero velocities.
            self.reference[self.layout.base_linear_velocity] = 0.0
            self.reference[self.layout.base_angular_velocity] = 0.0
            self.reference[self.layout.joint_velocity] = 0.0
        actions = self.perturb_action()
        states = self.rollout_func(observation, actions)
        costs = self.cost_func(states, actions, self.reference)
        if not np.isfinite(costs).all():
            raise FloatingPointError("Non-finite MPPI rollout cost")

        minimum_cost = float(np.min(costs))
        normalized_costs = costs - minimum_cost

        weights = np.exp(-normalized_costs / self.temperature)

        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 1e-12:
            weights = np.full(self.n_samples, 1.0 / self.n_samples)
        else:
            weights /= weight_sum
        updated = np.einsum("n,nha->ha", weights, actions)
        updated = np.clip(updated, self.act_min, self.act_max)

        self.last_costs = costs
        self.exp_weights = weights
        self.selected_trajectory = updated.copy()
        self.trajectory[:-1] = updated[1:]
        self.trajectory[-1] = updated[-1]
        return updated[0].copy()

    def eval_best_trajectory(self) -> float:
        return float(np.min(self.last_costs))


# Backward-compatible name used by the first sit-hold integration scripts.
B2Z1SitHoldMPPI = B2Z1PoseHoldMPPI
