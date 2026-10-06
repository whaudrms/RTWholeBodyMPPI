"""CPU spline search with optional Hermite PD targets and annealing.

Hermite commands are encoded for affine position actuators as
ctrl = q_ref + (Kd / Kp) * dq_ref. This preserves force saturation in MuJoCo.
The encoded ctrl is NOT a joint-position target for a hardware interface.
"""
from __future__ import annotations

import time
import numpy as np
import mujoco
from scipy.interpolate import CubicHermiteSpline


def bounded_hermite(times, positions, velocities, query, lower, upper):
    """Bound the entire position spline using its equivalent Bezier controls."""
    positions = np.clip(positions, lower, upper)
    widths = np.diff(times)
    adjacent = np.maximum(np.r_[widths[0], widths], np.r_[widths, widths[-1]])
    # q0 + dt*v0/3 and q1 - dt*v1/3 stay within joint bounds.
    limit = 3.0 * np.minimum(upper - positions, positions - lower) / adjacent[:, None]
    velocities = np.clip(velocities, -limit, limit)
    curve = CubicHermiteSpline(times, positions, velocities, axis=-2)
    return curve(query), curve(query, 1)


def configure_pd_target_model(model):
    """Validate fixed-gain joint PD actuators and allow velocity feedforward."""
    kp = model.actuator_gainprm[:, 0]
    kd = -model.actuator_biasprm[:, 2]
    if (np.any(kp <= 0) or np.any(kd < 0)
            or not np.allclose(model.actuator_biasprm[:, 0], 0)
            or not np.allclose(model.actuator_biasprm[:, 1], -kp)
            or np.any(model.actuator_gaintype != mujoco.mjtGain.mjGAIN_FIXED)
            or np.any(model.actuator_biastype != mujoco.mjtBias.mjBIAS_AFFINE)
            or np.any(model.actuator_dyntype != mujoco.mjtDyn.mjDYN_NONE)
            or np.any(model.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT)
            or not np.allclose(model.actuator_gear[:, 0], 1)
            or not np.allclose(model.actuator_gear[:, 1:], 0)):
        raise ValueError("Hermite requires unit-gear, fixed-gain affine joint PD actuators")
    # Joint target bounds are enforced by bounded_hermite. Encoded control can
    # exceed those bounds due to velocity feedforward; actuator force limits stay on.
    model.actuator_ctrllimited[:] = False
    return kd / kp


class AnnealedSearchMixin:
    def _configure_rollout_mode(self, rollout_mode):
        super()._configure_rollout_mode(
            "original_spline" if rollout_mode == "hermite" else rollout_mode
        )
        if rollout_mode == "hermite":
            self.rollout_mode = "hermite"
            self.sample_type = "hermite"

    def _configure_search(self, params, anneal_iterations=None, horizon_noise_factor=None):
        self.anneal_iterations = (params.get("anneal_iterations", 3)
                                  if anneal_iterations is None else anneal_iterations)
        if (isinstance(self.anneal_iterations, bool)
                or int(self.anneal_iterations) != self.anneal_iterations
                or self.anneal_iterations < 1):
            raise ValueError("anneal_iterations must be a positive integer")
        self.anneal_iterations = int(self.anneal_iterations)
        self.anneal_factor = float(params.get("anneal_factor", 0.5))
        self.horizon_noise_factor = float(
            params.get("horizon_noise_factor", 0.9)
            if horizon_noise_factor is None else horizon_noise_factor
        )
        self.hermite_velocity_noise_scale = float(params.get("hermite_velocity_noise_scale", 4.0))
        if not (0 < self.anneal_factor <= 1 and 0 < self.horizon_noise_factor <= 1):
            raise ValueError("annealing factors must be in (0, 1]")
        if not np.isfinite(self.hermite_velocity_noise_scale) or self.hermite_velocity_noise_scale < 0:
            raise ValueError("hermite_velocity_noise_scale must be finite and nonnegative")
        if self.horizon < 2 or not 2 <= self.n_knots <= self.horizon or self.n_samples < 2:
            raise ValueError("Search requires horizon >= knots >= 2 and at least two candidates")
        self.planning_budget_ms = float(params.get("planning_budget_ms", 0.0))
        if not np.isfinite(self.planning_budget_ms) or self.planning_budget_ms < 0:
            raise ValueError("planning_budget_ms must be finite and nonnegative")
        self._iteration_seconds_estimate = 0.0
        self.validate_mean = bool(params.get("validate_mean", True))
        self._iteration_scale = 1.0
        self._hermite_velocity_warm = None
        self._hermite_position_warm = None
        if self.rollout_mode == "hermite":
            self._pd_velocity_ratio = configure_pd_target_model(self.model)

    def configure_execution_model(self, model):
        if self.rollout_mode == "hermite":
            ratio = configure_pd_target_model(model)
            if (model.nu != self.act_dim or not np.allclose(ratio, self._pd_velocity_ratio)
                    or not np.allclose(model.actuator_gainprm, self.model.actuator_gainprm)
                    or not np.allclose(model.actuator_biasprm, self.model.actuator_biasprm)
                    or not np.array_equal(model.actuator_forcelimited, self.model.actuator_forcelimited)
                    or not np.allclose(model.actuator_forcerange, self.model.actuator_forcerange)):
                raise ValueError("Execution and rollout PD parameters must match")

    def generate_noise(self, size):
        noise = super().generate_noise(size)
        factor = getattr(self, "horizon_noise_factor", 1.0)
        scale = getattr(self, "_iteration_scale", 1.0)
        return noise * (scale * factor ** np.arange(size[1])[::-1])[None, :, None]

    def _encode_pd_targets(self, q, v):
        return q + self._pd_velocity_ratio * v

    def _hermite_candidates(self, q, v):
        indices = self._rollout_spline_indices()
        times = indices * self.model.opt.timestep
        shape = (self.n_samples, len(indices), self.act_dim)
        q_nodes = q[indices][None] + self.generate_noise(shape)
        v_nodes = v[indices][None] + self.generate_noise(shape) * self.hermite_velocity_noise_scale
        # The nominal itself is re-evaluated at every iteration.
        q_nodes[0], v_nodes[0] = q[indices], v[indices]
        return bounded_hermite(times, q_nodes, v_nodes,
                               np.arange(self.horizon) * self.model.opt.timestep, self.act_min, self.act_max)

    def _update_annealed(self, obs):
        started = time.perf_counter()
        self.last_timing = dict.fromkeys(("reference", "sampling", "rollout", "cost", "selection"), 0.0)
        self.obs = np.asarray(obs, dtype=float)
        self._update_motion_reference(self.obs)
        self._update_arm_reference(self.obs)
        if self.nominal_from_gait:
            self._refresh_gait_nominal()
        self.joints_ref = self._joint_reference()
        nominal_actions = self.joints_ref[:self.act_dim].T
        self.last_timing["reference"] = time.perf_counter() - started
        hermite = self.rollout_mode == "hermite"
        q = self.trajectory.copy()
        if hermite:
            if self._hermite_velocity_warm is None:
                v = np.gradient(q, self.model.opt.timestep, axis=0)
            else:
                # Account for gait/IK changes made since the last horizon shift.
                v = self._hermite_velocity_warm + np.gradient(
                    q - self._hermite_position_warm, self.model.opt.timestep, axis=0)
        incumbent = None
        best_q = best_v = None
        self.iteration_diagnostics = []
        self.budget_limited = False
        try:
            for iteration in range(self.anneal_iterations):
                elapsed = time.perf_counter() - started
                # Always produce one freshly evaluated plan. Additional batches
                # start only when their estimated duration fits the soft deadline.
                if (iteration > 0 and self.planning_budget_ms > 0
                        and elapsed + 1.5 * self._iteration_seconds_estimate + .003
                        > self.planning_budget_ms / 1000.0):
                    self.budget_limited = True
                    break
                iteration_started = time.perf_counter()
                self._iteration_scale = self.anneal_factor ** iteration
                tick = time.perf_counter()
                if hermite:
                    qs, vs = self._hermite_candidates(q, v)
                    if incumbent is not None:
                        qs[0], vs[0] = best_q, best_v
                        # Evaluate the previous weighted mean in the normal batch.
                        if not self.validate_mean and self.n_samples > 2:
                            qs[1], vs[1] = q, v
                    controls = self._encode_pd_targets(qs, vs)
                else:
                    controls = self.perturb_action()
                    controls[0] = (self._noise_free_rollout_candidate()
                                   if incumbent is None else incumbent)
                self.last_timing["sampling"] += time.perf_counter() - tick
                tick = time.perf_counter()
                states = self.rollout_func(self.obs, controls)
                self.last_timing["rollout"] += time.perf_counter() - tick
                tick = time.perf_counter()
                costs = self.calculate_total_cost(states, controls, self.joints_ref,
                                                  self.body_ref, rollout_sensors=self.last_rollout_sensors)
                self.last_timing["cost"] += time.perf_counter() - tick
                tick = time.perf_counter()
                if hermite:
                    valid = (self.collision_valid_rollouts & np.isfinite(costs)
                             & np.all(np.isfinite(controls), axis=(1, 2)))
                    if not np.any(valid):
                        self.execution_valid = False
                        self.cached_best_cost = float("inf")
                        self.selection_source = "rejected"
                        raise RuntimeError("No valid Hermite execution plan at the current state")
                    ids = np.flatnonzero(valid)
                    best = ids[np.argmin(costs[valid])]
                    best_q, best_v = qs[best].copy(), vs[best].copy()
                    self.best_sample_cost = float(costs[best])
                    span = np.ptp(costs[valid])
                    weights = (np.ones(len(ids)) if span < 1e-12 else
                               np.exp(-(costs[valid] - costs[best]) / span / self.temperature))
                    weights /= weights.sum()
                    self.exp_weights = np.zeros(self.n_samples)
                    self.exp_weights[valid] = weights
                    q = np.einsum("n,nij->ij", weights, qs[valid])
                    v = np.einsum("n,nij->ij", weights, vs[valid])
                    mean = self._encode_pd_targets(q, v)
                    mean_cost, mean_valid = (self._evaluate_execution_candidate(mean)
                                             if self.validate_mean else (float("nan"), False))
                    self.mean_candidate_cost = mean_cost
                    self.cached_best_cost = self.best_sample_cost
                    self.selection_source = "best_sample"
                    if mean_valid and mean_cost < self.best_sample_cost:
                        best_q, best_v = q.copy(), v.copy()
                        self.cached_best_cost = mean_cost
                        self.selection_source = "validated_mean"
                    incumbent = self._encode_pd_targets(best_q, best_v)
                    self.execution_valid = True
                else:
                    incumbent = self._select_updated_actions(controls, costs)
                    # Do not refresh the gait between inner iterations.
                    self.trajectory = incumbent.copy()
                self.last_timing["selection"] += time.perf_counter() - tick
                duration = time.perf_counter() - iteration_started
                self._iteration_seconds_estimate = max(
                    duration, .9 * self._iteration_seconds_estimate)
                self.iteration_diagnostics.append({
                    "noise_scale": self._iteration_scale,
                    "best_cost": self.cached_best_cost,
                    "source": self.selection_source,
                })
            self.last_safe_trajectory = incumbent.copy()
            evaluated_reference = self.joints_ref.copy()
            self._advance_selected_trajectory(best_q if hermite else incumbent, nominal_actions)
            # Keep cost diagnostics tied to the reference used during this solve.
            self.joints_ref = evaluated_reference
            if hermite:
                self.selected_position_targets = best_q.copy()
                self.selected_velocity_targets = best_v.copy()
                self.selected_trajectory = incumbent.copy()  # executable actuator commands
                self._hermite_position_warm = np.concatenate((best_q[1:], best_q[-1:]))
                self._hermite_velocity_warm = np.concatenate((best_v[1:], best_v[-1:]))
            return incumbent[0].copy()
        finally:
            self._iteration_scale = 1.0
            self.last_timing["total"] = time.perf_counter() - started
            self.deadline_missed = bool(self.planning_budget_ms > 0 and
                                        self.last_timing["total"] * 1000 > self.planning_budget_ms)
