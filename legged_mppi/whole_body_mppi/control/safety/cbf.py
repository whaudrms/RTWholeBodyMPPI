"""Velocity-aware DCBF penalties and a locally linearized predictive filter.

This is an experimental, soft-constrained filter, not a safety certificate for
hybrid contact dynamics. Its nonlinear verification and infeasibility flag are
part of the public diagnostics. No per-sample QP or Python physics loop is used.
"""
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Optional

import mujoco
import numpy as np
from scipy.optimize import minimize
from whole_body_mppi.control.costs.end_effector import quaternion_matrix


@dataclass(frozen=True)
class CBFOptions:
    tilt_degrees: float = 45.
    min_world_height: float = .12
    decay_rate: float = 10.
    lookahead_seconds: float = .05
    joint_lookahead_seconds: float = .02
    penalty_weight: float = 2000.
    shield_steps: int = 8
    correction_limit: float = .12
    finite_difference: float = .002
    slack_weight: float = 10000.
    tolerance: float = 1e-5
    ee_radius: Optional[float] = None  # default: task ee_position_tolerance [m]
    ee_initial_margin: float = .05
    ee_convergence_rate: float = .5
    ee_lookahead_seconds: float = .05
    body_speed_max: float = .6  # body-frame XY norm [m/s], cmd_vel only
    ee_speed_max: float = .6  # world-frame XYZ norm [m/s], cmd_vel only
    body_velocity_error_max: float = .25  # body-frame XY command-error norm [m/s]
    ee_velocity_error_max: float = .25  # world-frame XYZ command-error norm [m/s]
    ee_penalty_scale: float = .1
    body_height_tolerance: Optional[float] = None  # relative to body_ref z [m]
    body_angular_speed_max: Optional[float] = None  # roll/pitch angular-speed norm [rad/s]
    leg_correction_limit: float = .02
    leg_correction_weight: float = 25.
    arm_only_ee_correction: bool = True
    body_tilt_guard_degrees: float = 2.
    body_height_guard: float = .01
    body_velocity_guard: float = .1
    body_angular_velocity_guard: float = .3


class PredictiveCBF:
    def __init__(self, agent, enabled, options=None):
        self.agent = agent
        self.enabled = enabled
        self.options = CBFOptions(**(options or {}))
        p = self.options
        positive = (p.decay_rate, p.lookahead_seconds, p.joint_lookahead_seconds,
                    p.penalty_weight, p.correction_limit, p.finite_difference,
                    p.slack_weight, p.tolerance, p.ee_initial_margin,
                    p.ee_convergence_rate, p.ee_lookahead_seconds, p.body_speed_max,
                    p.ee_speed_max, p.body_velocity_error_max, p.ee_velocity_error_max,
                    p.ee_penalty_scale, p.leg_correction_limit, p.leg_correction_weight,
                    p.body_tilt_guard_degrees, p.body_height_guard,
                    p.body_velocity_guard, p.body_angular_velocity_guard)
        if (not np.all(np.isfinite(positive)) or min(positive) <= 0
                or not np.isfinite(p.min_world_height) or p.min_world_height < 0
                or not np.isfinite(p.tilt_degrees) or not 0 < p.tilt_degrees < 90
                or type(p.shield_steps) is not int or p.shield_steps < 1
                or (p.ee_radius is not None and (not np.isfinite(p.ee_radius) or p.ee_radius <= 0))):
            raise ValueError('Invalid cbf configuration')
        for value in (p.body_height_tolerance, p.body_angular_speed_max):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError('Body operating limits must be finite and positive')
        if type(p.arm_only_ee_correction) is not bool:
            raise ValueError('arm_only_ee_correction must be boolean')
        model = agent.model
        joints = np.unique(model.actuator_trnid[:, 0])
        joints = joints[(joints >= 0) & (joints < model.njnt)]
        joints = joints[(model.jnt_limited[joints] != 0)
                        & (model.jnt_type[joints] == mujoco.mjtJoint.mjJNT_HINGE)]
        self.qadr = model.jnt_qposadr[joints]
        self.vadr = model.jnt_dofadr[joints]
        self.lower = model.jnt_range[joints, 0]
        self.upper = model.jnt_range[joints, 1]
        self.ranges = self.upper-self.lower
        self.last = {}
        self.extra_rollouts = 0
        self.ee_enabled = (hasattr(agent, 'ee_objective')
                           and agent.task_data.get('control_mode') != 'cmd_vel')
        self.velocity_enabled = agent.task_data.get('control_mode') == 'cmd_vel'
        # One group per scalar barrier: body=0, joints=1, EE=2. Preserve the
        # last EE/velocity channels used by report consumers.
        groups = [0, 0]+[1]*(2*len(self.qadr))
        if p.body_height_tolerance is not None:
            groups += [0, 0]
        if p.body_angular_speed_max is not None:
            groups += [0]
        if self.ee_enabled:
            groups += [2]
        if self.velocity_enabled:
            groups += [0, 2, 0, 2]
        self.barrier_groups = np.asarray(groups)
        self.time = 0.
        self.external_clock = False
        self._ee_target = None
        self._ee_waypoint = None
        self._ee_frame = None
        self._ee_start_time = 0.
        self._ee_start_radius = 0.
        self.ee_final_radius = (float(p.ee_radius if p.ee_radius is not None
                                      else agent.task_data['ee_position_tolerance'])
                                if self.ee_enabled else None)
        if self.ee_enabled and (not np.isfinite(self.ee_final_radius) or self.ee_final_radius <= 0):
            raise ValueError('EE CBF radius must be finite and positive')

    def configuration(self):
        return dict(enabled=self.enabled, method='velocity-aware DCBF cost + predictive soft QP filter',
                    options=asdict(self.options),
                    ee=dict(active=self.ee_enabled and self.enabled, monitored=self.ee_enabled,
                            final_radius=self.ee_final_radius,
                            frame=self.agent.ee_objective.frame if self.ee_enabled else None,
                            type='exponentially shrinking waypoint-error ball; reset only on waypoint change'),
                    velocity=dict(active=self.velocity_enabled and self.enabled,
                                  monitored=self.velocity_enabled, body_frame='body XY', ee_frame='world XYZ',
                                  type='speed norms and command-error norms; direct discrete-time barriers',
                                  command_transitions='Predicted phase schedule; abrupt jumps can be infeasible'),
                    priority='Separate body/joint/EE slacks; protected body/joint constraints cannot degrade below nominal violation floors. Full-horizon body-motion guards.',
                    height_frame='world z; a floor-height bound, not terrain clearance',
                    guarantee='None: local linearization, soft constraints, finite prediction window; inspect infeasible flag')

    def prepare_reference(self, obs):
        """Anchor the funnel to measured state, never to a sampled prediction."""
        if not self.ee_enabled:
            return
        objective = self.agent.ee_objective
        waypoint = getattr(self.agent, 'waypoint_index', None)
        changed = (self._ee_target is None or waypoint != self._ee_waypoint
                   or objective.frame != self._ee_frame
                   or not np.array_equal(objective.target, self._ee_target))
        if changed:
            x = np.asarray(obs)
            if x.ndim != 1:
                raise ValueError('Initialize EE CBF reference from a single measured observation')
            position = objective.positions(x[:self.agent.model.nq])
            error = np.linalg.norm(position-objective.target)
            self._ee_target = objective.target.copy()
            self._ee_waypoint, self._ee_frame = waypoint, objective.frame
            self._ee_start_time = self.time
            self._ee_start_radius = max(self.ee_final_radius, error+self.options.ee_initial_margin)

    def observe(self, obs, time):
        """Use physics time, including time spent executing a stale sync plan."""
        if time < self.time:
            self._ee_target = None
        self.time = float(time)
        self.external_clock = True
        self.prepare_reference(obs)

    def ee_envelope(self, time_offset=0.):
        elapsed = np.maximum(0., self.time-self._ee_start_time+np.asarray(time_offset))
        excess = (self._ee_start_radius-self.ee_final_radius)*np.exp(-self.options.ee_convergence_rate*elapsed)
        return self.ee_final_radius+excess, -self.options.ee_convergence_rate*excess

    def velocity_values(self, obs, time_offset=0.):
        """Actual velocity and phase-correct commands, without position goals."""
        a = self.agent
        x = np.asarray(obs)
        q, v = x[..., :a.model.nq], x[..., a.model.nq:a.model.nq+a.model.nv]
        rotation = quaternion_matrix(q[..., 3:7])
        body = np.einsum('...ji,...j->...i', rotation, v[..., :3])[..., :2]
        _, ee = a.ee_position_velocity(q, v)
        indices = a.phase_indices(a.command_time+np.asarray(time_offset))
        return body, ee, a.body_commands[indices], a.ee_commands[indices]

    def velocity_barriers(self, obs, time_offset=0.):
        p = self.options
        body, ee, body_command, ee_command = self.velocity_values(obs, time_offset)
        return np.stack((1-np.sum(body*body, axis=-1)/p.body_speed_max**2,
                         1-np.sum(ee*ee, axis=-1)/p.ee_speed_max**2,
                         1-np.sum((body-body_command)**2, axis=-1)/p.body_velocity_error_max**2,
                         1-np.sum((ee-ee_command)**2, axis=-1)/p.ee_velocity_error_max**2), axis=-1)

    def barriers(self, obs, time_offset=0.):
        """Normalized h and b=h+tau*h_dot, for arbitrary leading dimensions.

        obs is qpos+qvel, with MuJoCo world linear / body angular velocity.
        Tilt uses body z dot world z and is independent of heading.
        """
        a, p = self.agent, self.options
        x = np.asarray(obs)
        q, v = x[..., :a.model.nq], x[..., a.model.nq:a.model.nq+a.model.nv]
        quat = q[..., 3:7]
        quat = quat/np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-12)
        w, qx, qy, qz = np.moveaxis(quat, -1, 0)
        r20, r21 = 2*(qx*qz-w*qy), 2*(qy*qz+w*qx)
        scale = 1-np.cos(np.deg2rad(p.tilt_degrees))
        tilt = (1-2*(qx*qx+qy*qy)-np.cos(np.deg2rad(p.tilt_degrees)))/scale
        tilt_dot = (r20*v[..., 4]-r21*v[..., 3])/scale
        height = (q[..., 2]-p.min_world_height)/.15
        height_dot = v[..., 2]/.15
        jl = (q[..., self.qadr]-self.lower)/self.ranges
        ju = (self.upper-q[..., self.qadr])/self.ranges
        jdot = v[..., self.vadr]/self.ranges
        h = np.concatenate((tilt[..., None], height[..., None], jl, ju), axis=-1)
        derivative = np.concatenate((tilt_dot[..., None], height_dot[..., None], jdot, -jdot), axis=-1)
        tau = np.r_[np.full(2, p.lookahead_seconds),
                    np.full(2*len(self.qadr), p.joint_lookahead_seconds)]
        b = h+tau*derivative
        if p.body_height_tolerance is not None:
            reference = np.asarray(a.body_ref)[..., 2]
            height_error = (q[..., 2]-reference)/p.body_height_tolerance
            speed = v[..., 2]/p.body_height_tolerance
            operating_h = np.stack((1+height_error, 1-height_error), axis=-1)
            operating_b = operating_h+p.lookahead_seconds*np.stack((speed, -speed), axis=-1)
            h = np.concatenate((h, operating_h), axis=-1)
            b = np.concatenate((b, operating_b), axis=-1)
        if p.body_angular_speed_max is not None:
            angular_h = 1-np.sum(v[..., 3:5]**2, axis=-1)/p.body_angular_speed_max**2
            h = np.concatenate((h, angular_h[..., None]), axis=-1)
            b = np.concatenate((b, angular_h[..., None]), axis=-1)
        if self.ee_enabled:
            if self._ee_target is None:
                self.prepare_reference(obs)
            position, velocity = a.ee_objective.position_velocity(q, v)
            error = position-self._ee_target
            radius, radius_dot = self.ee_envelope(time_offset)
            scale = self._ee_start_radius**2
            ee_h = (radius**2-np.sum(error*error, axis=-1))/scale
            ee_dot = (2*radius*radius_dot-2*np.sum(error*velocity, axis=-1))/scale
            h = np.concatenate((h, ee_h[..., None]), axis=-1)
            b = np.concatenate((b, (ee_h+p.ee_lookahead_seconds*ee_dot)[..., None]), axis=-1)
        if self.velocity_enabled:
            speed_h = self.velocity_barriers(obs, time_offset)
            # Speed is already a velocity-level constraint. The simulator gives
            # its next value; do not invent an acceleration or apply h+tau*h_dot.
            h = np.concatenate((h, speed_h), axis=-1)
            b = np.concatenate((b, speed_h), axis=-1)
        return h, b

    def constraints(self, states, obs):
        """Include current->first-predicted transition; never mutate states."""
        self.prepare_reference(obs)
        offsets = self.agent.h*np.arange(1, states.shape[1]+1)
        h, b = self.barriers(states[..., 1:], offsets)
        _, b0 = self.barriers(obs)
        previous = np.concatenate((np.broadcast_to(b0, (len(states), 1, len(b0))), b[:, :-1]), axis=1)
        residual = b-np.exp(-self.options.decay_rate*self.agent.h)*previous
        return np.concatenate((h, b, residual), axis=-1)

    def penalty(self, states, obs):
        violations = np.minimum(self.constraints(states, obs), 0.)
        # Finite values keep all solver weighting/normalization paths defined.
        violations = np.nan_to_num(violations, nan=-1e3, posinf=0., neginf=-1e3)
        weights = np.tile(np.where(self.barrier_groups == 2, self.options.ee_penalty_scale, 1.), 3)
        return self.options.penalty_weight*(np.square(np.clip(violations, -1e3, 0.))*weights).sum(axis=(1, 2))

    def _predict(self, plans, obs, return_states=False):
        a = self.agent
        plans = np.ascontiguousarray(plans)
        initial = np.tile(np.r_[0., obs], (len(plans), 1))
        states = np.empty((len(plans), plans.shape[1],
                           mujoco.mj_stateSize(a.model, mujoco.mjtState.mjSTATE_FULLPHYSICS)))
        a.rollout_func(states, plans, initial, num_workers=a.num_workers, nstep=plans.shape[1])
        self.extra_rollouts += len(plans)
        values = self.constraints(states, obs).reshape(len(plans), -1)
        return (values, states) if return_states else values

    def _body_guard(self, nominal, candidates):
        """Compare physical body motion over the entire plan, not just 80ms."""
        p, nq = self.options, self.agent.model.nq
        def tilt(states):
            quat = states[..., 4:8]
            quat = quat/np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-12)
            return np.arccos(np.clip(1-2*np.sum(quat[..., 1:3]**2, axis=-1), -1., 1.))
        return (np.all(tilt(candidates) <= tilt(nominal)+np.deg2rad(p.body_tilt_guard_degrees), axis=1)
                & np.all(np.abs(candidates[..., 3]-nominal[..., 3]) <= p.body_height_guard, axis=1)
                & np.all(np.linalg.norm(candidates[..., 1+nq:4+nq]-nominal[..., 1+nq:4+nq], axis=-1) <= p.body_velocity_guard, axis=1)
                & np.all(np.linalg.norm(candidates[..., 4+nq:7+nq]-nominal[..., 4+nq:7+nq], axis=-1) <= p.body_angular_velocity_guard, axis=1))

    def filter_plan(self, actions, obs):
        """One soft QP when needed, then nonlinear verification of its result.

        The correction has one variable per free actuator and decays along the
        plan. If verification fails, retain whichever evaluated plan has the
        smallest worst violation, and report infeasible. Never call it safe.
        """
        started = perf_counter()
        a, p = self.agent, self.options
        self.extra_rollouts = 0
        actions = np.clip(actions, a.act_min, a.act_max)
        count = min(p.shield_steps, len(actions))
        base = actions[:count]
        nominal_values, nominal_states = self._predict(actions[None], obs, return_states=True)
        full_g0 = nominal_values[0]
        width = 3*len(self.barrier_groups)
        g0 = full_g0[:count*width]
        groups = np.tile(self.barrier_groups, 3*count)
        full_groups = np.tile(self.barrier_groups, 3*len(actions))
        protected = full_groups != 2
        protected_floor = np.minimum(full_g0[protected], 0.)
        if not np.all(np.isfinite(full_g0)):
            raise FloatingPointError('Non-finite nominal CBF prediction')
        result = actions.copy()
        best_g = full_g0
        qp_success = False
        body_guard_rejected = False
        physical = full_g0.reshape(len(actions), 3, -1)[:, :2, :][..., self.barrier_groups == 0]
        body_needs_recovery = bool(np.any(physical < -p.tolerance))
        # When the nominal whole-body plan already leaves the body operating
        # envelope, let MPPI restore locomotion instead of spending leg authority
        # on a reactive EE correction. This also avoids an infeasible large QP.
        defer_ee = bool(a.act_dim > 12 and p.arm_only_ee_correction and body_needs_recovery)
        if full_g0.min() < -p.tolerance and not defer_ee:
            free = np.flatnonzero(a.act_max-a.act_min > 1e-10)
            if a.act_dim > 12 and p.arm_only_ee_correction and not body_needs_recovery:
                free = free[free >= 12]
            # Saturation-aware one-sided differences; fixed arm axes excluded.
            taper = np.exp(-np.arange(len(actions))/max(count, 1))
            directions = np.zeros((len(free), a.act_dim))
            for row, axis in enumerate(free):
                sign = 1 if base[0, axis]+p.finite_difference <= a.act_max[axis] else -1
                directions[row, axis] = sign*p.finite_difference
            if len(free):
                probes = np.clip(base[None]+taper[None, :count, None]*directions[:, None], a.act_min, a.act_max)
                gp = self._predict(probes, obs)
                if not np.all(np.isfinite(gp)):
                    raise FloatingPointError('Non-finite CBF finite-difference prediction')
                jac = (gp-g0).T / directions[np.arange(len(free)), free]
                # Separate slacks prevent an EE violation from relaxing the
                # body/joint constraints. Protected rows also keep their own
                # nominal violation floor; delta=0 remains feasible.
                matrix = np.column_stack((jac, np.eye(3)[groups]))
                action_weights = np.where(free < 12, p.leg_correction_weight, 1.)
                slack_weights = np.array([100., 10., 1.])
                def objective(z):
                    return .5*np.dot(action_weights*z[:-3], z[:-3])/p.slack_weight+.5*np.dot(slack_weights*z[-3:], z[-3:])
                def gradient(z):
                    return np.r_[action_weights*z[:-3]/p.slack_weight, slack_weights*z[-3:]]
                slack0 = [max(0., -g0[groups == group].min())+p.tolerance
                          if np.any(groups == group) else 0. for group in range(3)]
                initial = np.r_[np.zeros(len(free)), slack0]
                limits = np.where((free < 12) & (a.act_dim > 12),
                                  min(p.leg_correction_limit, p.correction_limit), p.correction_limit)
                bounds = [(max(-limit, a.act_min[j]-base[0, j]),
                           min(limit, a.act_max[j]-base[0, j])) for j, limit in zip(free, limits)]+[(0., None)]*3
                keep = groups != 2
                keep_matrix = np.column_stack((jac[keep], np.zeros((np.sum(keep), 3))))
                keep_offset = g0[keep]-np.minimum(g0[keep], 0.)+p.tolerance
                qp = minimize(objective, initial, jac=gradient, bounds=bounds, method='SLSQP',
                              constraints=[dict(type='ineq', fun=lambda z: g0+matrix@z,
                                                jac=lambda z: matrix),
                                           dict(type='ineq', fun=lambda z: keep_offset+keep_matrix@z,
                                                jac=lambda z: keep_matrix)],
                              options=dict(maxiter=40, ftol=1e-10))
                qp_success = bool(qp.success)
                if np.all(np.isfinite(qp.x)):
                    delta = np.zeros(a.act_dim)
                    delta[free] = qp.x[:-3]
                    scales = np.array([1., .5, .25, .125])
                    candidates = np.clip(actions[None]+scales[:, None, None]*taper[None, :, None]*delta,
                                         a.act_min, a.act_max)
                    checked, states = self._predict(candidates, obs, return_states=True)
                    guard = self._body_guard(nominal_states[0], states)
                    keep_ok = np.all(checked[:, protected] >= protected_floor-p.tolerance, axis=1)
                    finite = np.all(np.isfinite(checked), axis=1)
                    worst_ok = checked.min(axis=1) >= full_g0.min()-1e-12
                    weights = np.where(full_groups == 2, p.ee_penalty_scale, 1.)
                    scores = (np.minimum(checked, 0.)**2*weights).sum(axis=1)
                    score0 = np.sum(np.minimum(full_g0, 0.)**2*weights)
                    valid = np.flatnonzero(guard & keep_ok & finite & worst_ok & (scores < score0-1e-12))
                    body_guard_rejected = bool(np.linalg.norm(delta) > 1e-8 and not len(valid))
                    if len(valid):
                        index = valid[np.argmin(scores[valid])]
                        result, best_g = candidates[index], checked[index]
        self.last = dict(nominal_min_residual=float(full_g0.min()),
                         verified_min_residual=float(best_g.min()),
                         correction_norm=float(np.linalg.norm(result[0]-actions[0])),
                         leg_correction_norm=float(np.linalg.norm(result[0, :12]-actions[0, :12])),
                         arm_correction_norm=float(np.linalg.norm(result[0, 12:]-actions[0, 12:])),
                         body_guard_rejected=body_guard_rejected,
                         ee_correction_deferred=defer_ee,
                         nominal_body_residual=float(full_g0[full_groups == 0].min()),
                         verified_body_residual=float(best_g[full_groups == 0].min()),
                         infeasible=bool(best_g.min() < -p.tolerance), qp_success=qp_success,
                         extra_rollouts=self.extra_rollouts, shield_seconds=perf_counter()-started)
        return result


def configure_cbf(agent, mode='off'):
    if mode not in ('on', 'off'):
        raise ValueError('cbf must be on or off')
    agent.cbf = PredictiveCBF(agent, mode == 'on', agent.params.get('cbf'))
