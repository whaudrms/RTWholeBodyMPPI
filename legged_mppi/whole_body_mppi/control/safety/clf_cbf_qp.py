"""One-command Go1 CLF-CBF filter. Linearized QP, nonlinear acceptance check."""
from time import perf_counter
import numpy as np
import osqp
from scipy import sparse
from whole_body_mppi.control.costs.cmd_vel import heading_velocity
from whole_body_mppi.control.safety.shield import SafetyShield

from whole_body_mppi.control.safety.errors import FilterFailure


class CLFCBFQP:
    def __init__(self, agent, config, barrier_config):
        self.agent = agent
        self.cfg = dict(config)
        self.barrier = SafetyShield(agent, barrier_config)
        self.scale = np.asarray(self.cfg['velocity_scale'], dtype=float)
        self.limit = np.broadcast_to(np.asarray(self.cfg['max_correction'], dtype=float), (agent.act_dim,)).copy()
        if self.scale.shape != (3,) or not np.all(np.isfinite(self.scale) & (self.scale > 0)):
            raise ValueError('velocity_scale must contain three finite positive values')
        if not np.all(np.isfinite(self.limit) & (self.limit > 0)):
            raise ValueError('max_correction must be finite and positive')
        for key in ('decay_rate', 'slack_weight', 'fd_step', 'tolerance', 'solver_tolerance'):
            if not np.isfinite(self.cfg[key]) or self.cfg[key] <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if not isinstance(self.cfg['max_iterations'], int) or self.cfg['max_iterations'] < 1:
            raise ValueError('max_iterations must be a positive integer')
        self.rho = np.exp(-self.cfg['decay_rate'] * agent.h)
        self.history = []
        self.failure = None
        self.previous_solution = None

    def values(self, states, reference):
        """Return [V, h_height, h_tilt] from full MuJoCo physics states."""
        states = np.atleast_2d(states)
        velocity, yaw_rate, _ = heading_velocity(states[:, 4:8], states[:, 20:23], states[:, 23:26])
        error = (np.column_stack((velocity[:, :2], yaw_rate)) - reference) / self.scale
        V = .5 * np.sum(error**2, axis=1)
        return np.column_stack((V, self.barrier.barriers(states)))

    def predict(self, obs, controls, reference):
        agent = self.agent
        count = len(controls)
        states = np.empty((count, 1, agent.state_rollouts.shape[-1]))
        initial = np.repeat(np.r_[0., obs][None], count, axis=0)
        agent.rollout_func(states, controls[:, None, :], initial,
                           num_workers=agent.num_workers, nstep=1)
        return self.values(states[:, 0], reference)

    def repair(self, obs, nominal):
        start = perf_counter()
        a, c = self.agent, self.cfg
        n = a.act_dim
        u0 = nominal[0].copy()
        self.failure = None
        self.failure_candidate = None
        reference = a.command_velocity.copy()  # Held fixed for this one-step test.
        row = dict(status='not_solved', accepted=False, slack=None, clf_residual=None,
                   min_cbf_residual=None, correction_norm=None, elapsed_ms=None)

        def stop(reason):
            row['reason'] = reason
            row['elapsed_ms'] = 1000*(perf_counter()-start)
            self.history.append(row.copy())
            self.failure = dict(reason=reason, observation=np.asarray(obs).tolist(),
                                nominal_input=u0.tolist(), candidate_input=self.failure_candidate, reference=reference.tolist(),
                                diagnostics=row.copy())
            raise FilterFailure(reason)

        if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(u0)):
            stop('Non-finite observation or nominal input')
        try:
            current = self.values(np.r_[0., obs], reference)[0]
            # Central differences, reduced to one-sided differences at actuator bounds.
            plus = np.repeat(u0[None], n, axis=0)
            minus = plus.copy()
            idx = np.arange(n)
            plus[idx, idx] = np.minimum(u0 + c['fd_step'], a.act_max)
            minus[idx, idx] = np.maximum(u0 - c['fd_step'], a.act_min)
            width = plus[idx, idx] - minus[idx, idx]
            if np.any(width <= 0):
                stop('Invalid finite-difference interval')
            values = self.predict(obs, np.vstack((u0, plus, minus)), reference)
            jac = ((values[1:1+n] - values[1+n:]) / width[:, None]).T
            residual = values[0] - np.r_[self.rho, self.barrier.cfg['alpha'], self.barrier.cfg['alpha']] * current
            if not np.isfinite(jac).all() or not np.isfinite(residual).all():
                stop('Non-finite model linearization')
            # z=[delta_u(12), CLF slack]. Hard CBF, soft aggregate CLF.
            A = np.zeros((3 + n + 1, n + 1))
            A[:3, :n] = jac
            A[0, -1] = -1
            A[3:, :] = np.eye(n + 1)
            lower_delta = np.maximum(a.act_min-u0, -self.limit)
            upper_delta = np.minimum(a.act_max-u0, self.limit)
            lower = np.r_[-np.inf, -residual[1:], lower_delta, 0.]
            upper = np.r_[-residual[0], np.inf, np.inf, upper_delta, np.inf]
            P = sparse.diags(np.r_[1/self.limit**2, c['slack_weight']], format='csc')
            solver = osqp.OSQP()
            solver.setup(P=P, q=np.zeros(n+1), A=sparse.csc_matrix(A), l=lower, u=upper,
                         verbose=False, eps_abs=c['solver_tolerance'], eps_rel=c['solver_tolerance'],
                         max_iter=c['max_iterations'], polishing=True)
            if self.previous_solution is not None:
                solver.warm_start(x=self.previous_solution)
            result = solver.solve(raise_error=False)
            row['status'] = result.info.status
            if result.info.status_val != 1 or result.x is None or not np.isfinite(result.x).all():
                stop(f'OSQP did not solve: {result.info.status}')
            z = result.x
            tol = c['tolerance']
            Az = A @ z
            if np.any(Az < lower-tol) or np.any(Az > upper+tol):
                stop('QP solution violates linear constraints beyond tolerance')
            # Clip only tiny numerical bound errors; then recheck the actual command.
            delta = np.clip(z[:n], lower_delta, upper_delta)
            command = np.clip(u0 + delta, a.act_min, a.act_max)
            self.failure_candidate = command.tolist()
            slack = max(float(z[-1]), 0.)
            actual = self.predict(obs, command[None], reference)[0]
            actual_residual = actual - np.r_[self.rho, self.barrier.cfg['alpha'], self.barrier.cfg['alpha']] * current
            # CLF is soft: report its true relaxation, not a hard safety failure.
            # Keep `slack` as the QP value for compatibility with earlier logs.
            row.update(slack=slack, qp_slack=slack,
                       actual_required_slack=float(max(actual_residual[0], 0.)),
                       clf_slack_excess=float(max(actual_residual[0]-slack, 0.)),
                       clf_decay_satisfied=bool(actual_residual[0] <= tol),
                       clf_decreased=bool(actual[0] <= current[0]),
                       clf_current=float(current[0]), clf_next=float(actual[0]),
                       clf_residual=float(actual_residual[0]-slack),
                       min_cbf_residual=float(actual_residual[1:].min()),
                       correction_norm=float(np.linalg.norm(command-u0)))
            if not np.isfinite(actual_residual).all():
                stop('Non-finite nonlinear verification result')
            if np.any(actual_residual[1:] < -tol) or np.any(actual[1:] < -tol):
                stop('Nonlinear one-step CBF verification failed')
            self.previous_solution = np.r_[command-u0, slack]
            plan = nominal.copy()
            plan[0] = command  # Future controls are never changed.
            row.update(accepted=True, elapsed_ms=1000*(perf_counter()-start), reason='accepted')
            self.history.append(row)
            return plan
        except FilterFailure:
            raise
        except Exception as exc:
            stop(f'Filter evaluation failed: {type(exc).__name__}: {exc}')
