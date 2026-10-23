"""Experimental Go1 DCBF cost and bounded short-horizon local repair.

Inspired by arXiv:2302.11719, equations 17 and 22. L-BFGS-B enforces actuator
bounds; unlike the paper's vehicle example, barriers concern height and tilt.
Finite iterations and model error preclude an unconditional safety guarantee.
"""
from time import perf_counter
import numpy as np
from scipy.optimize import minimize


class SafetyShield:
    def __init__(self, agent, config):
        self.agent = agent
        self.cfg = dict(config)
        c = self.cfg
        required = ('alpha', 'cost_weight', 'min_height', 'max_tilt_deg',
                    'repair_steps', 'max_iterations', 'max_evaluations', 'tolerance', 'fd_step')
        if any(k not in c for k in required):
            raise ValueError('Incomplete shield_mppi configuration')
        if not all(np.isfinite(c[k]) for k in required):
            raise ValueError('Shield parameters must be finite')
        if not (0 < c['alpha'] < 1 and c['cost_weight'] > 0 and c['min_height'] > 0
                and 0 < c['max_tilt_deg'] < 90 and c['tolerance'] >= 0 and c['fd_step'] > 0):
            raise ValueError('Invalid shield barrier parameters')
        for k in ('repair_steps', 'max_iterations', 'max_evaluations'):
            if not isinstance(c[k], int) or c[k] < 1:
                raise ValueError(f'{k} must be a positive integer')
        if c['repair_steps'] > agent.horizon:
            raise ValueError('Shield repair_steps exceeds planner horizon')
        self.history = []

    def barriers(self, states):
        """FULLPHYSICS layout: time, xyz, quaternion, ...; dimensionless margins."""
        q = states[..., 4:8]
        norm = np.maximum(np.sum(q*q, axis=-1), 1e-15)
        up = 1 - 2*(q[..., 1]**2 + q[..., 2]**2)/norm
        cosine = np.cos(np.deg2rad(self.cfg['max_tilt_deg']))
        return np.stack(((states[..., 3]-self.cfg['min_height'])/self.cfg['min_height'],
                         (up-cosine)/(1-cosine)), axis=-1)

    def residuals(self, obs, states):
        h = self.barriers(states)
        initial = self.barriers(np.r_[0., obs])
        previous = np.concatenate((np.broadcast_to(initial, (len(states), 1, 2)), h[:, :-1]), axis=1)
        return h - self.cfg['alpha'] * previous

    def cost(self, obs, states):
        return self.cfg['cost_weight'] * np.maximum(-self.residuals(obs, states), 0).sum(axis=(1, 2))

    def repair(self, obs, nominal):
        started = perf_counter()
        a, c = self.agent, self.cfg
        steps = c['repair_steps']
        plan = np.clip(nominal.copy(), a.act_min, a.act_max)
        shape = (steps, a.act_dim)
        states = np.empty((1, steps, a.state_rollouts.shape[-1]))
        initial = np.r_[0., obs][None]
        best = plan[:steps].reshape(-1).copy()
        best_loss = np.inf
        best_margin = -np.inf
        evaluations = 0

        class BudgetExceeded(Exception):
            pass

        def objective(flat):
            nonlocal best, best_loss, best_margin, evaluations
            if evaluations >= c['max_evaluations']:
                raise BudgetExceeded
            evaluations += 1
            a.rollout_func(states, flat.reshape((1,) + shape), initial,
                           num_workers=1, nstep=steps)
            residual = self.residuals(obs, states)
            loss = float(np.maximum(-residual, 0).sum())
            if np.isfinite(loss) and loss < best_loss:
                best, best_loss, best_margin = flat.copy(), loss, float(residual.min())
            return loss if np.isfinite(loss) else 1e30

        objective(best)
        before = best_loss
        optimizer_success = False
        if best_margin < -c['tolerance']:
            try:
                result = minimize(objective, best.copy(), method='L-BFGS-B',
                                  bounds=list(zip(np.tile(a.act_min, steps), np.tile(a.act_max, steps))),
                                  options={'maxiter': c['max_iterations'], 'maxfun': c['max_evaluations'],
                                           'eps': c['fd_step'], 'ftol': 1e-12, 'gtol': 1e-8})
                optimizer_success = bool(result.success)
            except BudgetExceeded:
                pass
        plan[:steps] = best.reshape(shape)
        self.history.append(dict(before=before, after=best_loss, min_residual=best_margin,
                                 feasible=bool(best_margin >= -c['tolerance']),
                                 evaluations=evaluations, optimizer_success=optimizer_success,
                                 elapsed_ms=1000*(perf_counter()-started)))
        # Best evaluated bounded plan is returned even if infeasible; never label
        # this as a certified backup controller. Diagnostics expose that failure.
        return plan
