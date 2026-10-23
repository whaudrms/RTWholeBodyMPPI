"""Read-only observation of MPPI proposals; no changes to solver mathematics."""
import numpy as np


def cross_correlation(actions, weights):
    """Population weighted correlation over samples, separately at each horizon step.

    Returns (H, 12, 6) correlation and covariance. Constant axes are NaN.
    """
    x = np.asarray(actions, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.ndim != 3 or x.shape[2] != 18 or w.shape != (len(x),):
        raise ValueError('Expected actions (N,H,18) and weights (N,)')
    if not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError('Actions and weights must be finite; weights nonnegative with positive sum')
    w = w / w.sum()
    centered = x - np.einsum('n,nhj->hj', w, x)[None]
    variance = np.einsum('n,nhj->hj', w, centered**2)
    covariance = np.einsum('n,nhi,nhj->hij', w, centered[:, :, :12], centered[:, :, 12:])
    denom = np.sqrt(variance[:, :12, None] * variance[:, None, 12:])
    valid = (variance[:, :12, None] > 1e-20) & (variance[:, None, 12:] > 1e-20)
    correlation = np.full_like(covariance, np.nan)
    np.divide(covariance, denom, out=correlation, where=valid)
    return np.clip(correlation, -1, 1), covariance


class LegArmObserver:
    """Instance-local wrappers, restored on exit; consume no controller RNG.

    Flush a proposal only after the original solver has assigned exp_weights.
    This supports standard/configured/annealed MPPI, not other solver families.
    """
    def __init__(self, agent, ess_min=20., ess_fraction=.2):
        if agent.act_dim != 18 or agent.mppi_mode not in (None, 'mppi', 'dial-mpc'):
            raise ValueError('Requires 18-action whole-body MPPI or DIAL-MPC')
        if not np.isfinite([ess_min, ess_fraction]).all() or ess_min < 1 or not 0 <= ess_fraction <= 1:
            raise ValueError('Invalid ESS thresholds')
        self.agent, self.ess_min, self.ess_fraction = agent, ess_min, ess_fraction
        self.records, self.pending, self.update = [], None, 0

    def __enter__(self):
        a = self.agent
        self.saved = {name: a.__dict__.get(name) for name in ('perturb_action', 'optimize_actions')}
        sample, optimize = a.perturb_action, a.optimize_actions

        def observed_sample(iteration=0, n_samples=None):
            self.flush()
            actions = sample(iteration, n_samples=n_samples)
            self.pending = (iteration, actions.copy())
            return actions

        def observed_optimize(obs, evaluate_cost):
            result = optimize(obs, evaluate_cost)
            self.flush()
            self.update += 1
            return result

        a.perturb_action, a.optimize_actions = observed_sample, observed_optimize
        return self

    def flush(self):
        if self.pending is None:
            return
        iteration, actions = self.pending
        w = np.asarray(self.agent.exp_weights).copy()
        weighted, covariance = cross_correlation(actions, w)
        uniform, _ = cross_correlation(actions, np.ones(len(actions)))
        w /= w.sum()
        ess = float(1 / np.sum(w*w))
        self.records.append(dict(update=self.update, iteration=iteration,
                                 weighted=weighted, uniform=uniform, covariance=covariance,
                                 ess=ess, count=len(actions),
                                 low_ess=ess < max(self.ess_min, self.ess_fraction * len(actions))))
        self.pending = None

    def __exit__(self, *exc):
        for name, value in self.saved.items():
            if value is None:
                self.agent.__dict__.pop(name, None)
            else:
                setattr(self.agent, name, value)

    def save(self, path):
        if not self.records:
            raise ValueError('No MPPI updates observed')
        np.savez_compressed(path, **{key: np.asarray([r[key] for r in self.records])
                                    for key in self.records[0]})
