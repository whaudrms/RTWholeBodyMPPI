"""Elite proposal adaptation and final-cycle MPPI weighting.

Port of the local study MPOPI optimizer to existing task costs and cubic knots.
Covariance couples action axes at each knot/step, not different time points.
n_samples is the shared total rollout budget across all iterations.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from whole_body_mppi.control.solvers.budget import rollout_batches


def normalized_rank_weights(count):
    ranks = np.arange(1, count + 1, dtype=float)
    weights = np.log((count + 1) / ranks)
    return weights / weights.sum()


def exponential_weights(costs, temperature):
    finite = np.isfinite(costs)
    if not finite.any():
        raise FloatingPointError('All MPOPI rollout costs are non-finite')
    weights = np.zeros(len(costs))
    weights[finite] = np.exp(-(costs[finite] - costs[finite].min()) / temperature)
    return weights / weights.sum()


def adapt_proposal(mean, covariance, elites, weights, rate, floor):
    """Update normalized covariance around the OLD proposal mean."""
    displacement = elites - mean
    moment = np.einsum('n,nku,nkv->kuv', weights, displacement, displacement)
    updated = (1 - rate) * covariance + rate * moment
    updated = 0.5 * (updated + updated.swapaxes(-1, -2))
    values, vectors = np.linalg.eigh(updated)
    values = np.maximum(values, floor)
    updated = np.einsum('kui,ki,kvi->kuv', vectors, values, vectors)
    mean = (1 - rate) * mean + rate * np.einsum('n,nku->ku', weights, elites)
    return mean, updated


def optimize_mpopi_actions(solver, obs, evaluate_cost):
    if solver.sample_type == 'cubic':
        indices = np.round(np.linspace(0, solver.horizon - 1, solver.n_knots)).astype(int)
        if len(indices) < 2 or np.any(np.diff(indices) <= 0):
            raise ValueError('MPOPI cubic sampling requires distinct knots within the horizon')
    elif solver.sample_type == 'normal':
        indices = np.arange(solver.horizon)
    else:
        raise ValueError(f'Unsupported MPOPI sample type: {solver.sample_type}')
    sigma = np.asarray(solver.noise_sigma)
    if sigma.shape != (solver.act_dim,) or not np.all(np.isfinite(sigma) & (sigma > 0)):
        raise ValueError('MPOPI requires positive finite noise_sigma per action axis')
    if solver.n_samples < 2:
        raise ValueError('MPOPI requires at least two samples per iteration')
    p = solver.mpopi_params
    proposal = solver.trajectory[indices].copy()
    # Reinitialize at every MPC cycle, unlike persistent CMA covariance.
    covariance = np.broadcast_to(np.eye(solver.act_dim),
                                  (len(indices), solver.act_dim, solver.act_dim)).copy()
    solver.iteration_best_costs = []
    solver.mpopi_covariance_history = []
    solver.mpopi_proposal_history = []
    for iteration, (count, initial_state) in enumerate(rollout_batches(solver, obs)):
        solver.mpopi_covariance_history.append(covariance.copy())
        solver.mpopi_proposal_history.append(proposal.copy())
        gaussian = solver.random_generator.normal(
            size=(count, len(indices), solver.act_dim))
        noise = np.einsum('kuv,nkv->nku', np.linalg.cholesky(covariance), gaussian) * sigma
        noise[0] = 0.0  # Include the current proposal as the nominal candidate.
        knots = np.clip(proposal + noise, solver.act_min, solver.act_max)
        if solver.sample_type == 'cubic':
            actions = CubicSpline(indices, knots, axis=1)(np.arange(solver.horizon))
            actions = np.clip(actions, solver.act_min, solver.act_max)
        else:
            actions = knots
        solver.rollout_func(solver.state_rollouts, actions, initial_state,
                            num_workers=solver.num_workers, nstep=solver.horizon)
        costs = np.asarray(evaluate_cost(actions))
        weights = exponential_weights(costs, solver.temperature)
        finite = np.flatnonzero(np.isfinite(costs))
        solver.iteration_best_costs.append(float(costs[finite].min()))
        if iteration < solver.n_iterations - 1:
            elite_count = max(1, int(np.ceil(p['elite_fraction'] * count)))
            elites = finite[np.argsort(costs[finite], kind='stable')[:elite_count]]
            proposal, covariance = adapt_proposal(
                proposal / sigma, covariance, actions[elites][:, indices] / sigma,
                normalized_rank_weights(len(elites)), p['learning_rate'], p['covariance_floor'])
            proposal = np.clip(proposal * sigma, solver.act_min, solver.act_max)
    # Earlier cycles only adapt the proposal; output uses the final cycle alone.
    solver.exp_weights = weights
    solver.mpopi_final_actions = actions.copy()
    solver.mpopi_final_costs = costs.copy()
    solver.trajectory = np.clip(np.einsum('n,nhj->hj', weights, actions),
                                solver.act_min, solver.act_max)
    return solver.trajectory
