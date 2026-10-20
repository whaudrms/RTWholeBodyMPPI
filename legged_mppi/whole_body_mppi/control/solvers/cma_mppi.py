"""Diagonal covariance adaptation using the existing rollout and task cost.

Cubic sampling adapts covariance at knots; normal sampling adapts each step.
n_samples is the total rollout budget across all iterations.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from whole_body_mppi.control.solvers.budget import rollout_batches


def optimize_cma_actions(agent, obs, evaluate_cost):
    if agent.sample_type == 'cubic':
        indices = np.round(np.linspace(0, agent.horizon - 1, agent.n_knots)).astype(int)
        if len(indices) < 2 or np.any(np.diff(indices) <= 0):
            raise ValueError('CMA cubic sampling requires distinct knots within the horizon')
    elif agent.sample_type == 'normal':
        indices = np.arange(agent.horizon)
    else:
        raise ValueError(f'Unsupported CMA sample type: {agent.sample_type}')
    p = agent.cma_params
    rate, lower, upper = p['covariance_learning_rate'], p['sigma_min'], p['sigma_max']
    base = np.broadcast_to(np.clip(agent.noise_sigma, lower, upper)**2,
                           (len(indices), agent.act_dim)).copy()
    if agent.cma_covariance is None or agent.cma_covariance.shape != base.shape:
        agent.cma_covariance = base.copy()
    covariance = agent.cma_covariance.copy()
    agent.iteration_best_costs = []
    agent.cma_covariance_history = []
    for count, initial_state in rollout_batches(agent, obs):
        knots = agent.trajectory[indices] + agent.random_generator.normal(
            size=(count, len(indices), agent.act_dim)) * np.sqrt(covariance)
        knots = np.clip(knots, agent.act_min, agent.act_max)
        if agent.sample_type == 'cubic':
            actions = CubicSpline(indices, knots, axis=1)(np.arange(agent.horizon))
            actions = np.clip(actions, agent.act_min, agent.act_max)
        else:
            actions = knots
        agent.rollout_func(agent.state_rollouts, actions, initial_state,
                           num_workers=agent.num_workers, nstep=agent.horizon)
        costs = evaluate_cost(actions)
        finite = np.isfinite(costs)
        if not finite.any():
            raise FloatingPointError('All CMA-MPPI rollout costs are non-finite')
        agent.iteration_best_costs.append(float(costs[finite].min()))
        scale = max(float(np.std(costs[finite])), 1e-10) if p['reward_normalization'] else 1.0
        weights = np.zeros(count)
        if p['elite_ratio']:
            ranked = np.flatnonzero(finite)
            ranked = ranked[np.argsort(costs[ranked], kind='stable')]
            keep = max(1, int(count * p['elite_ratio']))
            finite[:] = False
            finite[ranked[:keep]] = True
        weights[finite] = np.exp(-(costs[finite] - costs[finite].min()) / (agent.temperature * scale))
        weights /= weights.sum()
        agent.exp_weights = weights
        agent.trajectory = np.clip(np.einsum('n,nhj->hj', weights, actions),
                                   agent.act_min, agent.act_max)
        mean = agent.trajectory[indices]
        variance = np.einsum('n,nkj->kj', weights, (actions[:, indices] - mean)**2)
        covariance = np.clip((1-rate)*covariance + rate*variance, lower**2, upper**2)
        agent.cma_covariance_history.append(covariance.copy())
    # Shift variance by ONE control step, not by one knot. Interpolate variance
    # onto the shifted knot times; restore the new tail to base variance.
    shifted = np.empty_like(covariance)
    for joint in range(agent.act_dim):
        shifted[:, joint] = np.interp(indices + 1, indices, covariance[:, joint])
    shifted[-1] = base[-1]
    agent.cma_covariance = shifted
    return agent.trajectory
