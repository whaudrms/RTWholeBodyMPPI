"""Model-based diffusion in bounded action/knot space using existing task costs.

Finite forward diffusion of the warm start is followed by deterministic
Monte Carlo score ascent. n_samples is the total rollout budget.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from whole_body_mppi.control.solvers.budget import rollout_batches


def diffusion_schedule(iterations, first, last):
    if iterations < 1 or not np.isfinite([first, last]).all() or not 0 < last <= first < 1:
        raise ValueError('MBD requires iterations >= 1 and 0 < alpha_bar_last <= alpha_bar_first < 1')
    levels = np.geomspace(first, last, iterations) if iterations > 1 else np.array([last])
    bars = np.r_[1., levels]
    return bars, bars[1:] / bars[:-1]


def forward_process(clean, alphas, noise):
    """Return normalized Y0..YN without evaluating dynamics."""
    if np.shape(noise) != (len(alphas), *clean.shape):
        raise ValueError('MBD forward noise must have shape (iterations, *clean.shape)')
    trace = [clean.copy()]
    for alpha, epsilon in zip(alphas, noise):
        trace.append(np.sqrt(alpha) * trace[-1] + np.sqrt(1-alpha) * epsilon)
    return np.asarray(trace)


def estimate_score(latent, weighted_clean, alpha_bar):
    return (np.sqrt(alpha_bar) * weighted_clean - latent) / (1-alpha_bar)


def reverse_step(latent, score, alpha, alpha_bar):
    return (latent + (1-alpha_bar) * score) / np.sqrt(alpha)


def optimize_mbd_actions(solver, obs, evaluate_cost):
    if solver.sample_type == 'cubic':
        indices = np.round(np.linspace(0, solver.horizon-1, solver.n_knots)).astype(int)
        if len(indices) < 2 or np.any(np.diff(indices) <= 0):
            raise ValueError('MBD cubic sampling requires distinct knots within the horizon')
    elif solver.sample_type == 'normal':
        indices = np.arange(solver.horizon)
    else:
        raise ValueError(f'Unsupported MBD sample type: {solver.sample_type}')
    sigma = np.asarray(solver.noise_sigma)
    if sigma.shape != (solver.act_dim,) or not np.all(np.isfinite(sigma) & (sigma > 0)):
        raise ValueError('MBD requires positive finite noise_sigma per action axis')
    if solver.n_samples < 2:
        raise ValueError('MBD requires at least two samples per reverse stage')
    p = solver.mbd_params
    bars, alphas = diffusion_schedule(solver.n_iterations, p['alpha_bar_first'], p['alpha_bar_last'])
    solver.mbd_alpha_bars, solver.mbd_alphas = bars, alphas
    center = (solver.act_min + solver.act_max) / 2
    clean = (solver.trajectory[indices] - center) / sigma
    noise = solver.random_generator.normal(size=(solver.n_iterations, *clean.shape))
    solver.mbd_forward_trace = forward_process(clean, alphas, noise)
    latent = solver.mbd_forward_trace[-1].copy()
    backward, scores = [latent.copy()], []
    solver.iteration_best_costs = []
    for iteration, (count, initial_state) in enumerate(rollout_batches(solver, obs)):
        stage = solver.n_iterations - iteration
        abar, alpha = bars[stage], alphas[stage-1]
        proposal = latent / np.sqrt(abar)
        raw = proposal + np.sqrt(1/abar-1) * solver.random_generator.normal(
            size=(count, len(indices), solver.act_dim))
        raw[0] = proposal
        knots = np.clip(center + sigma * raw, solver.act_min, solver.act_max)
        if solver.sample_type == 'cubic':
            actions = np.clip(CubicSpline(indices, knots, axis=1)(np.arange(solver.horizon)),
                              solver.act_min, solver.act_max)
        else:
            actions = knots
        solver.rollout_func(solver.state_rollouts, actions, initial_state,
                            num_workers=solver.num_workers, nstep=solver.horizon)
        costs = np.asarray(evaluate_cost(actions))
        finite = np.isfinite(costs)
        if not finite.any():
            raise FloatingPointError('All MBD rollout costs are non-finite')
        weights = np.zeros(count)
        weights[finite] = np.exp(-(costs[finite] - costs[finite].min()) / solver.temperature)
        weights /= weights.sum()
        solver.iteration_best_costs.append(float(costs[finite].min()))
        candidates = (actions[:, indices] - center) / sigma
        weighted_clean = np.einsum('n,nkj->kj', weights, candidates)
        score = estimate_score(latent, weighted_clean, abar)
        latent = reverse_step(latent, score, alpha, abar)
        scores.append(score.copy())
        backward.append(latent.copy())
    solver.mbd_backward_trace = np.asarray(backward)
    solver.mbd_score_trace = np.asarray(scores)
    solver.exp_weights = weights
    solver.mbd_final_actions, solver.mbd_final_costs = actions.copy(), costs.copy()
    final_knots = np.clip(center + sigma * latent, solver.act_min, solver.act_max)
    if solver.sample_type == 'cubic':
        plan = CubicSpline(indices, final_knots, axis=0)(np.arange(solver.horizon))
    else:
        plan = final_knots
    solver.trajectory = np.clip(plan, solver.act_min, solver.act_max)
    return solver.trajectory
