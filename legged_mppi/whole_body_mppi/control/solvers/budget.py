"""Shared control-cycle rollout budget for every optimization method."""
import numpy as np


def sample_counts(total, iterations):
    if (not isinstance(total, (int, np.integer))
            or not isinstance(iterations, (int, np.integer))
            or iterations < 1 or total < 2 * iterations):
        raise ValueError('Require integer n_samples >= 2 * n_iterations (total rollout budget)')
    quotient, remainder = divmod(total, iterations)
    return [quotient + (index < remainder) for index in range(iterations)]


def rollout_batches(solver, obs):
    """Expose only the active state buffer to task cost callbacks each stage."""
    counts = sample_counts(solver.n_samples, solver.n_iterations)
    solver.iteration_sample_counts = counts
    storage = getattr(solver, '_rollout_storage', solver.state_rollouts)
    solver._rollout_storage = storage
    for count in counts:
        solver.state_rollouts = storage[:count]
        initial_state = np.repeat(np.array([np.concatenate([[0], obs])]), count, axis=0)
        yield count, initial_state
