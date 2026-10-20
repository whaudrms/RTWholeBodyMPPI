"""Nominal and annealed iterative MPPI updates."""
import numpy as np
from whole_body_mppi.control.solvers.budget import rollout_batches


def optimize_mppi_actions(agent, obs, evaluate_cost):
    """Iterative MPPI with fixed task costs and optional noise annealing."""
    agent.iteration_best_costs = []
    for iteration, (count, initial_state) in enumerate(rollout_batches(agent, obs)):
        actions = agent.perturb_action(iteration, n_samples=count)
        agent.rollout_func(agent.state_rollouts, actions, initial_state,
                          num_workers=agent.num_workers, nstep=agent.horizon)
        costs_sum = evaluate_cost(actions)
        min_cost = np.min(costs_sum)
        max_cost = np.max(costs_sum)
        agent.iteration_best_costs.append(float(min_cost))
        span = max(max_cost - min_cost, 1e-12)
        agent.exp_weights = np.exp(
            -1 / agent.temperature * ((costs_sum - min_cost) / span))
        weighted_actions = np.sum(
            agent.exp_weights.reshape(count, 1, 1) * actions, axis=0)
        updated_actions = np.clip(
            weighted_actions / (np.sum(agent.exp_weights) + 1e-10),
            agent.act_min, agent.act_max)
        agent.trajectory = updated_actions
    return updated_actions

