"""Command-line selection of gait-based MPPI optimization modes."""
import numpy as np


def add_mppi_mode_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dial-mpc', dest='mppi_mode', action='store_const',
                       const='dial-mpc', help='Use configured iterations and covariance annealing.')
    group.add_argument('--cma-mppi', dest='mppi_mode', action='store_const',
                       const='cma-mppi', help='Use adaptive diagonal covariance MPPI.')
    group.add_argument('--mpopi', dest='mppi_mode', action='store_const',
                       const='mpopi', help='Use elite full-covariance proposal adaptation and MPPI.')
    group.add_argument('--mbd', dest='mppi_mode', action='store_const',
                       const='mbd', help='Use model-based diffusion and Monte Carlo score ascent.')
    group.add_argument('--mppi', dest='mppi_mode', action='store_const',
                       const='mppi', help='Use one iteration without covariance annealing.')


def apply_mppi_mode(agent, mode):
    # Without a flag, retain the task YAML settings.
    agent.mppi_mode = mode
    if mode == 'mppi':
        agent.n_iterations = 1
        agent.beta_iteration = np.inf
        agent.beta_horizon = np.inf
    elif mode not in (None, 'dial-mpc', 'cma-mppi', 'mpopi', 'mbd'):
        raise ValueError(f'Unknown MPPI mode: {mode}')
