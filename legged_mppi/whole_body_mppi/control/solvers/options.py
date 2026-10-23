"""Command-line selection of gait-based MPPI optimization modes."""
import numpy as np


def add_mppi_mode_arguments(parser, *, include_safety=False):
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

    if include_safety:
        safety = parser.add_mutually_exclusive_group()
        safety.add_argument('--shield-mppi', dest='safety_mode', action='store_const',
                            const='shield-mppi', help='Add DCBF sample cost and local repair to the selected optimizer.')
        safety.add_argument('--clf-cbf-qp', dest='safety_mode', action='store_const',
                            const='clf-cbf-qp', help='Filter the selected optimizer current command with OSQP.')


def apply_mppi_mode(agent, mode, safety_mode=None):
    # Preserve callers using the old combined mode API.
    if mode in ('shield-mppi', 'clf-cbf-qp'):
        if safety_mode is not None:
            raise ValueError('Specify safety only once')
        mode, safety_mode = 'mppi', mode
    if safety_mode not in (None, 'shield-mppi', 'clf-cbf-qp'):
        raise ValueError(f'Unknown safety mode: {safety_mode}')
    if safety_mode is not None and mode is None:
        mode = 'mppi'  # Safety-only CLI retains the previous one-pass baseline.
    if mode not in (None, 'mppi', 'dial-mpc', 'cma-mppi', 'mpopi', 'mbd'):
        raise ValueError(f'Unknown MPPI mode: {mode}')
    if safety_mode is not None and (getattr(agent, 'task', None) != 'walk_straight_cmd_vel'
                                    or getattr(agent, 'robot', None) != 'go1'):
        raise ValueError(f'--{safety_mode} currently requires Go1 walk_straight_cmd_vel')
    agent.mppi_mode = mode
    agent.safety_mode = safety_mode
    # Restore YAML settings when changing modes on an existing controller.
    params = getattr(agent, 'params', {})
    agent.n_iterations = params.get('n_iterations', agent.n_iterations)
    agent.beta_iteration = float(params.get('beta_iteration', agent.beta_iteration))
    agent.beta_horizon = float(params.get('beta_horizon', agent.beta_horizon))
    if mode == 'mppi':
        agent.n_iterations = 1
        agent.beta_iteration = np.inf
        agent.beta_horizon = np.inf
    agent.shield = None
    agent.qp_filter = None
    if safety_mode == 'clf-cbf-qp':
        from whole_body_mppi.control.safety.clf_cbf_qp import CLFCBFQP
        agent.qp_filter = CLFCBFQP(agent, agent.params['clf_cbf_qp'], agent.params['shield_mppi'])
    elif safety_mode == 'shield-mppi':
        from whole_body_mppi.control.safety.shield import SafetyShield
        agent.shield = SafetyShield(agent, agent.params['shield_mppi'])
