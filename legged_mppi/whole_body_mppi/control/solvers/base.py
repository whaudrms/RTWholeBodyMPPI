"""Task-independent MPPI state, sampling, solver selection and warm start."""
import mujoco
import numpy as np

from whole_body_mppi.control.rollouts.mujoco import MujocoRollout
from whole_body_mppi.control.solvers.sampling import ActionSampling
from whole_body_mppi.control.solvers.budget import sample_counts
from whole_body_mppi.control.solvers.mppi import optimize_mppi_actions
from whole_body_mppi.control.solvers.cma_mppi import optimize_cma_actions
from whole_body_mppi.control.solvers.mpopi import optimize_mpopi_actions
from whole_body_mppi.control.solvers.mbd import diffusion_schedule, optimize_mbd_actions


class MPPISolver(ActionSampling):
    def __init__(self, model, params, *, initial_action, action_min, action_max):
        self.model = model
        # MPPI parameters
        self.temperature = params['lambda']
        self.horizon = params['horizon']
        self.n_samples = params['n_samples']
        self.n_iterations = params.get('n_iterations', 1)
        self.beta_iteration = float(params.get('beta_iteration', np.inf))
        self.beta_horizon = float(params.get('beta_horizon', np.inf))
        if (not isinstance(self.n_iterations, int) or self.n_iterations < 1
                or not self.beta_iteration > 0 or not self.beta_horizon > 0):
            raise ValueError('Invalid MPPI iteration or annealing parameters')
        self.noise_sigma = np.array(params['noise_sigma'])
        self.mppi_mode = None
        self.cma_params = dict(covariance_learning_rate=0.5, sigma_min=0.05,
                               sigma_max=3.0, elite_ratio=0.0, reward_normalization=True)
        self.cma_params.update(params.get('cma_mppi', {}))
        c = self.cma_params
        if (not np.isfinite([c['covariance_learning_rate'], c['sigma_min'],
                             c['sigma_max'], c['elite_ratio']]).all()
                or not 0 < c['covariance_learning_rate'] <= 1
                or not 0 < c['sigma_min'] <= c['sigma_max']
                or not 0 <= c['elite_ratio'] <= 1):
            raise ValueError('Invalid CMA-MPPI covariance or elite parameters')
        self.mpopi_params = dict(elite_fraction=0.2, learning_rate=0.5, covariance_floor=1e-3)
        self.mpopi_params.update(params.get('mpopi', {}))
        p = self.mpopi_params
        if (not np.isfinite([p['elite_fraction'], p['learning_rate'], p['covariance_floor']]).all()
                or not 0 < p['elite_fraction'] <= 1
                or not 0 <= p['learning_rate'] <= 1
                or not 0 < p['covariance_floor'] <= 1):
            raise ValueError('Invalid MPOPI elite or covariance parameters')
        self.mbd_params = dict(alpha_bar_first=0.98, alpha_bar_last=0.2)
        self.mbd_params.update(params.get('mbd', {}))
        diffusion_schedule(self.n_iterations, self.mbd_params['alpha_bar_first'],
                           self.mbd_params['alpha_bar_last'])
        self.cma_covariance = None
        self.num_workers = params['n_workers']

        # Initialize rollouts and sampling configurations
        self.h = params['dt']
        self.sample_type = params['sample_type']
        self.n_knots = params['n_knots']
        self.random_generator = np.random.default_rng(params["seed"])

        self.sampling_init = np.asarray(initial_action, dtype=float).copy()
        self.act_min = np.asarray(action_min, dtype=float).copy()
        self.act_max = np.asarray(action_max, dtype=float).copy()
        self.act_dim = len(self.sampling_init)
        self.rollout_backend = MujocoRollout(model, self.num_workers)
        self.rollout_func = self.rollout_backend.threaded_rollout
        self.state_rollouts = np.zeros(
            (self.n_samples, self.horizon,
             mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value)))
        self._rollout_storage = self.state_rollouts
        self.selected_trajectory = None
        self.reset_planner()

    def reset_planner(self):
        """Reset the action planner to its initial state."""
        self.trajectory = np.zeros((self.horizon, self.act_dim))
        self.trajectory += self.sampling_init
        self.iteration_sample_counts = []
        self.cma_covariance = None
        self.cma_covariance_history = []
        self.mpopi_covariance_history = []
        self.mpopi_proposal_history = []
        self.mpopi_final_actions = None
        self.mpopi_final_costs = None
        self.mbd_alpha_bars = None
        self.mbd_alphas = None
        self.mbd_forward_trace = None
        self.mbd_backward_trace = None
        self.mbd_score_trace = None
        self.mbd_final_actions = None
        self.mbd_final_costs = None

    def optimize_actions(self, obs, evaluate_cost):
        """Evaluate task-provided costs using the selected optimization method."""
        if self.mppi_mode == 'mbd':
            return optimize_mbd_actions(self, obs, evaluate_cost)
        if self.mppi_mode == 'mpopi':
            return optimize_mpopi_actions(self, obs, evaluate_cost)
        if self.mppi_mode == 'cma-mppi':
            return optimize_cma_actions(self, obs, evaluate_cost)
        return optimize_mppi_actions(self, obs, evaluate_cost)

    def commit_plan(self, actions):
        """Store the selected plan and advance the warm start by one step."""
        self.selected_trajectory = actions
        self.trajectory = np.roll(actions, shift=-1, axis=0)
        self.trajectory[-1] = actions[-1]
        return actions[0]

    def shutdown(self):
        backend = getattr(self, 'rollout_backend', None)
        if backend is not None:
            backend.shutdown()

    def set_params(self, horizon, lambda_, N):
        """
        Update MPPI parameters and reset controller.

        Args:
            horizon (int): Time horizon.
            lambda_ (float): Temperature parameter for MPPI.
            N (int): Number of samples for MPPI rollouts.
        """
        sample_counts(N, 1 if self.mppi_mode == 'mppi' else self.n_iterations)
        self.horizon = horizon
        self.temperature = lambda_
        self.n_samples = N

        # Reset state rollouts with updated dimensions
        self.state_rollouts = np.zeros(
            (self.n_samples, self.horizon, mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value))
        )

        self._rollout_storage = self.state_rollouts

        # Reset the planner to its initial state
        self.reset_planner()

    def __del__(self):
        self.shutdown()
