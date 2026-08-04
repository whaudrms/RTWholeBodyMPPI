import numpy as np
import concurrent.futures
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
import mujoco
from scipy.interpolate import CubicSpline
from mujoco import rollout
import yaml


_ROLLOUT_HAS_NROLL = "nroll" in inspect.signature(rollout.rollout).parameters

class BaseMPPI:
    """
    Base class for Model Predictive Path Integral (MPPI) controllers.
    Provides shared functionality for all task-specific controllers.
    """

    def __init__(self, model_path, config_path, backend="cpu"):
        """
        Initialize common MPPI parameters and configurations.

        Args:
            params (dict): Dictionary of task parameters from configuration.
            model_path (str): Path to the MuJoCo model XML file.
            config_path (str): Path to the configuration file.
        """

        if backend not in ("cpu", "warp"):
            raise ValueError("backend must be either 'cpu' or 'warp'")
        self.backend = backend
        self._warp_backend = None

        # Load task-specific configurations
        with open(config_path, 'r') as file:
            params = yaml.safe_load(file)

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        supported_state_sizes = ((19, 18), (26, 24))
        if (
            self.model.nu != 12
            or (self.model.nq, self.model.nv) not in supported_state_sizes
        ):
            raise ValueError(
                "The controller requires a 12-actuator floating-base "
                "quadruped, optionally preceded by one free prop body."
            )
        self.model.opt.timestep = params['dt']
        if self.backend == "warp":
            # MJX-Warp does not support mjENBL_OVERRIDE. Applying the same
            # solref to every geom preserves the intended global contact
            # setting without enabling the unsupported runtime override.
            unsupported_bits = int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)
            # MULTICCD cannot be combined with the non-zero robot geom margin
            # in MJX-Warp (notably in the push-box scene).
            self.model.opt.enableflags &= ~unsupported_bits
            self.model.opt.disableflags |= (
                int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
                | int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
            )
            self.model.geom_solref[:, :2] = np.asarray(params['o_solref'])
            # MuJoCo CPU accepts zero sliding friction, while Warp warns that
            # it can create NaNs for condim >= 3 contacts.
            self.model.geom_friction[:, 0] = np.maximum(
                self.model.geom_friction[:, 0], 1.0e-5
            )
        else:
            self.model.opt.enableflags = 1  # Override contact settings
            self.model.opt.o_solref = np.array(params['o_solref'])

        # MPPI parameters
        self.temperature = params['lambda']
        self.horizon = params['horizon']
        self.n_samples = params['n_samples']
        self.noise_sigma = np.array(params['noise_sigma'])
        self.num_workers = params['n_workers']
        self.sampling_init = np.array(
            [-0.3, 1.34, -2.83, 0.3, 1.34, -2.83] * 2
        )

        # Initialize rollouts and sampling configurations
        self.h = params['dt']
        self.sample_type = params['sample_type']
        self.n_knots = params['n_knots']
        self.random_generator = np.random.default_rng(params["seed"])
        self.cost_func = self.calculate_total_cost

        # Threading
        self.thread_local = None
        self.executor = None
        self._native_rollout_data = None
        if _ROLLOUT_HAS_NROLL:
            # MuJoCo 3.1 releases the GIL in rollout but does not provide its
            # own batch thread pool, so retain the existing Python workers.
            self.thread_local = threading.local()
            self.executor = ThreadPoolExecutor(
                max_workers=self.num_workers,
                initializer=self.thread_initializer,
            )
            self.rollout_func = self.threaded_rollout
        else:
            # New MuJoCo rollout owns a native thread pool.  Calling several
            # instances concurrently from Python can crash, so give one
            # batched call a reusable MjData object per native worker.
            self._native_rollout_data = [
                mujoco.MjData(self.model) for _ in range(self.num_workers)
            ]
            self.rollout_func = self.native_threaded_rollout

        # Initialize rollouts
        self.state_rollouts = np.zeros(
            (self.n_samples, self.horizon, mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value))
        )
        self.selected_trajectory = None
        self._planner_initialized = False

        # Keep controls and costs in actuator order (FR, FL, RR, RL).  MuJoCo
        # stores qpos/qvel in body-tree order, which is different in the Go2
        # model, so retain the model indices needed to canonicalize rollouts.
        self.act_dim = self.model.nu
        actuator_joint_ids = self.model.actuator_trnid[:, 0]
        self.joint_qpos_indices = self.model.jnt_qposadr[actuator_joint_ids]
        self.joint_qvel_indices = (
            self.model.nq + self.model.jnt_dofadr[actuator_joint_ids]
        )

        # Read limits from the selected robot rather than using Go1 constants.
        self.act_min = self.model.actuator_ctrlrange[:, 0].copy()
        self.act_max = self.model.actuator_ctrlrange[:, 1].copy()
        self.sampling_init = np.clip(
            self.sampling_init, self.act_min, self.act_max
        )

    def reset_planner(self):
        """Reset the action planner to its initial state."""
        self.trajectory = np.zeros((self.horizon, self.act_dim))
        self.trajectory += self.sampling_init
        self.selected_trajectory = None
        self._planner_initialized = False

    def prepare_planner_update(self, advance_steps=1):
        """Advance the warm-start trajectory to the current control tick.

        ``advance_steps`` is the number of command-loop ticks elapsed
        since the state used by the previous planner update.  The first
        planner call has no previous trajectory and therefore does not shift.
        """
        if isinstance(advance_steps, bool) or not isinstance(
            advance_steps, (int, np.integer)
        ):
            raise TypeError("advance_steps must be a non-negative integer")
        if advance_steps < 0:
            raise ValueError("advance_steps must be a non-negative integer")
        if not self._planner_initialized or advance_steps == 0:
            return 0

        steps = min(int(advance_steps), self.horizon)
        terminal_action = self.trajectory[-1].copy()
        if steps == self.horizon:
            self.trajectory[:] = terminal_action
        else:
            self.trajectory[:-steps] = self.trajectory[steps:]
            self.trajectory[-steps:] = terminal_action
        return int(advance_steps)

    def complete_planner_update(self, updated_actions):
        """Store a complete, unshifted plan for execution and warm starting."""
        self.selected_trajectory = updated_actions.copy()
        self.trajectory = updated_actions.copy()
        self._planner_initialized = True

    def state_in_actuator_order(self, state):
        """Return qpos/qvel joints arranged in actuator (controller) order."""
        ordered_state = state.copy()
        ordered_state[:, 7:19] = state[:, self.joint_qpos_indices]
        ordered_state[:, 25:37] = state[:, self.joint_qvel_indices]
        return ordered_state

    def sample_delta_u(self):
        if self.sample_type == 'normal':
            size = (self.n_samples, self.horizon, self.act_dim)
            return self.generate_noise(size)
        elif self.sample_type == 'cubic':
            indices = np.arange(self.n_knots)*self.horizon//self.n_knots
            size = (self.n_samples, self.n_knots, self.act_dim)
            knot_points = self.generate_noise(size)
            cubic_spline = CubicSpline(indices, knot_points, axis=1)
            return cubic_spline(np.arange(self.horizon))
        
    def perturb_action(self):
        if self.sample_type == 'normal':
            size = (self.n_samples, self.horizon, self.act_dim)
            actions = self.trajectory + self.generate_noise(size)
            actions = np.clip(actions, self.act_min, self.act_max)
            return actions
        
        elif self.sample_type == 'cubic':
            indices_float = np.linspace(0, self.horizon - 1, num=self.n_knots)
            indices = np.round(indices_float).astype(int)
            size = (self.n_samples, self.n_knots, self.act_dim)
            noise = self.generate_noise(size)
            knot_points = self.trajectory[indices] + noise
            #knot_points[:, 0, :] = self.trajectory[0]
            cubic_spline = CubicSpline(indices, knot_points, axis=1)
            actions = cubic_spline(np.arange(self.horizon))
            actions = np.clip(actions, self.act_min, self.act_max)
            return actions
        
    def generate_noise(self, size):
        """
        Generate noise for sampling actions.

        Args:
            size (tuple): Shape of the noise array.

        Returns:
            np.ndarray: Generated noise scaled by `noise_sigma`.
        """
        return self.random_generator.normal(size=size) * self.noise_sigma

    def thread_initializer(self):
        """Initialize thread-local storage for MuJoCo data."""
        self.thread_local.data = mujoco.MjData(self.model)

    def shutdown(self):
        """Shutdown the thread pool executor."""
        if getattr(self, "executor", None) is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
        if (
            getattr(self, "_native_rollout_data", None) is not None
            and hasattr(rollout, "shutdown_persistent_pool")
        ):
            rollout.shutdown_persistent_pool()
            self._native_rollout_data = None

    def get_warp_backend(self, cost_mode):
        """Lazily construct the simulation-only GPU backend."""
        if self.backend != "warp":
            raise RuntimeError("Warp backend requested from a CPU controller")
        if self._warp_backend is None:
            from whole_body_mppi.control.controllers.raw_warp_mppi import (
                RawWarpMPPI,
            )

            self._warp_backend = RawWarpMPPI(self, cost_mode)
        elif self._warp_backend.cost_mode != cost_mode:
            raise RuntimeError("A controller cannot mix Warp cost modes")
        return self._warp_backend

    def call_rollout(self, initial_state, ctrl, state):
        """
        Perform a rollout of the model given the initial state and control actions.

        Args:
            initial_state (np.ndarray): Initial state of the model.
            ctrl (np.ndarray): Control actions to apply during the rollout.
            state (np.ndarray): State array to store the results of the rollout.
        """
        # MuJoCo <= 3.1 forwards ``nroll`` directly to its C++ binding when
        # checks are skipped, so the batch size must be explicit.
        rollout.rollout(
            self.model,
            self.thread_local.data,
            skip_checks=True,
            nroll=state.shape[0],
            nstep=state.shape[1],
            initial_state=initial_state,
            control=ctrl,
            state=state,
        )

    def native_threaded_rollout(
        self, state, ctrl, initial_state, num_workers=32, nstep=5
    ):
        """Run a full batch using MuJoCo's native persistent thread pool."""
        del num_workers, nstep
        rollout.rollout(
            self.model,
            self._native_rollout_data,
            initial_state=initial_state,
            control=ctrl,
            state=state,
            nstep=state.shape[1],
            persistent_pool=True,
        )

    def threaded_rollout(self, state, ctrl, initial_state, num_workers=32, nstep=5):
        """
        Perform rollouts in parallel using a thread pool.

        Args:
            state (np.ndarray): Array to store the results of the rollouts.
            ctrl (np.ndarray): Control actions for the rollouts.
            initial_state (np.ndarray): Initial states for the rollouts.
            num_workers (int): Number of parallel threads to use.
            nstep (int): Number of steps in each rollout.
        """
        n = len(initial_state) // num_workers

        # Divide tasks into chunks for each worker
        chunks = [(initial_state[i * n:(i + 1) * n], ctrl[i * n:(i + 1) * n], state[i * n:(i + 1) * n])
                for i in range(num_workers - 1)]

        # Add remaining chunk
        chunks.append((initial_state[(num_workers - 1) * n:], ctrl[(num_workers - 1) * n:], state[(num_workers - 1) * n:]))

        # Submit tasks to thread pool
        futures = [self.executor.submit(self.call_rollout, *chunk) for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Ensure all threads complete execution

    def set_params(self, horizon, lambda_, N):
        """
        Update MPPI parameters and reset controller.

        Args:
            horizon (int): Time horizon.
            lambda_ (float): Temperature parameter for MPPI.
            N (int): Number of samples for MPPI rollouts.
        """
        if horizon <= 0 or N <= 0:
            raise ValueError("horizon and sample count must be positive")
        if self.sample_type == "cubic" and horizon < self.n_knots:
            raise ValueError("cubic sampling requires horizon >= n_knots")
        self.horizon = horizon
        self.temperature = lambda_
        self.n_samples = N
        self._warp_backend = None

        # Reset state rollouts with updated dimensions
        self.state_rollouts = np.zeros(
            (self.n_samples, self.horizon, mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value))
        )

        # Reset the planner to its initial state
        self.reset_planner()

    def __del__(self):
        self.shutdown()
