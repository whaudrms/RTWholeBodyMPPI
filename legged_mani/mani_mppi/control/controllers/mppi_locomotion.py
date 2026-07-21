import os
import yaml
import mujoco
import numpy as np

# Local imports (ensure these are part of your package structure)
from mani_mppi.utils.tasks import get_task
from mani_mppi.control.controllers.base_controller import BaseMPPI
from mani_mppi.control.gait_scheduler.scheduler import GaitScheduler
from mani_mppi.control.gait_scheduler.scheduler import Timer
from mani_mppi.utils.transforms import batch_world_to_local_velocity, calculate_orientation_quaternion

# Define base directory and paths for resource files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GAIT_DIR = os.path.join(BASE_DIR, "../gait_scheduler/gaits/")

# Paths for gait files
### must generate gait data ###
GAIT_INPLACE_PATH = os.path.join(
    GAIT_DIR, "FAST/b2_retargeted/walking_gait_raibert_FAST_0_0_15cm_100hz.tsv"
)
GAIT_TROT_PATH = os.path.join(
    GAIT_DIR, "FAST/b2_retargeted/walking_gait_raibert_FAST_0_5_15cm_100hz.tsv"
)
GAIT_WALK_PATH = os.path.join(
    GAIT_DIR, "FAST/b2_retargeted/walking_gait_raibert_FAST_0_1_15cm_100hz.tsv"
)
GAIT_WALK_FAST_PATH = os.path.join(
    GAIT_DIR, "FAST/b2_retargeted/walking_gait_raibert_FAST_0_1_15cm_100hz.tsv"
)

class MPPI(BaseMPPI):
    """
    Model Predictive Path Integral (MPPI) Controller for quadruped robots.

    Attributes:
        - Task-specific parameters and goals.
        - Gait scheduler and configurations.
        - MPPI sampling and cost calculation configurations.
    """

    def __init__(self, task='stand', rollout_mode=None) -> None:
        """
        Initialize the MPPI controller with task-specific configurations.

        Args:
            task (str): The name of the task ('stand', 'walk').
            rollout_mode (str | None): ``gait`` keeps the phase-aligned gait
                nominal. ``original`` reproduces the original Go1 MPPI
                previous-solution and absolute-cubic sampling behavior. None
                uses the task configuration unchanged.
        """
        print("Task: ", task)

        # Retrieve task-specific parameters
        self.task = task
        self.task_data = get_task(task)

        self.goal_pos = self.task_data['goal_pos']
        self.goal_ori = self.task_data['default_orientation'] 
        self.cmd_vel = self.task_data['cmd_vel']
        self.goal_thresh = self.task_data['goal_thresh']
        self.desired_gait = self.task_data['desired_gait']
        model_path = self.task_data['model_path'] 
        config_path = self.task_data['config_path']
        waiting_times = self.task_data['waiting_times']

        # Dynamically resolve paths for model and configuration files
        CONFIG_PATH = os.path.join(BASE_DIR, config_path)
        MODEL_PATH = os.path.join(BASE_DIR, "../..", model_path)

        # Initialize base MPPI
        super().__init__(MODEL_PATH, CONFIG_PATH)

        # load the configuration file
        with open(CONFIG_PATH, 'r') as file:
            params = yaml.safe_load(file)
        # Cost weights
        self.Q = np.diag(np.array(params['Q_diag']))
        self.R = np.diag(np.array(params['R_diag']))
        self.cost_func = self.calculate_total_cost
        self.nominal_from_gait = bool(params.get('nominal_from_gait', True))
        if rollout_mode not in (None, 'gait', 'original'):
            raise ValueError(
                "rollout_mode must be None, 'gait', or 'original'"
            )
        if rollout_mode == 'gait':
            self.nominal_from_gait = True
            self.sample_type = 'cubic'
        elif rollout_mode == 'original':
            self.nominal_from_gait = False
            self.sample_type = 'cubic_original'
        self.rollout_mode = rollout_mode or 'configured'
        self.gait_startup_blend_steps = int(
            params.get('gait_startup_blend_steps', 50)
        )
        if self.gait_startup_blend_steps < 0:
            raise ValueError("gait_startup_blend_steps must be non-negative")

        # Set initial parameters and state
        self.obs = None
        self.cached_best_cost = None
        self.internal_ref = True
        self.exp_weights = np.ones(self.n_samples) / self.n_samples  # Initial MPPI weights
        self.waiting_times = waiting_times
        self.timer = Timer(end_time=self.waiting_times[0])

        # Initialize gait schedulers
        self.gaits = {
            'in_place': GaitScheduler(gait_path=GAIT_INPLACE_PATH, name='in_place'),
            'trot': GaitScheduler(gait_path=GAIT_TROT_PATH, name='trot'),
            'walk': GaitScheduler(gait_path=GAIT_WALK_PATH, name='walk'),
            'walk_fast': GaitScheduler(gait_path=GAIT_WALK_FAST_PATH, name='walk_fast')
        }
        self.gait_scheduler = self.gaits['in_place']

        # Initialize planner and goals
        self.reset_planner()
        self.goal_index = 0
        self.body_ref = np.concatenate((self.goal_pos[self.goal_index],
                                        self.goal_ori[self.goal_index],
                                        self.cmd_vel[self.goal_index],
                                        np.zeros(4)))
        
        self.gait_scheduler = self.gaits[self.desired_gait[self.goal_index]]
        self.set_noise_for_gait(self.desired_gait[self.goal_index])
        self._reset_gait_nominal(self.trajectory)
        self.task_success = False

        # Debug information
        print(f"Initial goal {self.goal_index}: {self.goal_pos[self.goal_index] }")
        print(f"Initial gait {self.desired_gait[self.goal_index]}")
        print(f"Rollout mode: {self.rollout_mode} ({self.sample_type})")
    
    def next_goal(self):
        """
        Progress to the next goal based on the task sequence.
        Updates the internal reference trajectory and gait scheduler.
        """
        self.timer.increment()

        if self.goal_index < len(self.goal_pos) - 1 and self.timer.done:
            # Move to the next goal
            previous_gait = self.gait_scheduler
            transition_source = self.trajectory.copy()
            self.goal_index += 1
            self.body_ref[:3] = self.goal_pos[self.goal_index]
            self.body_ref[7:9] = self.cmd_vel[self.goal_index]
            self.gait_scheduler = self.gaits[self.desired_gait[self.goal_index]]
            if self.gait_scheduler is not previous_gait:
                self.gait_scheduler.phase_time = 0
                self.gait_scheduler.indices = np.arange(
                    self.gait_scheduler.phase_length
                )
                self._reset_gait_nominal(transition_source)
            self.timer.reset()
            self.timer.end_time = self.waiting_times[self.goal_index]
            print(f"Moved to next goal {self.goal_index}: {self.goal_pos[self.goal_index]}")
            print(f"Gait: {self.desired_gait[self.goal_index]}")
            self.timer.waiting = False

        elif self.goal_index == len(self.goal_pos) - 1 and not self.task_success and self.timer.done:
            # Final goal reached
            print("Task succeeded.")
            self.task_success = True

        else:
            self.timer.waiting = True

        if not self.task_success:
            self.set_noise_for_gait(self.desired_gait[self.goal_index])

    def _gait_blend(self):
        """Return one startup blend factor for each point in the horizon."""
        if self.gait_startup_blend_steps == 0:
            return np.ones(self.horizon)
        horizon_steps = self._gait_nominal_step + np.arange(self.horizon)
        return np.clip(
            horizon_steps / self.gait_startup_blend_steps, 0.0, 1.0
        )

    def _build_gait_reference(self):
        """Build the phase-aligned, startup-blended [q, dq] gait horizon."""
        indices = self.gait_scheduler.indices[:self.horizon]
        gait_reference = self.gait_scheduler.gait[:, indices].copy()
        expected_rows = 2 * self.act_dim
        if gait_reference.shape != (expected_rows, self.horizon):
            raise ValueError(
                "Gait reference must have shape "
                f"({expected_rows}, {self.horizon}), got "
                f"{gait_reference.shape}"
            )

        if not self.nominal_from_gait:
            return gait_reference

        blend = self._gait_blend()
        gait_q = gait_reference[:self.act_dim].T
        source_q = self._gait_blend_source
        gait_reference[:self.act_dim] = (
            source_q + blend[:, None] * (gait_q - source_q)
        ).T
        gait_reference[self.act_dim:] *= blend[None, :]
        return gait_reference

    def _reset_gait_nominal(self, blend_source=None):
        """Reset the gait prior and its warm-started correction trajectory."""
        self._gait_nominal_step = 0
        self.gait_correction = np.zeros((self.horizon, self.act_dim))
        if blend_source is None:
            blend_source = np.repeat(
                self.sampling_init[None, :], self.horizon, axis=0
            )
        blend_source = np.asarray(blend_source, dtype=float)
        if blend_source.shape != (self.horizon, self.act_dim):
            raise ValueError(
                "gait blend source must have shape "
                f"({self.horizon}, {self.act_dim}), got {blend_source.shape}"
            )
        self._gait_blend_source = blend_source.copy()

        if self.nominal_from_gait:
            self.joints_ref = self._build_gait_reference()
            self.gait_nominal = self.joints_ref[:self.act_dim].T.copy()
            self.trajectory = self.gait_nominal.copy()
            self.selected_trajectory = self.trajectory.copy()

    def _advance_gait_nominal(self, updated_actions, nominal_actions):
        """Warm-start corrections while advancing the nominal gait phase."""
        selected_correction = updated_actions - nominal_actions
        self.gait_correction[:-1] = selected_correction[1:]
        # The new horizon tail has no optimized predecessor.  Seed it with
        # the future gait itself instead of repeating the last action.
        self.gait_correction[-1] = 0.0

        self.gait_scheduler.roll()
        self._gait_nominal_step += 1
        self._gait_blend_source[:-1] = self._gait_blend_source[1:]
        self._gait_blend_source[-1] = self._gait_blend_source[-2]
        next_reference = self._build_gait_reference()
        self.gait_nominal = next_reference[:self.act_dim].T.copy()
        self.trajectory = np.clip(
            self.gait_nominal + self.gait_correction,
            self.act_min,
            self.act_max,
        )
        
    def update(self, obs):
        """
        Update the MPPI controller based on the current observation.

        Args:
            obs (np.ndarray): Current state observation.
        Returns:
            np.ndarray: Selected action based on the optimal trajectory.
        """
        # Generate perturbed actions around the phase-aligned gait nominal.
        actions = self.perturb_action()
        self.obs = obs

        # Match the original Go1 controller: point the body toward the full
        # 3-D goal. A raised waypoint therefore generates a nose-up pitch
        # reference before the robot climbs the obstacle.
        direction = self.body_ref[:3] - obs[:3]
        goal_delta = np.linalg.norm(direction)

        # Update desired orientation based on the goal position
        if goal_delta > 0.1 and not self.timer.waiting:
            self.goal_ori = calculate_orientation_quaternion(
                obs[:3], self.body_ref[:3]
            )
        else:
            self.goal_ori = np.array([1, 0, 0, 0])

        self.body_ref[3:7] = self.goal_ori

        # Perform rollouts using threaded rollout function
        rollout_states = self.rollout_func(obs, actions)

        # Update joint references from the gait scheduler
        if self.internal_ref:
            self.joints_ref = self._build_gait_reference()
        nominal_actions = self.joints_ref[:self.act_dim].T

        # Calculate costs for each sampled trajectory
        costs_sum = self.cost_func(rollout_states, actions, self.joints_ref, self.body_ref)

        # Calculate MPPI weights for the samples
        min_cost = np.min(costs_sum)
        # Reuse the cost already computed by the main MPPI rollout for
        # trajectory logging. This avoids running an additional rollout from
        # Simulator.store_trajectory() on every simulation step.
        self.cached_best_cost = float(min_cost)
        max_cost = np.max(costs_sum)
        cost_range = max(max_cost - min_cost, np.finfo(float).eps)
        self.exp_weights = np.exp(
            -1 / self.temperature * ((costs_sum - min_cost) / cost_range)
        )

        # Weighted average of action deltas
        weighted_delta_u = self.exp_weights.reshape(self.n_samples, 1, 1) * actions
        weighted_delta_u = np.sum(weighted_delta_u, axis=0) / (np.sum(self.exp_weights) + 1e-10)
        updated_actions = np.clip(weighted_delta_u, self.act_min, self.act_max)

        # Update the trajectory with the optimal action
        self.selected_trajectory = updated_actions
        if self.nominal_from_gait:
            self._advance_gait_nominal(updated_actions, nominal_actions)
        else:
            self.gait_scheduler.roll()
            self.trajectory = np.roll(updated_actions, shift=-1, axis=0)
            self.trajectory[-1] = updated_actions[-1]

        # Return the first action in the trajectory as the output action
        return updated_actions[0]
    
    def quaternion_rotation_error_np(self, q, q_ref):
        """Return the shortest relative rotation vector for [w, x, y, z]."""
        eps = 1e-12
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), eps)
        q_ref = q_ref / np.maximum(
            np.linalg.norm(q_ref, axis=1, keepdims=True), eps
        )

        # Relative orientation q_error = conjugate(q_ref) * q.
        rw, rx, ry, rz = q_ref.T
        w, x, y, z = q.T
        q_error = np.column_stack(
            (
                rw * w + rx * x + ry * y + rz * z,
                rw * x - rx * w - ry * z + rz * y,
                rw * y + rx * z - ry * w - rz * x,
                rw * z - rx * y + ry * x - rz * w,
            )
        )

        # q and -q represent the same orientation. Select the branch whose
        # rotation angle is in [0, pi].
        q_error[q_error[:, 0] < 0.0] *= -1.0
        scalar = np.clip(q_error[:, 0], -1.0, 1.0)
        vector = q_error[:, 1:]
        vector_norm = np.linalg.norm(vector, axis=1)
        angle = 2.0 * np.arctan2(vector_norm, scalar)
        scale = np.divide(
            angle,
            vector_norm,
            out=np.full_like(angle, 2.0),
            where=vector_norm > eps,
        )
        return vector * scale[:, None]


    def quadruped_cost_np(self, x, u, x_ref):
        """
        Compute the cost for quadruped motion based on state and action errors.

        Args:
            x (np.ndarray): Current states (N x state_dim).
            u (np.ndarray): Current actions (N x action_dim).
            x_ref (np.ndarray): Reference states (N x state_dim).

        Returns:
            np.ndarray: Computed cost for each sample.
        """
        # Match the PD law encoded by each MuJoCo actuator.  For the B2-Z1
        # model, biasprm[2] stores the negative damping coefficient:
        # torque = gainprm[0] * (ctrl - q) - kd * qvel.
        kp = np.asarray(self.model.actuator_gainprm[:, 0], dtype=float)
        kd = -np.asarray(self.model.actuator_biasprm[:, 2], dtype=float)

        # Compute state error relative to the reference
        x_error = x - x_ref

        # Store the relative rotation vector in the quaternion slots so that
        # Q_diag[4:7] independently weights roll, pitch, and yaw errors.
        rotation_error = self.quaternion_rotation_error_np(
            x[:, 3:7], x_ref[:, 3:7]
        )
        x_error[:, 3] = 0.0
        x_error[:, 4:7] = rotation_error

        # Compute joint and velocity errors
        x_joint = x[:, 7:23]
        v_joint = x[:, 29:45]
        uv = x_ref[:, 29:45]
        u_error = kp * (u - x_joint) - kd * (v_joint)

        # Compute positional cost (L1 norm for positional error)
        x_error[:, :3] = 0  # Ignore positional error for simplicity
        x_pos_error = x[:, :3] - x_ref[:, :3]
        L1_norm_pos_cost = np.abs(np.dot(x_pos_error, self.Q[:3, :3])).sum(axis=1)

        # Compute total cost
        cost = (
            np.einsum('ij,ik,jk->i', x_error, x_error, self.Q) +
            np.einsum('ij,ik,jk->i', u_error, u_error, self.R) +
            L1_norm_pos_cost
        )
        return cost


    def calculate_total_cost(self, states, actions, joints_ref, body_ref):
        """
        Calculate the total cost for all rollouts.

        Args:
            states (np.ndarray): Rollout states (samples x time steps x state_dim).
            actions (np.ndarray): Rollout actions (samples x time steps x action_dim).
            joints_ref (np.ndarray): Reference joint positions (time steps x joint_dim).
            body_ref (np.ndarray): Reference body state (state_dim).

        Returns:
            np.ndarray: Total cost for each sample.
        """
        num_samples = states.shape[0]
        num_pairs = states.shape[1]

        # Repeat the base pose and construct the full B2-Z1 base-velocity
        # reference. The task stores commanded x/y velocity in body_ref[7:9].
        traj_body_ref = np.repeat(
            body_ref[np.newaxis, :], num_samples * num_pairs, axis=0
        )

        # Flatten states and actions for batch processing
        states = states.reshape(-1, states.shape[2])
        actions = actions.reshape(-1, actions.shape[2])

        # Repeat and reshape the 32-row [q, dq] joint reference.
        joints_ref = np.tile(joints_ref.T, (num_samples, 1, 1))
        joints_ref = joints_ref.reshape(-1, joints_ref.shape[2])
        base_velocity_ref = np.zeros((len(joints_ref), 6))
        base_velocity_ref[:, :2] = traj_body_ref[:, 7:9]

        # B2-Z1 state order: base qpos, 16 joint q, base dq, 16 joint dq.
        x_ref = np.concatenate(
            [traj_body_ref[:, :7], joints_ref[:, :16], base_velocity_ref, joints_ref[:, 16:],],
            axis=1,
        )

        # Rotate velocity vectors to the local frame
        rotated_ref = batch_world_to_local_velocity(states[:, 3:7], states[:, 23:26])
        states[:, 23:26] = rotated_ref

        # Compute cost for each rollout
        costs = self.quadruped_cost_np(states, actions, x_ref)

        # Sum costs across time steps for each sample
        total_costs = costs.reshape(num_samples, num_pairs).sum(axis=1)
        return total_costs

    def eval_best_trajectory(self):
        """
        Return the cached cost of the best sample from the latest MPPI update.

        Returns:
            float: Latest best sampled-trajectory cost, or None before the
                first controller update.
        """
        return self.cached_best_cost

    def __del__(self):
        if hasattr(self, "_closed") and hasattr(self, "executor"):
            self.close()
    
if __name__ == "__main__":
    mppi = MPPI()
