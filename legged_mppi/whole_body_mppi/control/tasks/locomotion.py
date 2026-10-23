"""Locomotion goals, gait references and controller lifecycle."""
import numpy as np

from whole_body_mppi.control.tasks.definitions import get_task
from whole_body_mppi.control.tasks.base import (
    BaseTaskController, PACKAGE_ROOT, CONTROL_ROOT, GAIT_DIR)
from whole_body_mppi.control.costs.locomotion import LocomotionCost
from whole_body_mppi.control.gait_scheduler.scheduler import GaitScheduler, Timer
from whole_body_mppi.utils.transforms import calculate_orientation_quaternion

# Paths for gait files
GAIT_INPLACE_PATH = str(GAIT_DIR / "FAST/walking_gait_raibert_FAST_0_0_10cm_100hz.tsv")
GAIT_TROT_PATH = str(GAIT_DIR / "MED/walking_gait_raibert_MED_0_5_15cm_100hz.tsv")
GAIT_WALK_PATH = str(GAIT_DIR / "MED/walking_gait_raibert_MED_0_1_10cm_100hz.tsv")
GAIT_WALK_FAST_PATH = str(GAIT_DIR / "FAST/walking_gait_raibert_FAST_0_1_10cm_100hz.tsv")

class LocomotionController(LocomotionCost, BaseTaskController):
    def __init__(self, task='stand', robot='go1', rollout_model='full') -> None:
        """
        Initialize the MPPI controller with task-specific configurations.

        Args:
            task (str): The name of the task ('stand', 'walk').
        """
        print("Task: ", task)

        # Retrieve task-specific parameters
        if robot not in ('go1', 'go1_dm'):
            raise ValueError(f'Unknown robot: {robot}')
        if rollout_model not in ('simple', 'full'):
            raise ValueError(f'Unknown rollout model: {rollout_model}')
        self.rollout_model = rollout_model
        self.robot = robot
        self.task = task
        self.task_data = get_task(task)

        self.velocity_tracking = self.task_data.get('tracking_mode') == 'cmd_vel'
        if self.velocity_tracking and robot != 'go1':
            raise ValueError('walk_straight_cmd_vel currently supports Go1 only')

        self.goal_pos = self.task_data['goal_pos']
        self.goal_ori = self.task_data['default_orientation']
        self.cmd_vel = self.task_data['cmd_vel']
        self.goal_thresh = self.task_data['goal_thresh']
        self.desired_gait = self.task_data['desired_gait']
        model_path = self.task_data['model_path']
        config_path = self.task_data['config_path']
        waiting_times = self.task_data['waiting_times']

        # Dynamically resolve paths for model and configuration files
        CONFIG_PATH = CONTROL_ROOT / config_path
        MODEL_PATH = PACKAGE_ROOT / model_path

        # Initialize base MPPI
        model = None
        if robot == 'go1_dm':
            from whole_body_mppi.models.go1_dm.rollout_model import load_runtime_model
            model = load_runtime_model(MODEL_PATH, rollout_model)
        super().__init__(MODEL_PATH, CONFIG_PATH, model=model)
        if robot == 'go1_dm':
            from whole_body_mppi.control.rollouts.mounted_arm import MountedArmRollout
            self.rollout_func = MountedArmRollout(self.rollout_backend, self.h)

        params = self.params
        # Cost weights
        self.Q = np.diag(np.array(params['Q_diag']))
        self.R = np.diag(np.array(params['R_diag']))

        # Set initial parameters and state
        self.obs = None
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
                                        self.cmd_vel[self.goal_index][:2],
                                        np.zeros(4)))

        self.gait_scheduler = self.gaits[self.desired_gait[self.goal_index]]
        self.task_success = False
        if self.velocity_tracking:
            self.command_target = np.asarray(self.cmd_vel[0], dtype=float).copy()
            self.command_velocity = np.zeros(3)
            self.command_acceleration = np.asarray(self.task_data['command_acceleration'], dtype=float)
            self.body_ref[7:9] = 0
            self.Q[:2, :] = 0
            self.Q[:, :2] = 0

        # Debug information
        print(f"Initial goal {self.goal_index}: {self.goal_pos[self.goal_index] }")
        print(f"Initial gait {self.desired_gait[self.goal_index]}")

    def next_goal(self):
        """
        Progress to the next goal based on the task sequence.
        Updates the internal reference trajectory and gait scheduler.
        """
        if self.velocity_tracking:
            return  # Continuous task; proximity never completes or changes the command.
        self.timer.increment()

        if self.goal_index < len(self.goal_pos) - 1 and self.timer.done:
            # Move to the next goal
            self.goal_index += 1
            self.body_ref[:3] = self.goal_pos[self.goal_index]
            self.body_ref[7:9] = self.cmd_vel[self.goal_index]
            self.gait_scheduler = self.gaits[self.desired_gait[self.goal_index]]
            self.timer.reset()
            self.timer.end_time = self.waiting_times[self.goal_index]
            print(f"Moved to next goal {self.goal_index}: {self.goal_pos[self.goal_index]}")
            print(f"Gait: {self.desired_gait[self.goal_index]}")
            self.timer.waiting = False

        elif self.goal_index == len(self.goal_pos) - 1 and not self.task_success and self.timer.done:
            # Final goal reached
            print("Locomotion task succeeded." if hasattr(self, "ee_objective") else "Task succeeded.")
            self.task_success = True

        else:
            self.timer.waiting = True

        if not self.task_success:
            if self.desired_gait[self.goal_index] in ['in_place', 'walk', 'walk_fast']:
                self.noise_sigma = np.array([0.06, 0.1, 0.1] * 4)
            elif self.desired_gait[self.goal_index] in ['trot']:
                self.noise_sigma = np.array([0.06, 0.2, 0.2] * 4)

    def set_cmd_vel(self, vx, vy, yaw_rate):
        """Set heading-frame m/s and Euler yaw rate in rad/s (rate limited on update)."""
        if not self.velocity_tracking:
            raise ValueError('set_cmd_vel requires a cmd_vel tracking task')
        command = np.asarray([vx, vy, yaw_rate], dtype=float)
        if not np.isfinite(command).all():
            raise ValueError('cmd_vel must be finite')
        self.command_target = command

    def update(self, obs):
        """
        Update the MPPI controller based on the current observation.

        Args:
            obs (np.ndarray): Current state observation.
        Returns:
            np.ndarray: Selected action based on the optimal trajectory.
        """
        self.obs = obs

        if self.velocity_tracking:
            limit = self.command_acceleration * self.h
            self.command_velocity += np.clip(self.command_target - self.command_velocity, -limit, limit)
            self.body_ref[7:9] = self.command_velocity[:2]
            self.body_ref[12] = self.command_velocity[2]
            if self.internal_ref:
                self.joints_ref = self.gait_scheduler.gait[:, self.gait_scheduler.indices[:self.horizon]]
            return self.solve_task(obs)

        # Calculate the direction and distance to the goal
        direction = self.body_ref[:3] - obs[:3]
        goal_delta = np.linalg.norm(direction)

        # Update desired orientation based on the goal position
        if goal_delta > 0.1 and not self.timer.waiting:
            self.goal_ori = calculate_orientation_quaternion(obs[:3], self.body_ref[:3])
        else:
            self.goal_ori = np.array([1, 0, 0, 0])

        self.body_ref[3:7] = self.goal_ori

        if self.internal_ref:
            self.joints_ref = self.gait_scheduler.gait[:, self.gait_scheduler.indices[:self.horizon]]

        action = self.solve_task(obs)
        if self.robot == 'go1_dm' and self.act_dim == 12:
            return self.rollout_func.expand_actions(action)
        return action

    def task_cost(self, states, actions):
        if self.velocity_tracking:
            from whole_body_mppi.control.costs.cmd_vel import command_velocity_cost
            return command_velocity_cost(self, states, actions)
        if self.robot == 'go1_dm':
            # FULLPHYSICS = time + qpos + qvel; omit both arm position and velocity.
            nq = self.model.nq
            states = states[:, :, np.r_[1:20, 1+nq:1+nq+18]]
        else:
            states = states[:, :, 1:]
        return self.cost_func(states, actions, self.joints_ref, self.body_ref)
