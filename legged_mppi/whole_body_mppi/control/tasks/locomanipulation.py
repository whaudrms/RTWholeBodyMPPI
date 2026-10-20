"""Locomanipulation goals, gait references and controller lifecycle."""
import numpy as np

from whole_body_mppi.control.tasks.definitions import get_task
from whole_body_mppi.control.tasks.base import (
    BaseTaskController, PACKAGE_ROOT, CONTROL_ROOT, GAIT_DIR)
from whole_body_mppi.control.costs.locomanipulation import LocomanipulationCost
from whole_body_mppi.control.gait_scheduler.scheduler import GaitScheduler
from whole_body_mppi.utils.transforms import calculate_orientation_quaternion

# Paths for gait files
GAIT_INPLACE_PATH = str(GAIT_DIR / "FAST/walking_gait_raibert_FAST_0_0_10cm_100hz.tsv")
GAIT_TROT_PATH = str(GAIT_DIR / "MED/walking_gait_raibert_MED_0_5_15cm_100hz.tsv")
GAIT_WALK_PATH = str(GAIT_DIR / "MED/walking_gait_raibert_MED_0_1_10cm_100hz.tsv")

class LocomanipulationController(LocomanipulationCost, BaseTaskController):
    def __init__(self, task='push_box') -> None:
        """
        Initialize the MPPI controller with task-specific configurations.

        Args:
            task (str): The registered task name, such as 'push_box' or 'push_box_hw'.
        """
        print("Task: ", task)

        # Retrieve task-specific parameters
        self.task = task
        self.follow_box = False
        self.task_data = get_task(task)
        self.goal_pos = self.task_data['goal_pos']
        self.goal_ori = self.task_data['default_orientation']
        self.cmd_vel = self.task_data['cmd_vel']
        self.goal_thresh = self.task_data['goal_thresh']
        self.desired_gait = self.task_data['desired_gait']
        model_path = self.task_data['model_path']
        config_path = self.task_data['config_path']

        # Dynamically resolve paths for model and configuration files
        CONFIG_PATH = CONTROL_ROOT / config_path
        MODEL_PATH = PACKAGE_ROOT / model_path

        # Initialize base MPPI
        super().__init__(MODEL_PATH, CONFIG_PATH)

        params = self.params
        # Cost
        self.Q_robot = np.diag(np.array(params['Q_robot']))
        self.Q_box = np.diag(np.array(params['Q_box']))
        self.R = np.diag(np.array(params['R_diag']))
        self.x_box_ref = np.concatenate([np.array(params['box_pos_ref']), np.array(params['box_quat_ref'])])


        # Set initial parameters and state
        self.obs = None
        self.internal_ref = True
        self.exp_weights = np.ones(self.n_samples) / self.n_samples  # Initial MPPI weights

        # Initialize gait schedulers
        self.gaits = {
            'in_place': GaitScheduler(gait_path=GAIT_INPLACE_PATH, name='in_place'),
            'trot': GaitScheduler(gait_path=GAIT_TROT_PATH, name='trot'),
            'walk': GaitScheduler(gait_path=GAIT_WALK_PATH, name='walk')
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
        self.task_success = False
        self.box_state = np.zeros(13)
        self.robot_state = np.zeros(37)

        # Debug information
        print(f"Initial goal {self.goal_index}: {self.goal_pos[self.goal_index] }")
        print(f"Initial gait {self.desired_gait[self.goal_index]}")

    def next_goal(self):
        """
        Progress to the next goal based on the task sequence.
        Updates the internal reference trajectory and gait scheduler.
        """
        if (self.goal_index == 0 and (len(self.goal_pos) - 1 > 0 and self.gait_scheduler.phase_time > 300)):
                if self.goal_index == 0:
                    self.follow_box = True
                self.goal_index = (self.goal_index + 1)
                self.body_ref[:2] = self.box_state[:2]
                self.body_ref[7:9] = self.cmd_vel[self.goal_index]
                self.gait_scheduler = self.gaits[self.desired_gait[self.goal_index]]
        elif self.follow_box:
            self.prev_body_ref = self.body_ref[:2]*1
            self.body_ref[:2] = self.box_state[:2]
            print("still working...")
            if np.linalg.norm(np.array(self.box_state[:2])-np.array(self.x_box_ref[:2])) < 0.3:
                print("Task succeed. Time to back up!")
                self.follow_box = False
                self.goal_index = (self.goal_index + 1)
                # Change Q box to 0
                self.Q_box = np.diag(np.zeros(3))
                self.body_ref[:2] = self.prev_body_ref
                self.body_ref[7:9] = [0,0]
                self.gait_scheduler = self.gaits['in_place']
        else:
            pass

    def update(self, obs):
        """
        Update the MPPI controller based on the current observation.

        Args:
            obs (np.ndarray): Current state observation.
        Returns:
            np.ndarray: Selected action based on the optimal trajectory.
        """
        self.obs = obs

        # Update the robot and box states
        self.box_state[:7] = obs[:7]
        self.robot_state[:19] = obs[7:26]
        self.box_state[7:] = obs[26:32]
        self.robot_state[19:] = obs[32:]

        # Calculate the direction and distance to the goal
        direction = self.body_ref[:3] - self.robot_state[:3]
        goal_delta = np.linalg.norm(direction)

        # Update desired orientation based on the goal position
        if (goal_delta > 0.1) and ((self.gait_scheduler == self.gaits['walk']) or (self.gait_scheduler == self.gaits['trot'])):
            self.goal_ori = calculate_orientation_quaternion(self.robot_state[:3], self.body_ref[:3])
        elif self.gait_scheduler == self.gaits['in_place']:
            self.goal_ori = np.array([1,0,0,0])

        self.body_ref[3:7] = self.goal_ori
        if self.internal_ref:
            self.joints_ref = self.gait_scheduler.gait[:, self.gait_scheduler.indices[:self.horizon]]

        return self.solve_task(obs)

    def task_cost(self, states, actions):
        return self.cost_func(states[:, :, np.r_[8:27, 33:51]],
                              states[:, :, np.r_[1:8, 27:33]],
                              actions, self.joints_ref, self.body_ref)
