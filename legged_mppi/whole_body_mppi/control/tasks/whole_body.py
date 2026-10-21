"""Joint leg/arm optimization with an end-effector position target."""
import numpy as np

from whole_body_mppi.control.costs.whole_body import WholeBodyCost
from whole_body_mppi.control.tasks.locomotion import LocomotionController
from whole_body_mppi.control.costs.end_effector import EndEffectorObjective
from whole_body_mppi.control.tasks.definitions import get_task


class WholeBodyController(WholeBodyCost, LocomotionController):
    def __init__(self, task='ee_reach', robot='go1_dm', rollout_model='full'):
        task_data = get_task(task)
        if task_data.get('controller') != 'whole_body':
            raise ValueError('WholeBodyController requires an EE task: ee_reach or ee_walk')
        if robot != task_data['required_robot'] or rollout_model != task_data['required_rollout_model']:
            raise ValueError('EE tasks require --robot go1_dm --rollout-model full')
        super().__init__(task, robot, rollout_model)
        cfg = self.params['whole_body']
        self.ee_objective = EndEffectorObjective(self.model, task_data['ee_target'], task_data['ee_frame'],
                                                cfg['weight'], cfg['terminal_weight'])
        self.arm_noise = np.asarray(cfg['arm_noise_sigma'], dtype=float)
        self.arm_joint_ref = np.asarray(task_data['arm_joint_ref'], dtype=float)
        self.arm_velocity_ref = np.asarray(task_data['arm_velocity_ref'], dtype=float)
        for name, ref in (('arm_joint_ref', self.arm_joint_ref), ('arm_velocity_ref', self.arm_velocity_ref)):
            if ref.shape != (6,) or not np.all(np.isfinite(ref)):
                raise ValueError(f'{name} must contain six finite values')
        for name, matrix, size in (('Q_diag', self.Q, 49), ('R_diag', self.R, 18)):
            if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)) or np.any(np.diag(matrix) < 0):
                raise ValueError(f'{name} must contain {size} finite nonnegative weights')
        if self.arm_noise.shape != (6,) or not np.all(np.isfinite(self.arm_noise) & (self.arm_noise > 0)):
            raise ValueError('whole_body.arm_noise_sigma must contain six positive finite values')
        self.act_dim = self.model.nu
        self.sampling_init = self.model.key('home').ctrl.copy()
        self.act_min = np.r_[self.act_min, self.model.actuator_ctrlrange[12:, 0]]
        self.act_max = np.r_[self.act_max, self.model.actuator_ctrlrange[12:, 1]]
        self.noise_sigma = np.r_[self.noise_sigma, self.arm_noise]
        self.rollout_func = self.rollout_backend.threaded_rollout
        self.reset_planner()

    def next_goal(self):
        super().next_goal()
        if self.noise_sigma.shape == (12,):
            self.noise_sigma = np.r_[self.noise_sigma, self.arm_noise]

    def task_cost(self, states, actions):
        state_cost = self.cost_func(states[..., 1:], actions, self.joints_ref, self.body_ref)
        qpos = states[..., 1:1+self.model.nq]
        return state_cost + self.ee_objective.cost(qpos)
