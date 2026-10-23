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
        self.initialize_ee_objective(task_data, cfg)
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
        # Keep the 18-actuator model/cost layout, but project every sampled
        # trajectory onto fixed command bounds for the selected arm joints.
        fixed = task_data.get('arm_fixed_joints', [])
        if any(type(j) is not int or not 1 <= j <= 6 for j in fixed) or len(set(fixed)) != len(fixed):
            raise ValueError('arm_fixed_joints must be unique 1-based indices in 1..6')
        self.fixed_arm_actions = 12 + np.asarray(fixed, dtype=int) - 1
        values = self.arm_joint_ref[np.asarray(fixed, dtype=int)-1]
        if np.any(values < self.act_min[self.fixed_arm_actions]) or np.any(values > self.act_max[self.fixed_arm_actions]):
            raise ValueError('Fixed arm reference exceeds actuator control limits')
        self.act_min[self.fixed_arm_actions] = values
        self.act_max[self.fixed_arm_actions] = values
        self.sampling_init[self.fixed_arm_actions] = values
        self.noise_sigma = np.r_[self.noise_sigma, self.arm_noise]
        self.rollout_func = self.rollout_backend.threaded_rollout
        self.reset_planner()

    def initialize_ee_objective(self, task_data, cfg):
        self.waypoints = np.asarray(task_data['ee_waypoints'], dtype=float)
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 3 or not len(self.waypoints) or not np.all(np.isfinite(self.waypoints)):
            raise ValueError('ee_waypoints must be a nonempty list of finite xyz targets')
        for name in ('ee_position_tolerance', 'ee_speed_tolerance', 'ee_settle_time'):
            if not np.isfinite(task_data[name]) or task_data[name] <= 0:
                raise ValueError(f'{name} must be finite and positive')
        self.waypoint_index = 0
        self.settle_elapsed = 0.
        self.ee_holding = False
        self.ee_objective = EndEffectorObjective(self.model, self.waypoints[0], task_data['ee_frame'],
                                                cfg['weight'], cfg['terminal_weight'])

    def ee_position_velocity(self, qpos, qvel):
        return self.ee_objective.position_velocity(qpos, qvel)

    def advance_ee_waypoint(self, position, velocity, dt=None):
        settled = (np.linalg.norm(position-self.ee_objective.target) < self.task_data['ee_position_tolerance']
                   and np.linalg.norm(velocity) < self.task_data['ee_speed_tolerance'])
        self.settle_elapsed = self.settle_elapsed+(self.params['dt'] if dt is None else dt) if settled else 0.
        self.ee_holding = False
        if self.settle_elapsed >= self.task_data['ee_settle_time']:
            if self.waypoint_index < len(self.waypoints)-1:
                self.waypoint_index += 1
                self.ee_objective.target = self.waypoints[self.waypoint_index].copy()
                self.settle_elapsed = 0.
            else:
                self.ee_holding = True

    def update(self, obs):
        position, velocity = self.ee_position_velocity(obs[:self.model.nq], obs[self.model.nq:])
        if not getattr(self, 'external_reference_updates', False):
            self.advance_ee_waypoint(position, velocity)
        return super().update(obs)

    def next_goal(self):
        super().next_goal()
        if self.noise_sigma.shape == (12,):
            self.noise_sigma = np.r_[self.noise_sigma, self.arm_noise]

    def task_cost(self, states, actions):
        state_cost = self.cost_func(states[..., 1:], actions, self.joints_ref, self.body_ref)
        qpos = states[..., 1:1+self.model.nq]
        return state_cost + self.ee_objective.cost(qpos)
