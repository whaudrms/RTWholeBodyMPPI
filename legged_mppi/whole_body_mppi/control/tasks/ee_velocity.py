"""Pure body-local XY and world-frame EE velocity command tracking."""
import numpy as np

from whole_body_mppi.control.tasks.whole_body import WholeBodyController
from whole_body_mppi.control.costs.end_effector import EndEffectorKinematics


class EEVelocityController(WholeBodyController):
    def __init__(self, task='ee_reach_cmd_vel', robot='go1_dm', rollout_model='full'):
        super().__init__(task, robot, rollout_model)
        settings, cfg = self.task_data, self.params['whole_body']
        if settings.get('control_mode') != 'cmd_vel':
            raise ValueError('EEVelocityController requires a cmd_vel task')
        if settings['body_velocity_frame'] != 'body' or settings['ee_velocity_frame'] != 'world':
            raise ValueError('cmd_vel requires body-local body velocity and world EE velocity')
        self.command_phases = settings['command_phases']
        durations = np.asarray([p['duration'] for p in self.command_phases], dtype=float)
        self.body_commands = np.asarray([p['body_cmd_vel'] for p in self.command_phases], dtype=float)
        self.ee_commands = np.asarray([p['ee_cmd_vel'] for p in self.command_phases], dtype=float)
        if (not len(durations) or not np.all(np.isfinite(durations) & (durations > 0))
                or self.body_commands.shape != (len(durations), 2)
                or self.ee_commands.shape != (len(durations), 3)
                or not np.isfinite(self.body_commands).all() or not np.isfinite(self.ee_commands).all()):
            raise ValueError('command_phases requires positive durations and finite body XY / EE XYZ velocities')
        if any(p['gait'] not in self.gaits for p in self.command_phases):
            raise ValueError('Unknown command phase gait')
        # The final command remains active after its duration.
        if np.any(self.body_commands[-1]) or np.any(self.ee_commands[-1]):
            raise ValueError('Final command phase must stop body and EE')
        if np.any(self.Q[:2]) or np.any(self.Q[:, :2]):
            raise ValueError('cmd_vel tasks require zero body XY position weights')
        self.phase_ends = np.cumsum(durations)
        self.velocity_weight = float(cfg['velocity_weight'])
        self.delta_weights = np.asarray(cfg['action_delta_diag'], dtype=float)
        if not np.isfinite(self.velocity_weight) or self.velocity_weight < 0:
            raise ValueError('velocity_weight must be finite and nonnegative')
        if self.delta_weights.shape != (18,) or not np.all(np.isfinite(self.delta_weights)) or np.any(self.delta_weights < 0):
            raise ValueError('action_delta_diag must contain 18 finite nonnegative weights')
        self.previous_action = self.sampling_init.copy()
        self.plan_previous_action = None
        self.command_time = 0.
        self.command_phase_index = -1
        self.ee_command = np.zeros(3)
        self.set_command_time(0.)

    def initialize_ee_objective(self, task_data, cfg):
        if any(k in task_data for k in ('ee_waypoints', 'goal_pos', 'ee_position_gain')):
            raise ValueError('cmd_vel tasks cannot define position goals or waypoint feedback')
        if 'weight' in cfg or 'terminal_weight' in cfg:
            raise ValueError('cmd_vel tasks cannot define EE position/terminal weights')
        self.ee_kinematics = EndEffectorKinematics(self.model, frame='world')

    def ee_position_velocity(self, qpos, qvel):
        return self.ee_kinematics.position_velocity(qpos, qvel)

    def phase_indices(self, times):
        return np.minimum(np.searchsorted(self.phase_ends, np.asarray(times)+1e-10, side='right'), len(self.command_phases)-1)

    def set_command_time(self, simulation_time):
        self.command_time = float(simulation_time)
        index = int(self.phase_indices(simulation_time))
        if index != self.command_phase_index:
            self.gait_scheduler = self.gaits[self.command_phases[index]['gait']]
        self.command_phase_index = index
        phase_start = 0. if index == 0 else self.phase_ends[index-1]
        gait_step = max(0, int(np.floor((simulation_time-phase_start+1e-10)/self.h)))
        self.gait_scheduler.phase_time = gait_step
        self.gait_scheduler.indices = np.roll(np.arange(self.gait_scheduler.phase_length), -gait_step)
        self.body_ref[7:9] = self.body_commands[index]
        self.ee_command = self.ee_commands[index].copy()

    def update(self, obs):
        self.obs = obs
        if self.internal_ref:
            times = self.command_time + np.arange(self.horizon)*self.h
            phases = self.phase_indices(times)
            refs = []
            for step, phase in enumerate(phases):
                gait = self.gaits[self.command_phases[int(phase)]['gait']]
                start = 0. if phase == 0 else self.phase_ends[int(phase)-1]
                offset = max(0, int(np.floor((times[step]-start+1e-10)/self.h)))
                refs.append(gait.gait[:, offset % gait.phase_length])
            self.joints_ref = np.stack(refs, axis=1)
        self.plan_previous_action = self.previous_action.copy()
        action = self.solve_task(obs)
        self.previous_action = action.copy()
        return action

    def task_cost(self, states, actions):
        steps = states.shape[1]
        indices = self.phase_indices(self.command_time + np.arange(steps)*self.h)
        refs = np.broadcast_to(self.body_ref, (steps, 13)).copy()
        refs[:, 7:9] = self.body_commands[indices]
        cost = self.cost_func(states[..., 1:], actions, self.joints_ref[:, :steps], refs)
        _, velocity = self.ee_position_velocity(states[..., 1:26], states[..., 26:50])
        cost += self.velocity_weight*np.sum((velocity-self.ee_commands[indices])**2, axis=(1, 2))
        previous = self.previous_action if self.plan_previous_action is None else self.plan_previous_action
        delta = np.diff(actions, axis=1, prepend=np.broadcast_to(previous, (len(actions), 1, 18)))
        return cost + np.sum(delta**2*self.delta_weights, axis=(1, 2))
