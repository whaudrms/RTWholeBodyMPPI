import mujoco
import os
import time as wall_time
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import tqdm
from PIL import Image

class Simulator:
    """
    A class representing a simulator for controlling and estimating the state of a system.
    
    Attributes:
        filter (object): The filter used for state estimation.
        agent (object): The agent used for control.
        model_path (str): The path to the XML model file.
        T (int): The number of time steps.
        dt (float): The time step size.
        viewer (bool): Flag indicating whether to enable the viewer.
        gravity (bool): Flag indicating whether to enable gravity.
        model (object): The MuJoCo model.
        data (object): The MuJoCo data.
        qpos (ndarray): The position trajectory.
        qvel (ndarray): The velocity trajectory.
        finite_diff_qvel (ndarray): The finite difference of velocity.
        ctrl (ndarray): The control trajectory.
        sensordata (ndarray): The sensor data trajectory.
        noisy_sensordata (ndarray): The noisy sensor data trajectory.
        time (ndarray): The time trajectory.
        state_estimate (ndarray): The estimated state trajectory.
        viewer (object): The MuJoCo viewer.
    """
    def __init__(self, agent=None,
                 model_path = os.path.join(os.path.dirname(__file__), "../models/go1/task_simulate.xml"),
                T = 200, dt = 0.01, viewer = True, gravity = True,
                # stiff=False
                timeconst=0.02, dampingratio=1.0, ctrl_rate=100,
                save_dir="./frames", save_frames=False, robot="go1", initial_key=1, model_variant="full", sync=False
                ):
        # filter

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.agent = agent
        self.sync = sync
        self.ctrl_rate = ctrl_rate
        ratio = 1 / (dt * ctrl_rate)
        if ratio < 1 or not np.isclose(ratio, round(ratio)):
            raise ValueError('Control period must be an integer multiple of physics timestep')
        self.update_ratio = round(ratio)
        self.interpolate_cam = False
        # model
        if robot == 'go1_dm':
            from whole_body_mppi.models.go1_dm.rollout_model import load_runtime_model
            self.model = load_runtime_model(model_path, model_variant, dt)
        else:
            self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.model.opt.timestep = dt
        #self.model.opt.enableflags = 1 # to override contact settings
        self.model.opt.o_solref = np.array([timeconst, dampingratio])
        # data
        self.data = mujoco.MjData(self.model)
        self.T = T
        # save
        self.save_frames = save_frames
        self.save_dir = save_dir
        # rollout
        mujoco.mj_resetData(self.model, self.data)
        key_id = self.model.key(initial_key).id if isinstance(initial_key, str) else initial_key
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)

        # The display model is independent of rollout/physics complexity.
        self.robot = robot
        self.model_path = model_path
        self.render_model, self.render_data = self.model, self.data
        self._detailed_visual = None
        # viewer
        if viewer:
            self.initialize_rendering()
            import mujoco_viewer
            self.viewer = mujoco_viewer.MujocoViewer(self.render_model, self.render_data, hide_menus=True)
        else:
            self.viewer = None

        # trajectories
        self.qpos = np.zeros((self.model.nq, self.T))
        self.qvel = np.zeros((self.model.nv, self.T))
        self.ctrl = np.zeros((self.model.nu, self.T))
        self.time = np.zeros(self.T)
        self.cost = np.zeros((1, self.T))
        self.ee_errors = np.zeros(self.T) if hasattr(agent, 'ee_objective') else None
        self.velocity_task = hasattr(agent, "set_command_time")
        self.command_phase_indices = np.zeros(self.T, dtype=int) if self.velocity_task else None
        self.ee_positions = np.zeros((3, self.T)) if hasattr(agent, "ee_position_velocity") else None

        self.ee_velocities = np.zeros((3, self.T)) if hasattr(agent, 'ee_command') else None
        self.ee_velocity_commands = np.zeros((3, self.T)) if self.ee_velocities is not None else None
        self.ee_targets = np.zeros((3, self.T)) if self.ee_errors is not None else None
        self.ee_world_targets = np.zeros((3, self.T)) if self.ee_errors is not None else None
        self.ee_hold_status = np.zeros(self.T, dtype=bool) if self.ee_errors is not None else None
        self.ee_waypoint_indices = np.zeros(self.T, dtype=int) if self.ee_errors is not None else None
        self.body_targets = np.zeros((3, self.T))
        self.body_velocity_commands = np.zeros((2, self.T))
        self.body_velocity_errors = np.zeros((2, self.T))
        self.body_position_errors = np.zeros(self.T)
        self.solve_seconds = np.full(self.T, np.nan)
        self.cost_seconds = np.full(self.T, np.nan)
        self.plan_age_seconds = np.full(self.T, np.nan)
        self.plan_application_delay_seconds = np.full(self.T, np.nan)
        self.sync_lag_seconds = np.full(self.T, np.nan)
        self.plan_exhausted = np.zeros(self.T, dtype=bool)
        self.latest_plan_cost = np.nan
        self.cost[:] = np.nan
        self.recorded_steps = 0
        self.run_wall_seconds = 0.
        self.output_dir = None

        # Ensure the directory exists
        if self.save_frames and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def initialize_rendering(self):
        """Lazily load detailed DM visuals; headless runs avoid this work."""
        if self.robot == 'go1_dm' and self._detailed_visual is None:
            from whole_body_mppi.models.go1_dm.visualization import DetailedDMVisual
            self._detailed_visual = DetailedDMVisual(self.model_path, self.model)
            self.render_model = self._detailed_visual.model
            self.render_data = self._detailed_visual.data
        self.sync_render_state()

    def sync_render_state(self):
        if self._detailed_visual is not None:
            self._detailed_visual.sync(self.data)

    def step(self, ctrl=None):
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)
        return self.data.qpos, self.data.qvel
    
    def store_trajectory(self, t):
        self.qpos[:, t] = self.data.qpos
        self.qvel[:, t] = self.data.qvel
        self.ctrl[:, t] = self.data.ctrl
        self.time[t] = self.data.time
        if self.ee_errors is not None:
            objective = self.agent.ee_objective
            position = objective.positions(self.data.qpos)
            if objective.frame == 'world':
                self.ee_positions[:, t] = position
            else:
                from whole_body_mppi.control.costs.end_effector import quaternion_matrix
                self.ee_positions[:, t] = self.data.qpos[:3]+quaternion_matrix(self.data.qpos[3:7])@position
            self.ee_errors[t] = np.linalg.norm(position-objective.target)
        if self.ee_velocities is not None:
            self.ee_positions[:, t], self.ee_velocities[:, t] = self.agent.ee_position_velocity(self.data.qpos, self.data.qvel)
            self.command_phase_indices[t] = self.agent.command_phase_index
            self.ee_velocity_commands[:, t] = self.agent.ee_command
        if self.ee_errors is not None:
            self.ee_targets[:, t] = self.agent.ee_objective.target
            self.ee_world_targets[:, t] = self.agent.ee_objective.target_world(self.data.qpos)
            self.ee_hold_status[t] = getattr(self.agent, 'ee_holding', False)
            self.ee_waypoint_indices[t] = getattr(self.agent, 'waypoint_index', 0)
        if self.agent is not None:
            from whole_body_mppi.control.costs.end_effector import quaternion_matrix
            self.body_targets[:, t] = self.agent.body_ref[:3]
            self.body_velocity_commands[:, t] = self.agent.body_ref[7:9]
            local_velocity = quaternion_matrix(self.data.qpos[3:7]).T @ self.data.qvel[:3]
            self.body_velocity_errors[:, t] = local_velocity[:2]-self.agent.body_ref[7:9]
            self.body_position_errors[t] = np.linalg.norm(self.data.qpos[:3]-self.agent.body_ref[:3])
        if self.velocity_task:
            self.body_position_errors[t] = np.nan
        self.cost[0, t] = self.latest_plan_cost
        self.recorded_steps = max(self.recorded_steps, t+1)
        return None
    
    def state_difference(self, pos1, pos2):
        # computes the finite difference between two states
        vel = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(self.model, vel, self.model.opt.timestep, pos1, pos2)
        return vel
    
    def capture_frame(self):
        # Capture the current frame from the viewer
        width, height = self.viewer.viewport.width, self.viewer.viewport.height
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Render the frame and store it in the array
        mujoco.mjr_readPixels(frame, None, self.viewer.viewport, self.viewer.ctx)

        # Save the frame as a PNG image
        filename = os.path.join(self.save_dir, f"frame_{self.t}.png")
        image = Image.fromarray(np.flipud(frame))
        image.save(filename)

    def capture_frame_traj(self):
        # Capture the current frame from the viewer
        width, height = self.viewer.viewport.width, self.viewer.viewport.height
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Render the frame and store it in the array
        mujoco.mjr_readPixels(frame, None, self.viewer.viewport, self.viewer.ctx)

        # Save the frame as a PNG image
        filename = os.path.join(self.save_dir, f"frame_{self.t}_{self.n}_{self.i}.png")
        image = Image.fromarray(np.flipud(frame))
        image.save(filename)

    @staticmethod
    def plan_command(plan, age, timestep):
        index = max(0, int(np.floor((age + 1e-10) / timestep)))
        return plan[min(index, len(plan)-1)].copy(), index >= len(plan)

    def _advance_warm_start(self, elapsed):
        # commit_plan already shifts by one prediction step.
        steps = max(0, int(round(elapsed / self.agent.h))-1)
        if steps:
            indices = np.minimum(np.arange(self.agent.horizon)+steps, self.agent.horizon-1)
            self.agent.trajectory = self.agent.trajectory[indices].copy()
            if not self.velocity_task:
                for _ in range(steps):
                    self.agent.gait_scheduler.roll()

    def run(self):
        run_started = wall_time.perf_counter()
        tqdm_range = tqdm.tqdm(range(self.T-1))
        action = self.data.ctrl.copy()
        if self.sync:
            # Position actuators start by holding the measured joint positions.
            action = self.data.qpos[self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]]].copy()
        active_plan = None
        pending = None
        active_origin = None
        previous_solve_time = None
        if self.agent is not None:
            self.agent.external_reference_updates = self.sync
        simulation_start = self.data.time
        for t in tqdm_range:
            self.t = t
            if self.velocity_task:
                self.agent.set_command_time(self.data.time-simulation_start)
            if self.sync and not self.velocity_task and hasattr(self.agent, 'advance_ee_waypoint'):
                position, velocity = self.agent.ee_position_velocity(self.data.qpos, self.data.qvel)
                self.agent.advance_ee_waypoint(position, velocity, dt=self.model.opt.timestep)
            self.store_trajectory(t)

            if self.agent is not None:
                now = self.data.time
                if self.sync and pending is not None and t >= pending['ready_step']:
                    active_plan, active_origin = pending['plan'], pending['origin']
                    self.latest_plan_cost = pending['cost']
                    self.plan_application_delay_seconds[t] = now-active_origin
                    pending = None
                if t % self.update_ratio == 0 and (not self.sync or pending is None):
                    if self.sync:
                        if previous_solve_time is not None:
                            self._advance_warm_start(now-previous_solve_time)
                        if hasattr(self.agent, 'previous_action'):
                            self.agent.previous_action = action[:self.agent.act_dim].copy()
                    solve_started = wall_time.perf_counter()
                    candidate = self.agent.update(np.concatenate([self.data.qpos, self.data.qvel], axis=0))
                    self.solve_seconds[t] = wall_time.perf_counter()-solve_started
                    cost_started = wall_time.perf_counter()
                    plan_cost = float(self.agent.eval_best_trajectory())
                    self.cost_seconds[t] = wall_time.perf_counter()-cost_started
                    if self.sync:
                        plan = self.agent.selected_trajectory.copy()
                        if plan.shape[1] != self.model.nu:
                            plan = self.agent.rollout_func.expand_actions(plan)
                        # Diagnostic evaluation blocks this serial worker too.
                        duration = self.solve_seconds[t]+self.cost_seconds[t]
                        delay_steps = max(1, int(np.ceil(duration/self.model.opt.timestep)))
                        pending = dict(plan=plan, origin=now, ready_step=t+delay_steps, cost=plan_cost)
                        previous_solve_time = now
                    else:
                        action = candidate
                        self.latest_plan_cost = plan_cost
                    self.store_trajectory(t)
                if self.sync and active_plan is not None:
                    age = now-active_origin
                    action, exhausted = self.plan_command(active_plan, age, self.agent.h)
                    self.plan_age_seconds[t] = age
                    self.plan_exhausted[t] = exhausted
                self.data.ctrl = action
            self.ctrl[:, t] = action

            mujoco.mj_step(self.model, self.data)
            
            error = np.linalg.norm(self.agent.body_ref[:3]-self.data.qpos[:3]) if self.agent is not None else np.inf
            if self.agent is not None and not self.velocity_task and t % self.update_ratio == 0 and error < self.agent.goal_thresh[self.agent.goal_index]:
                self.agent.next_goal()

            if self.viewer is not None and not self.viewer.is_alive:
                break
            if self.viewer is not None and t % self.update_ratio == 0:
                self.sync_render_state()
                if self.ee_errors is not None:
                    self.viewer.add_marker(pos=self.agent.ee_objective.target_world(self.data.qpos),
                                           size=[0.025]*3, rgba=[0, 1, 0, 0.8],
                                           type=mujoco.mjtGeom.mjGEOM_SPHERE, label='EE target')
                if not self.velocity_task:
                    self.viewer.add_marker(pos=self.agent.body_ref[:3].copy(), size=[.15]*3,
                                           rgba=[1, 0, 1, 1], type=mujoco.mjtGeom.mjGEOM_SPHERE, label='')

                self.viewer.render()
                if self.save_frames:
                    self.capture_frame()
            else:
                pass

            if self.sync:
                deadline = run_started + self.data.time-simulation_start
                remaining = deadline-wall_time.perf_counter()
                if remaining > 0:
                    wall_time.sleep(remaining)
                self.sync_lag_seconds[t] = max(0., wall_time.perf_counter()-deadline)

        # store last state
        if self.velocity_task:
            self.agent.set_command_time(self.data.time-simulation_start)
        self.store_trajectory(t+1 if self.T > 1 else 0)
        self.run_wall_seconds = wall_time.perf_counter()-run_started
        if self.agent is not None:
            self.agent.external_reference_updates = False
        
        if self.viewer is not None:
            self.viewer.close()
        return None

    def save_results(self, controller='configured', output_root=None):
        from whole_body_mppi.interface.results import save_results
        self.output_dir = save_results(self, controller, output_root)
        return self.output_dir

    def plot_trajectory(self):
        """Display the already saved report without writing legacy analysis files."""
        if self.output_dir is None:
            self.save_results(getattr(self.agent, 'mppi_mode', None) or 'configured')
        plt.figure()
        plt.imshow(plt.imread(self.output_dir / 'metrics.png'))
        plt.axis('off')
        plt.show()

    def playback(self):
        """Replay recorded states at simulation speed; never solve or step physics."""
        if self.recorded_steps == 0:
            raise ValueError('Run the simulation before playback')
        import glfw
        if not glfw.init():
            raise RuntimeError('Cannot open playback display. Results were saved; run --playback from a graphical desktop session.')
        self.initialize_rendering()
        import mujoco_viewer
        self.viewer = mujoco_viewer.MujocoViewer(self.render_model, self.render_data, hide_menus=True)
        saved = (self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy(), self.data.time)
        start = wall_time.perf_counter()
        times = self.time[:self.recorded_steps]
        try:
            while self.viewer.is_alive:
                elapsed = wall_time.perf_counter()-start
                index = min(int(np.searchsorted(times-times[0], elapsed, side='right')-1), len(times)-1)
                index = max(0, index)
                self.data.qpos[:] = self.qpos[:, index]
                self.data.qvel[:] = self.qvel[:, index]
                self.data.ctrl[:] = self.ctrl[:, index]
                self.data.time = times[index]
                mujoco.mj_forward(self.model, self.data)
                self.sync_render_state()
                if self.ee_world_targets is not None:
                    self.viewer.add_marker(pos=self.ee_world_targets[:, index], size=[.025]*3,
                                           rgba=[0, 1, 0, .8], type=mujoco.mjtGeom.mjGEOM_SPHERE, label='EE target')
                self.viewer.render()
                if index == len(times)-1:
                    break
                wall_time.sleep(1/60)
        finally:
            self.viewer.close()
            self.viewer = None
            self.data.qpos[:], self.data.qvel[:], self.data.ctrl[:], self.data.time = saved
            mujoco.mj_forward(self.model, self.data)
            self.sync_render_state()

    def get_state(self):
        return np.concatenate([self.data.qpos, self.data.qvel], axis=0)

def apply_translation_to_com(com_world, quaternion, translation_local):
    """
    Apply a translation in the robot's local frame to the CoM in the world frame.
    
    Parameters:
    - com_world: 3D array, [x, y, z] coordinates of the CoM in the world frame
    - quaternion: 4D array, [q_w, q_x, q_y, q_z] quaternion representing the robot's orientation in the world
    - translation_local: 3D array, [x, y, z] translation in the robot's local frame (e.g., [0.3, 0, 0] for 30 cm along x-axis)
    
    Returns:
    - new_com_world: 3D array, new [x, y, z] coordinates of the CoM in the world frame
    """
    # Convert quaternion to rotation matrix
    rotation = R.from_quat(quaternion[[1, 2, 3, 0]])
    
    # Transform the local translation to the world frame
    translation_world = rotation.apply(translation_local)
    
    # Apply the translation in the world frame to the CoM
    new_com_world = np.array(com_world) + translation_world
    
    return new_com_world
   
if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__), "../models/go1/task_simulate.xml")

    simulator = Simulator(T = 300, dt=0.002, viewer=True, gravity=True, model_path=model_path)
    simulator.run()
    simulator.plot_trajectory()



