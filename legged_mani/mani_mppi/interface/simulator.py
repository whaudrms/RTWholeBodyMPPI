import mujoco
import os
import mujoco_viewer
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import tqdm
from PIL import Image


def synchronize_contact_options(target_model, source_model):
    """Match simulator contact options to the controller rollout model.

    The simulator and MPPI load separate MuJoCo models.  Contact cone and
    override settings therefore need to be copied explicitly; otherwise MPPI
    predicts pyramidal/overridden contacts while the simulator executes the
    XML defaults (elliptic contacts without the override).
    """
    target_model.opt.cone = source_model.opt.cone

    override_bit = int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)
    target_model.opt.enableflags = (
        (int(target_model.opt.enableflags) & ~override_bit)
        | (int(source_model.opt.enableflags) & override_bit)
    )
    target_model.opt.o_solref[:] = source_model.opt.o_solref
    target_model.opt.o_solimp[:] = source_model.opt.o_solimp
    target_model.opt.o_friction[:] = source_model.opt.o_friction
    target_model.opt.o_margin = source_model.opt.o_margin


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
                 model_path = os.path.join(os.path.dirname(__file__), "../models/b2_z1_4dof.xml"),
                T = 200, dt = 0.01, viewer = True, gravity = True,
                # stiff=False
                timeconst=0.02, dampingratio=1.0, ctrl_rate=100,
                save_dir="./frames", save_frames=False
                ):
        # filter

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.agent = agent
        self.ctrl_rate = ctrl_rate
        self.update_ratio = max(1, 1/(dt*ctrl_rate))
        self.interpolate_cam = False
        # model
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.model.opt.timestep = dt
        if agent is not None and hasattr(agent, "model"):
            # MPPI and the closed-loop simulator use separate model instances.
            # Keep their contact dynamics identical so rollout costs describe
            # the contacts that will actually be executed by mj_step below.
            synchronize_contact_options(self.model, agent.model)
        else:
            # Preserve the standalone Simulator API when no controller model
            # is available as the source of truth.
            self.model.opt.o_solref = np.array([timeconst, dampingratio])
        # data
        self.data = mujoco.MjData(self.model)
        self.T = T
        # save
        self.save_frames = save_frames
        self.save_dir = save_dir
        # rollout
        keyframe_name = "stand"
        if agent is not None:
            keyframe_name = agent.params.get("keyframe", keyframe_name)
        keyframe_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name
        )
        if keyframe_id < 0:
            raise ValueError(f"Unknown simulator keyframe: {keyframe_name}")
        mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe_id)
        self.data.ctrl = self.model.key_ctrl[keyframe_id]
        mujoco.mj_forward(self.model, self.data)

        # viewer
        if viewer:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, hide_menus=True)
        else:
            self.viewer = None

        # trajectories
        self.qpos = np.zeros((self.model.nq, self.T))
        self.qvel = np.zeros((self.model.nv, self.T))
        self.ctrl = np.zeros((self.model.nu, self.T))
        self.time = np.zeros(self.T)
        self.cost = np.zeros((1, self.T))

        # Ensure the directory exists
        if self.save_frames and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def step(self, ctrl=None):
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)
            return self.data.qpos, self.data.qvel

    def store_trajectory(self, t):
        self.qpos[:, t] = self.data.qpos
        self.qvel[:, t] = self.data.qvel
        self.ctrl[:, t] = self.data.ctrl
        self.time[t] = self.data.time
        self.cost[0, t] = self.agent.eval_best_trajectory()
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

    def run(self):
        tqdm_range = tqdm.tqdm(range(self.T-1))
        for t in tqdm_range:
            self.t = t
            self.store_trajectory(t)
            self.ctrl[:, t] = self.data.ctrl

            if self.agent is not None:
                if t % self.update_ratio == 0:
                    action = self.agent.update(np.concatenate([self.data.qpos, self.data.qvel], axis=0))
                self.data.ctrl = action

            mujoco.mj_step(self.model, self.data)
            
            observation = np.concatenate([self.data.qpos, self.data.qvel], axis=0)
            if hasattr(self.agent, "goal_reached"):
                reached = self.agent.goal_reached(observation)
            else:
                error = np.linalg.norm(
                    np.asarray(self.agent.body_ref[:3]) - np.asarray(self.data.qpos[:3])
                )
                reached = error < self.agent.goal_thresh[self.agent.goal_index]
            if reached:
                self.agent.next_goal()

            if self.viewer is not None and self.viewer.is_alive:
                self.viewer.add_marker(
                    pos=self.agent.body_ref[:3]*1,         # Position of the marker
                    size=[0.15, 0.15, 0.15],     # Size of the sphere
                    rgba=[1, 0, 1, 1],           # Color of the sphere (red)
                    type=mujoco.mjtGeom.mjGEOM_SPHERE, # Specify that this is a sphere
                    label=""
                )

                # Show the current manipulator end-effector goal for locomani.
                if hasattr(self.agent, "ee_goal_pos"):
                    ee_goal = self.agent.ee_goal_pos[self.agent.goal_index]
                    self.viewer.add_marker(
                        pos=ee_goal,
                        size=[0.05, 0.05, 0.05],
                        rgba=[1, 0.2, 0.1, 1],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    )
                            
                self.viewer.render()
                if self.save_frames:
                    self.capture_frame()
            else:
                pass
        
        # store last state
        self.store_trajectory(self.T-1)
        
        if self.viewer is not None:
            self.viewer.close()
        return None

    def plot_trajectory(self):

        base_folfer = os.path.join(self.base_dir, "../analysis/")
        os.makedirs(base_folfer, exist_ok=True)

        trajectory_name = 'mujoco_traj_param_rate_{}_h_{}_lam_{}_n_{}_T_{}_task_{}.tsv'.format(self.ctrl_rate, \
                                                                                                 self.agent.horizon, \
                                                                                                 self.agent.temperature, \
                                                                                                 self.agent.n_samples, \
                                                                                                 self.T,\
                                                                                                 self.agent.task)
        analysis_path = os.path.join(base_folfer, trajectory_name)
                                                                                                                                               
        np.savetxt(analysis_path, self.cost, delimiter='\t')
        # position
        plt.figure()
        
        plt.plot(self.time, self.qpos[0, :], label="x (sim)", ls="--", color="blue")
        plt.plot(self.time, self.qpos[1, :], label="y (sim)", ls="--", color="orange")
        plt.plot(self.time, self.qpos[2, :], label="z (sim)", ls="--", color="magenta")

        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")

        # orientation plot
        fig = plt.figure()

        plt.plot(self.time, self.qpos[3, :], label="q0 (sim)", ls="--", color="blue")
        plt.plot(self.time, self.qpos[4, :], label="q1 (sim)", ls="--", color="orange")
        plt.plot(self.time, self.qpos[5, :], label="q2 (sim)", ls="--", color="magenta")
        plt.plot(self.time, self.qpos[6, :], label="q3 (sim)", ls="--", color="green")

        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Orientation")

        # plot controls
        fig = plt.figure()
        plt.plot(self.time[:], self.ctrl[0, :], label="FR_thigh", color="blue")
        plt.plot(self.time[:], self.ctrl[1, :], label="FR_hip", color="orange")
        plt.plot(self.time[:], self.ctrl[2, :], label="FR_knee", color="magenta")
        plt.plot(self.time[:], self.ctrl[3, :], label="FL_thigh", color="green")
        plt.plot(self.time[:], self.ctrl[4, :], label="FL_hip", color="red")
        plt.plot(self.time[:], self.ctrl[5, :], label="FL_knee", color="purple")
        plt.plot(self.time[:], self.ctrl[6, :], label="RR_thigh", color="black")
        plt.plot(self.time[:], self.ctrl[7, :], label="RR_hip", color="blue")
        plt.plot(self.time[:], self.ctrl[8, :], label="RR_knee", color="orange")
        plt.plot(self.time[:], self.ctrl[9, :], label="RL_thigh", color="magenta")
        plt.plot(self.time[:], self.ctrl[10, :], label="RL_hip", color="green")
        plt.plot(self.time[:], self.ctrl[11, :], label="RL_knee", color="red")

        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Control (angles)")
        plt.show()
        
        return None

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
    model_path = os.path.join(os.path.dirname(__file__), "../models/b2_z1_4dof.xml")

    simulator = Simulator(T = 300, dt=0.002, viewer=True, gravity=True, model_path=model_path)
    simulator.run()
    simulator.plot_trajectory()
