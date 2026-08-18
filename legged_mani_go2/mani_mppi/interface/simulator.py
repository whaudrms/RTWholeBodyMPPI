import mujoco
import os
import time
import mujoco_viewer
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
    target_model.opt.integrator = source_model.opt.integrator

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
                 model_path = os.path.join(os.path.dirname(__file__), "../models/scene.xml"),
                T = 200, dt = 0.01, viewer = True, gravity = True,
                # stiff=False
                timeconst=0.02, dampingratio=1.0, ctrl_rate=25,
                save_dir="./frames", save_frames=False, render_every=1,
                plot_enabled=False,
                ):
        # filter

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.agent = agent
        self.ctrl_rate = ctrl_rate
        self.update_ratio = max(1, 1/(dt*ctrl_rate))
        self.plot_enabled = bool(plot_enabled)
        self.render_every = int(render_every)
        if self.render_every < 1:
            raise ValueError("render_every must be at least 1")
        self.interpolate_cam = False
        # model
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.model.opt.timestep = dt
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
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
        self.cost = np.full((1, self.T), np.nan)
        self.body_ref_history = np.full((7, self.T), np.nan)
        self.ee_actual_history = np.full((3, self.T), np.nan)
        self.ee_ref_history = np.full((3, self.T), np.nan)
        self.ik_residual_history = np.full(self.T, np.nan)
        self.goal_index_history = np.full(self.T, -1, dtype=int)
        self.update_rate_hz = np.full(self.T, np.nan)
        self.task_completion_time = np.nan

        self.ee_site_id = -1
        if agent is not None and hasattr(agent, "ee_site_name"):
            self.ee_site_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_SITE,
                agent.ee_site_name,
            )

        self.arm_qpos_indices = np.empty(0, dtype=int)
        arm_dim = 0
        if agent is not None and hasattr(agent, "arm_qpos_indices"):
            self.arm_qpos_indices = np.asarray(
                agent.arm_qpos_indices, dtype=int
            )
            arm_dim = len(self.arm_qpos_indices)
        self.arm_actual_history = np.full((arm_dim, self.T), np.nan)
        self.arm_ref_history = np.full((arm_dim, self.T), np.nan)

        self.box_qpos_adr = None
        if agent is not None and hasattr(agent, "box_qpos_adr"):
            self.box_qpos_adr = int(agent.box_qpos_adr)
        self.box_ref_history = np.full((3, self.T), np.nan)

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
        self._store_agent_diagnostics(t)
        return None

    def _store_agent_diagnostics(self, t):
        """Store references and metrics aligned with one controller update."""
        best_cost = self.agent.eval_best_trajectory()
        self.cost[0, t] = (
            np.nan if best_cost is None else float(best_cost)
        )

        if hasattr(self.agent, "body_ref"):
            body_ref = np.asarray(self.agent.body_ref, dtype=float)
            self.body_ref_history[:, t] = body_ref[:7]
        if hasattr(self.agent, "goal_index"):
            self.goal_index_history[t] = int(self.agent.goal_index)
        if self.ee_site_id >= 0:
            self.ee_actual_history[:, t] = self.data.site_xpos[
                self.ee_site_id
            ]
        if hasattr(self.agent, "ee_goal_pos"):
            goals = np.asarray(self.agent.ee_goal_pos, dtype=float)
            goal_index = int(getattr(self.agent, "goal_index", 0))
            if 0 <= goal_index < len(goals):
                self.ee_ref_history[:, t] = goals[goal_index]
        if hasattr(self.agent, "last_ik_residual"):
            self.ik_residual_history[t] = float(
                self.agent.last_ik_residual
            )
        if len(self.arm_qpos_indices):
            self.arm_actual_history[:, t] = self.data.qpos[
                self.arm_qpos_indices
            ]
            if hasattr(self.agent, "arm_reference"):
                self.arm_ref_history[:, t] = np.asarray(
                    self.agent.arm_reference, dtype=float
                )
        if hasattr(self.agent, "x_box_ref"):
            self.box_ref_history[:, t] = np.asarray(
                self.agent.x_box_ref, dtype=float
            )[:3]

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
            if self.plot_enabled:
                self.store_trajectory(t)

            if self.agent is not None:
                if t % self.update_ratio == 0:
                    observation = np.concatenate(
                        [self.data.qpos, self.data.qvel], axis=0
                    )
                    if self.plot_enabled:
                        update_start = time.perf_counter()
                        action = self.agent.update(observation)
                        update_seconds = time.perf_counter() - update_start
                        if update_seconds > 0.0:
                            self.update_rate_hz[t] = 1.0 / update_seconds
                        # ``store_trajectory`` runs before the controller
                        # update so qpos/qvel match the current simulation
                        # time. Refresh controller-owned diagnostics afterward.
                        self._store_agent_diagnostics(t)
                    else:
                        action = self.agent.update(observation)
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
                if (
                    self.plot_enabled
                    and not np.isfinite(self.task_completion_time)
                    and bool(getattr(self.agent, "task_success", False))
                ):
                    self.task_completion_time = float(self.data.time)

            if (
                self.viewer is not None
                and self.viewer.is_alive
                and t % self.render_every == 0
            ):
                self.viewer.add_marker(
                    pos=self.agent.body_ref[:3]*1,         # Position of the marker
                    size=[0.15, 0.15, 0.15],     # Size of the sphere
                    rgba=[1, 0, 1, 1],           # Magenta body-reference marker
                    type=mujoco.mjtGeom.mjGEOM_SPHERE, # Specify that this is a sphere
                    label=""
                )

                # The magenta marker above is the rate-limited command. Show
                # the raw CEM base decision separately so planning latency is
                # not confused with deliberate motion interpolation.
                if (
                    hasattr(self.agent, "planned_base_xy")
                    and hasattr(self.agent, "planned_base_height")
                ):
                    planned_base_pos = np.array(
                        [
                            self.agent.planned_base_xy[0],
                            self.agent.planned_base_xy[1],
                            self.agent.planned_base_height,
                        ],
                        dtype=float,
                    )
                    self.viewer.add_marker(
                        pos=planned_base_pos,
                        size=[0.07, 0.07, 0.07],
                        rgba=[0.0, 1.0, 1.0, 1.0],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        label="CEM base target",
                    )

                # Show the current manipulator end-effector goal.
                if hasattr(self.agent, "ee_goal_pos"):
                    ee_goal = self.agent.ee_goal_pos[self.agent.goal_index]
                    self.viewer.add_marker(
                        pos=ee_goal,
                        size=[0.05, 0.05, 0.05],
                        rgba=[1, 0.2, 0.1, 1],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    )

                # Show the desired box position for push-box tasks.
                if hasattr(self.agent, "x_box_ref"):
                    self.viewer.add_marker(
                        pos=self.agent.x_box_ref[:3],
                        size=[0.08, 0.08, 0.08],
                        rgba=[1.0, 1.0, 0.0, 1.0],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    )
                            
                self.viewer.render()
                if self.save_frames:
                    self.capture_frame()
            else:
                pass
        
        # store last state
        if self.plot_enabled:
            self.store_trajectory(self.T-1)
        
        if self.viewer is not None:
            self.viewer.close()
        return None

    @staticmethod
    def _quaternion_history_to_rpy(quaternions):
        """Convert a (4, T) MuJoCo wxyz history to roll/pitch/yaw."""
        quaternions = np.asarray(quaternions, dtype=float)
        rpy = np.full((3, quaternions.shape[1]), np.nan)
        norms = np.linalg.norm(quaternions, axis=0)
        valid = np.isfinite(quaternions).all(axis=0) & (norms > 1e-12)
        if np.any(valid):
            normalized = quaternions[:, valid] / norms[valid]
            xyzw = normalized[[1, 2, 3, 0], :].T
            rpy[:, valid] = R.from_quat(xyzw).as_euler("xyz").T
        return rpy

    @staticmethod
    def _pyplot():
        """Import matplotlib only when plot recording was requested."""
        import matplotlib.pyplot as pyplot

        return pyplot

    def _plot_xyz(self, axis, actual, reference, title, *, step_ref=False):
        colors = ("tab:red", "tab:green", "tab:blue")
        labels = ("x", "y", "z")
        for index, (label, color) in enumerate(zip(labels, colors)):
            axis.plot(
                self.time,
                actual[index],
                color=color,
                label=f"{label} actual",
            )
            axis.plot(
                self.time,
                reference[index],
                color=color,
                linestyle="--",
                drawstyle="steps-post" if step_ref else "default",
                label=f"{label} ref",
            )
        axis.set_title(title)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Position (m)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=3)
        self._mark_task_completion(axis)

    def _plot_rpy(self, axis, actual_quat, reference_quat, title):
        actual = self._quaternion_history_to_rpy(actual_quat)
        reference = self._quaternion_history_to_rpy(reference_quat)
        colors = ("tab:red", "tab:green", "tab:blue")
        labels = ("roll", "pitch", "yaw")
        for index, (label, color) in enumerate(zip(labels, colors)):
            axis.plot(
                self.time,
                actual[index],
                color=color,
                label=f"{label} actual",
            )
            axis.plot(
                self.time,
                reference[index],
                color=color,
                linestyle="--",
                label=f"{label} ref",
            )
        axis.set_title(title)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Angle (rad)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=3)
        self._mark_task_completion(axis)

    def _plot_update_performance(self, axis):
        valid = np.isfinite(self.update_rate_hz)
        axis.plot(
            self.time[valid],
            self.update_rate_hz[valid],
            color="tab:orange",
            marker=".",
            markersize=2,
            label="update rate",
        )
        if np.any(valid):
            mean_rate = float(np.mean(self.update_rate_hz[valid]))
            axis.axhline(
                mean_rate,
                color="black",
                linestyle="--",
                linewidth=1.2,
                label=f"mean = {mean_rate:.1f} Hz",
            )
        axis.set_title("Controller update rate")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Update rate (Hz)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="best")
        self._mark_task_completion(axis)

    def _plot_cost(self, axis):
        valid = np.isfinite(self.cost[0])
        axis.plot(
            self.time[valid],
            self.cost[0, valid],
            color="tab:brown",
        )
        axis.set_title("Best MPPI rollout cost")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Cost")
        axis.set_yscale("symlog", linthresh=1.0)
        axis.grid(alpha=0.25)
        self._mark_task_completion(axis)

    def _mark_task_completion(self, axis):
        """Mark the first time at which the controller reports success."""
        if not np.isfinite(self.task_completion_time):
            return
        axis.axvline(
            self.task_completion_time,
            color="tab:gray",
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
        )
        axis.annotate(
            f"task complete\n{self.task_completion_time:.2f} s",
            xy=(self.task_completion_time, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, -3),
            textcoords="offset points",
            rotation=90,
            va="top",
            fontsize=7,
            color="tab:gray",
        )

    def _mark_goal_changes(self, axis):
        changes = np.flatnonzero(
            self.goal_index_history[1:] != self.goal_index_history[:-1]
        ) + 1
        for index in changes:
            goal_index = self.goal_index_history[index]
            if goal_index < 0:
                continue
            axis.axvline(
                self.time[index],
                color="black",
                linestyle=":",
                linewidth=1.0,
                alpha=0.7,
            )
            axis.annotate(
                f"goal {goal_index}",
                xy=(self.time[index], 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(3, -3),
                textcoords="offset points",
                rotation=90,
                va="top",
                fontsize=7,
            )

    def _plot_locomotion_dashboard(self):
        figure, axes = self._pyplot().subplots(
            2, 2, figsize=(14, 9), constrained_layout=True
        )
        self._plot_xyz(
            axes[0, 0],
            self.qpos[:3],
            self.body_ref_history[:3],
            "Body position",
        )
        self._plot_rpy(
            axes[0, 1],
            self.qpos[3:7],
            self.body_ref_history[3:7],
            "Body orientation",
        )
        self._plot_update_performance(axes[1, 0])
        self._plot_cost(axes[1, 1])
        return figure

    def _plot_big_box_dashboard(self):
        figure, axes = self._pyplot().subplots(
            3, 2, figsize=(14, 12), constrained_layout=True
        )
        self._plot_xyz(
            axes[0, 0],
            self.qpos[:3],
            self.body_ref_history[:3],
            "Body position",
        )
        self._plot_rpy(
            axes[0, 1],
            self.qpos[3:7],
            self.body_ref_history[3:7],
            "Body orientation",
        )
        axes[1, 0].plot(
            self.qpos[0],
            self.qpos[2],
            color="tab:blue",
            label="body actual",
        )
        axes[1, 0].plot(
            self.body_ref_history[0],
            self.body_ref_history[2],
            color="tab:orange",
            linestyle="--",
            label="body ref",
        )
        if hasattr(self.agent, "goal_pos"):
            goals = np.asarray(self.agent.goal_pos, dtype=float)
            axes[1, 0].plot(
                goals[:, 0],
                goals[:, 2],
                "ko:",
                label="waypoints",
            )
        axes[1, 0].set_title("Big-box x-z trajectory")
        axes[1, 0].set_xlabel("x (m)")
        axes[1, 0].set_ylabel("z (m)")
        axes[1, 0].grid(alpha=0.25)
        axes[1, 0].legend(fontsize=8)
        self._plot_update_performance(axes[1, 1])
        self._plot_cost(axes[2, 0])
        axes[2, 1].axis("off")
        return figure

    def _plot_locomani_dashboard(self):
        figure, axes = self._pyplot().subplots(
            4, 2, figsize=(15, 16), constrained_layout=True
        )
        self._plot_xyz(
            axes[0, 0],
            self.qpos[:3],
            self.body_ref_history[:3],
            "Body position",
        )
        self._plot_rpy(
            axes[0, 1],
            self.qpos[3:7],
            self.body_ref_history[3:7],
            "Body orientation",
        )
        self._plot_xyz(
            axes[1, 0],
            self.ee_actual_history,
            self.ee_ref_history,
            "End-effector position",
            step_ref=True,
        )
        self._mark_goal_changes(axes[1, 0])

        ee_error = np.linalg.norm(
            self.ee_actual_history - self.ee_ref_history, axis=0
        )
        axes[1, 1].plot(
            self.time,
            ee_error,
            color="tab:blue",
            label="EE position error",
        )
        axes[1, 1].set_title("EE error and IK residual")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("EE error (m)", color="tab:blue")
        axes[1, 1].tick_params(axis="y", labelcolor="tab:blue")
        axes[1, 1].grid(alpha=0.25)
        ik_axis = axes[1, 1].twinx()
        ik_axis.plot(
            self.time,
            self.ik_residual_history,
            color="tab:red",
            label="IK residual",
        )
        ik_axis.set_ylabel("IK residual (m)", color="tab:red")
        ik_axis.tick_params(axis="y", labelcolor="tab:red")
        lines = axes[1, 1].lines + ik_axis.lines
        axes[1, 1].legend(
            lines,
            [line.get_label() for line in lines],
            fontsize=8,
            loc="best",
        )
        self._mark_goal_changes(axes[1, 1])
        self._mark_task_completion(axes[1, 1])

        arm_colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
        for joint_index in range(len(self.arm_qpos_indices)):
            color = arm_colors[joint_index % len(arm_colors)]
            axes[2, 0].plot(
                self.time,
                self.arm_actual_history[joint_index],
                color=color,
                label=f"joint {joint_index + 1} actual",
            )
            axes[2, 0].plot(
                self.time,
                self.arm_ref_history[joint_index],
                color=color,
                linestyle="--",
                label=f"joint {joint_index + 1} IK ref",
            )
        axes[2, 0].set_title("Arm joints: actual vs IK reference")
        axes[2, 0].set_xlabel("Time (s)")
        axes[2, 0].set_ylabel("Joint angle (rad)")
        axes[2, 0].grid(alpha=0.25)
        axes[2, 0].legend(fontsize=7, ncol=2)
        self._mark_task_completion(axes[2, 0])
        self._plot_update_performance(axes[2, 1])
        self._plot_cost(axes[3, 0])
        axes[3, 1].axis("off")
        return figure

    def _plot_push_box_dashboard(self):
        figure, axes = self._pyplot().subplots(
            4, 2, figsize=(15, 16), constrained_layout=True
        )
        self._plot_xyz(
            axes[0, 0],
            self.qpos[:3],
            self.body_ref_history[:3],
            "Body position",
        )
        self._plot_rpy(
            axes[0, 1],
            self.qpos[3:7],
            self.body_ref_history[3:7],
            "Body orientation",
        )
        self._plot_xyz(
            axes[1, 0],
            self.ee_actual_history,
            self.ee_ref_history,
            "End-effector position",
        )

        box_position = self.qpos[
            self.box_qpos_adr:self.box_qpos_adr + 3
        ]
        axes[1, 1].plot(
            self.qpos[0],
            self.qpos[1],
            color="tab:blue",
            label="body actual",
        )
        axes[1, 1].plot(
            self.body_ref_history[0],
            self.body_ref_history[1],
            color="tab:blue",
            linestyle="--",
            label="body ref",
        )
        axes[1, 1].plot(
            box_position[0],
            box_position[1],
            color="tab:green",
            label="box",
        )
        axes[1, 1].plot(
            self.ee_actual_history[0],
            self.ee_actual_history[1],
            color="tab:red",
            alpha=0.8,
            label="EE",
        )
        axes[1, 1].plot(
            self.ee_ref_history[0],
            self.ee_ref_history[1],
            color="tab:red",
            linestyle="--",
            label="EE ref",
        )
        valid_box_ref = np.isfinite(self.box_ref_history).all(axis=0)
        if np.any(valid_box_ref):
            target = self.box_ref_history[:, valid_box_ref][:, -1]
            axes[1, 1].scatter(
                target[0],
                target[1],
                marker="*",
                s=120,
                color="gold",
                edgecolor="black",
                label="box goal",
                zorder=5,
            )
            if hasattr(self.agent, "box_goal_tolerance"):
                from matplotlib.patches import Circle

                tolerance = float(self.agent.box_goal_tolerance)
                axes[1, 1].add_patch(
                    Circle(
                        (target[0], target[1]),
                        tolerance,
                        facecolor="gold",
                        edgecolor="goldenrod",
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.18,
                        label=f"goal tolerance ({tolerance:.2f} m)",
                        zorder=1,
                    )
                )
        axes[1, 1].set_title("Push-box x-y trajectory")
        axes[1, 1].set_xlabel("x (m)")
        axes[1, 1].set_ylabel("y (m)")
        axes[1, 1].axis("equal")
        axes[1, 1].grid(alpha=0.25)
        axes[1, 1].legend(fontsize=7, ncol=2)

        box_error = np.linalg.norm(
            box_position - self.box_ref_history, axis=0
        )
        axes[2, 0].plot(
            self.time,
            box_error,
            color="tab:brown",
            label="box position error",
        )
        axes[2, 0].set_title("Box position error norm")
        axes[2, 0].set_xlabel("Time (s)")
        axes[2, 0].set_ylabel("Error (m)")
        axes[2, 0].grid(alpha=0.25)
        axes[2, 0].legend(fontsize=8)
        self._mark_task_completion(axes[2, 0])

        box_quaternion = self.qpos[
            self.box_qpos_adr + 3:self.box_qpos_adr + 7
        ]
        box_rpy = self._quaternion_history_to_rpy(box_quaternion)
        axes[2, 1].plot(
            self.time,
            box_rpy[0],
            color="tab:red",
            label="roll actual",
        )
        axes[2, 1].plot(
            self.time,
            np.zeros(self.T),
            color="tab:red",
            linestyle="--",
            label="roll ref",
        )
        axes[2, 1].plot(
            self.time,
            box_rpy[1],
            color="tab:green",
            label="pitch actual",
        )
        axes[2, 1].plot(
            self.time,
            np.zeros(self.T),
            color="tab:green",
            linestyle="--",
            label="pitch ref",
        )
        if hasattr(self.agent, "box_max_tilt"):
            max_tilt = float(self.agent.box_max_tilt)
            tilt_limit_label = (
                f"tilt limit = +/-{max_tilt:.3f} rad "
                f"({np.degrees(max_tilt):.1f} deg)"
            )
            axes[2, 1].axhline(
                max_tilt,
                color="black",
                linestyle="--",
                linewidth=1.2,
                label=tilt_limit_label,
            )
            axes[2, 1].axhline(
                -max_tilt,
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="_nolegend_",
            )
        axes[2, 1].set_title("Box roll/pitch: actual vs upright reference")
        axes[2, 1].set_xlabel("Time (s)")
        axes[2, 1].set_ylabel("Angle (rad)")
        axes[2, 1].grid(alpha=0.25)
        axes[2, 1].legend(fontsize=7, ncol=2)
        self._mark_task_completion(axes[2, 1])
        self._plot_update_performance(axes[3, 0])
        self._plot_cost(axes[3, 1])
        return figure

    def plot_trajectory(self):
        """Save one task-specific dashboard and one compact cost log."""
        if not self.plot_enabled:
            raise RuntimeError(
                "Plot recording is disabled. Construct Simulator with "
                "plot_enabled=True or run simulate_mppi.py with --plot."
            )
        task = str(self.agent.task)
        task_directory = os.path.abspath(
            os.path.join(self.base_dir, "../analysis/tasks", task)
        )
        os.makedirs(task_directory, exist_ok=True)
        run_name = (
            f"rate_{self.ctrl_rate}_h_{self.agent.horizon}_"
            f"lam_{self.agent.temperature}_n_{self.agent.n_samples}_"
            f"T_{self.T}"
        )
        cost_path = os.path.join(
            task_directory, f"{run_name}_cost.tsv"
        )
        np.savetxt(
            cost_path,
            np.column_stack((self.time, self.cost[0])),
            delimiter="\t",
            header="time_s\tbest_cost",
            comments="",
        )

        if task == "locomani":
            figure = self._plot_locomani_dashboard()
        elif task == "push_box":
            figure = self._plot_push_box_dashboard()
        elif task == "big_box":
            figure = self._plot_big_box_dashboard()
        else:
            figure = self._plot_locomotion_dashboard()
        figure.suptitle(f"{task} task diagnostics")
        dashboard_path = os.path.join(
            task_directory, f"{run_name}_dashboard.png"
        )
        figure.savefig(dashboard_path, dpi=180, bbox_inches="tight")
        self._pyplot().close(figure)
        print(f"Saved task dashboard: {dashboard_path}")
        print(f"Saved cost log: {cost_path}")
        return dashboard_path

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
    model_path = os.path.join(os.path.dirname(__file__), "../models/scene.xml")

    simulator = Simulator(T = 300, dt=0.002, viewer=True, gravity=True, model_path=model_path)
    simulator.run()
    simulator.plot_trajectory()
