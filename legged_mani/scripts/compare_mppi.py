"""Compare three arm-IK/MPPI ablations headlessly.

All modes use the same ``ee_tracking`` task, model, initial state, gait phase,
arm IK solver, and goal-transition logic.

``arm_ik_nominal``
    Run MPPI sampling and rollouts for all leg actuators while keeping every
    arm candidate fixed to the current IK nominal. Arm joint/control, EE, and
    arm-collision terms are excluded from the MPPI objective.

``arm_fixed_ee_cost``
    Keep the arm fixed to the same IK nominal and exclude arm joint/control
    costs, but retain EE and collision costs so MPPI can adapt the legs/base.

``whole_body_mppi``
    Use the same gait/IK reference as the nominal input, then optimize all
    leg and arm controls with the existing whole-body MPPI controller.

Example
-------
    python legged_mani/scripts/compare_mppi.py
    python legged_mani/scripts/compare_mppi.py --steps 1000
    python legged_mani/scripts/compare_mppi.py --output /tmp/compare_mppi.png
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, sleep

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from tqdm import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from mani_mppi.control.controllers.mppi_ee_tracking import MPPI
from mani_mppi.interface.simulator import Simulator
from mani_mppi.utils.tasks import get_task
from mani_mppi.utils.transforms import batch_world_to_local_velocity


TASK = "ee_tracking"
MODES = (
    "arm_ik_nominal",
    "arm_fixed_ee_cost",
    "whole_body_mppi",
)
DEFAULT_OUTPUT = (
    PACKAGE_ROOT
    / "mani_mppi"
    / "analysis"
    / "compare_mppi_ee_tracking.png"
)


@dataclass
class RunLog:
    """Time-series and summary state from one control mode."""

    mode: str
    time: np.ndarray
    ee_error: np.ndarray
    base_position: np.ndarray
    roll_pitch: np.ndarray
    clearance: np.ndarray
    action_delta: np.ndarray
    update_ms: np.ndarray
    goal_index: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    task_success: bool


def quaternion_to_roll_pitch(quaternion: np.ndarray) -> np.ndarray:
    """Convert one MuJoCo ``[w, x, y, z]`` quaternion to roll/pitch."""
    w, x, y, z = np.asarray(quaternion, dtype=float)
    roll = np.arctan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
    return np.array([roll, pitch])


def leg_only_cost(
    agent: MPPI,
    states: np.ndarray,
    actions: np.ndarray,
    joints_ref: np.ndarray,
    body_ref: np.ndarray,
    rollout_sensors: np.ndarray | None = None,
) -> np.ndarray:
    """Score only base and leg behavior for the arm-IK nominal baseline."""
    del rollout_sensors
    num_samples, horizon = states.shape[:2]
    flat_states = states.reshape(-1, states.shape[-1]).copy()
    flat_actions = actions.reshape(-1, actions.shape[-1])
    body_refs = np.repeat(body_ref[None, :], len(flat_states), axis=0)
    tiled_joints = np.tile(
        joints_ref.T,
        (num_samples, 1, 1),
    ).reshape(-1, joints_ref.shape[0])
    base_velocity_ref = np.zeros((len(flat_states), 6), dtype=float)
    base_velocity_ref[:, :2] = body_refs[:, 7:9]
    x_ref = np.concatenate(
        (
            body_refs[:, :7],
            tiled_joints[:, :16],
            base_velocity_ref,
            tiled_joints[:, 16:],
        ),
        axis=1,
    )
    flat_states[:, 23:26] = batch_world_to_local_velocity(
        flat_states[:, 3:7],
        flat_states[:, 23:26],
    )

    agent.collision_min_clearance = np.full(num_samples, np.inf)
    agent.collision_valid_rollouts = np.ones(num_samples, dtype=bool)
    agent.collision_exact_evaluations = 0
    costs = agent.quadruped_cost_np(
        flat_states,
        flat_actions,
        x_ref,
    )
    return costs.reshape(num_samples, horizon).sum(axis=1)


def freeze_arm_to_ik_nominal(agent: MPPI) -> np.ndarray:
    """Fix arm candidates to IK and remove arm joint/control costs."""
    actuator_joint_ids = agent.model.actuator_trnid[:, 0]
    arm_indices = np.flatnonzero(
        np.isin(
            actuator_joint_ids,
            agent.arm_kinematics.arm_joint_ids,
        )
    )
    if len(arm_indices) != 4:
        raise ValueError(
            "Expected four arm actuators, found "
            f"{len(arm_indices)} at indices {arm_indices}"
        )
    agent.base_noise_sigma[arm_indices] = 0.0
    agent.set_noise_for_gait(agent.default_gait)
    agent.gait_correction[:, arm_indices] = 0.0

    # State layout is [base pose 7, joint q 16, base velocity 6,
    # joint dq 16]. Remove the four arm joint position/velocity weights.
    arm_q_weights = 7 + arm_indices
    arm_dq_weights = 29 + arm_indices
    agent.state_cost_weights[arm_q_weights] = 0.0
    agent.state_cost_weights[arm_dq_weights] = 0.0
    agent.control_cost_weights[arm_indices] = 0.0
    agent.Q = np.diag(agent.state_cost_weights)
    agent.R = np.diag(agent.control_cost_weights)
    return arm_indices


def configure_arm_ik_nominal(agent: MPPI) -> np.ndarray:
    """Retain only base/leg costs while the arm follows IK nominal."""
    arm_indices = freeze_arm_to_ik_nominal(agent)
    # EE pose and arm collision remain evaluation metrics, but they do not
    # influence candidate weights or hard-rejection decisions in this mode.
    agent.ee_position_weight = 0.0
    agent.ee_orientation_weight = 0.0
    agent.ee_terminal_scale = 0.0
    agent.collision_enabled = False
    agent.capture_rollout_sensors = False
    agent.sensor_rollouts = None
    agent.cost_func = lambda *args, **kwargs: leg_only_cost(
        agent,
        *args,
        **kwargs,
    )
    return arm_indices


def configure_arm_fixed_ee_cost(agent: MPPI) -> np.ndarray:
    """Keep EE/collision costs active while fixing the arm to IK nominal."""
    return freeze_arm_to_ik_nominal(agent)


def arm_clearance(agent: MPPI, observation: np.ndarray) -> float:
    """Evaluate the minimum analytic arm-to-torso capsule clearance."""
    state = np.asarray(observation, dtype=float)[None, :]
    arm_positions, _ = agent._batch_arm_fk(state)
    clearance, _ = agent._arm_torso_clearance(state, arm_positions)
    return float(np.min(clearance))


def run_mode(mode: str, steps: int) -> RunLog:
    """Run one headless controller mode from a fresh identical reset."""
    if mode not in MODES:
        raise ValueError(f"Unknown comparison mode: {mode}")

    task_data = get_task(TASK)
    sim_path = PACKAGE_ROOT / "mani_mppi" / task_data["sim_path"]
    agent = MPPI(task=TASK)
    simulator = Simulator(
        agent=agent,
        model_path=sim_path,
        T=steps + 1,
        dt=float(agent.model.opt.timestep),
        viewer=False,
        ctrl_rate=round(1.0 / float(agent.model.opt.timestep)),
    )
    if mode == "arm_ik_nominal":
        arm_indices = configure_arm_ik_nominal(agent)
    elif mode == "arm_fixed_ee_cost":
        arm_indices = configure_arm_fixed_ee_cost(agent)
    else:
        arm_indices = np.empty(0, dtype=int)

    time_log = np.empty(steps)
    ee_error_log = np.empty(steps)
    base_position_log = np.empty((steps, 3))
    roll_pitch_log = np.empty((steps, 2))
    clearance_log = np.empty(steps)
    action_delta_log = np.empty(steps)
    update_ms_log = np.empty(steps)
    goal_index_log = np.empty(steps, dtype=int)
    qpos_log = np.empty((steps, simulator.model.nq))
    qvel_log = np.empty((steps, simulator.model.nv))
    previous_action = None

    try:
        progress = tqdm(
            range(steps),
            desc=mode,
            unit="step",
            dynamic_ncols=True,
        )
        for step in progress:
            observation = np.concatenate(
                (simulator.data.qpos, simulator.data.qvel)
            )
            active_goal = agent.goal_index
            target = agent.ee_goal_pos[active_goal].copy()

            start = perf_counter()
            if mode in {"arm_ik_nominal", "arm_fixed_ee_cost"}:
                # Eliminate any numerical carry-over from previous weighted
                # updates. Zero arm noise makes every sampled arm trajectory
                # exactly equal to the refreshed IK nominal, while the legs
                # retain their configured MPPI exploration.
                agent.gait_correction[:, arm_indices] = 0.0
            action = agent.update(observation)
            update_ms_log[step] = 1000.0 * (perf_counter() - start)

            simulator.step(action)
            result = np.concatenate(
                (simulator.data.qpos, simulator.data.qvel)
            )
            ee_position, _ = agent._ee_pose(result)

            time_log[step] = simulator.data.time
            ee_error_log[step] = np.linalg.norm(ee_position - target)
            base_position_log[step] = simulator.data.qpos[:3]
            roll_pitch_log[step] = quaternion_to_roll_pitch(
                simulator.data.qpos[3:7]
            )
            clearance_log[step] = arm_clearance(agent, result)
            action_delta_log[step] = (
                0.0
                if previous_action is None
                else np.linalg.norm(action - previous_action)
            )
            goal_index_log[step] = active_goal
            qpos_log[step] = simulator.data.qpos
            qvel_log[step] = simulator.data.qvel
            previous_action = action.copy()

            if agent.goal_reached(result):
                agent.next_goal()

            if (step + 1) % 100 == 0:
                progress.set_postfix(
                    goal=agent.goal_index,
                    ee_mm=f"{1000.0 * ee_error_log[step]:.1f}",
                    hz=f"{1000.0 / np.mean(update_ms_log[:step + 1]):.1f}",
                )
    finally:
        agent.close()

    return RunLog(
        mode=mode,
        time=time_log,
        ee_error=ee_error_log,
        base_position=base_position_log,
        roll_pitch=roll_pitch_log,
        clearance=clearance_log,
        action_delta=action_delta_log,
        update_ms=update_ms_log,
        goal_index=goal_index_log,
        qpos=qpos_log,
        qvel=qvel_log,
        task_success=bool(agent.task_success),
    )


def save_tsv(log: RunLog, output: Path) -> Path:
    """Save one mode's time series next to the comparison figure."""
    path = output.with_name(f"{output.stem}_{log.mode}.tsv")
    data = np.column_stack(
        (
            log.time,
            log.goal_index,
            log.ee_error,
            log.base_position,
            np.rad2deg(log.roll_pitch),
            log.clearance,
            log.action_delta,
            log.update_ms,
        )
    )
    np.savetxt(
        path,
        data,
        delimiter="\t",
        fmt="%.9g",
        header=(
            "time_s\tgoal_index\tee_error_m\tbase_x_m\tbase_y_m\t"
            "base_z_m\troll_deg\tpitch_deg\tclearance_m\t"
            "action_delta_l2\tupdate_ms"
        ),
        comments="",
    )
    return path


def transition_times(log: RunLog) -> np.ndarray:
    """Return times at which the active EE goal changed."""
    changes = np.flatnonzero(np.diff(log.goal_index) != 0) + 1
    return log.time[changes]


def plot_comparison(logs: list[RunLog], output: Path) -> None:
    """Create a common six-panel comparison plot."""
    colors = {
        "arm_ik_nominal": "tab:orange",
        "arm_fixed_ee_cost": "tab:green",
        "whole_body_mppi": "tab:blue",
    }
    labels = {
        "arm_ik_nominal": "Arm IK nominal (arm costs off)",
        "arm_fixed_ee_cost": "Arm fixed + EE/collision costs",
        "whole_body_mppi": "Whole-body MPPI",
    }
    figure, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    axes = axes.ravel()

    for log in logs:
        color = colors[log.mode]
        label = labels[log.mode]
        axes[0].plot(log.time, log.ee_error, color=color, label=label)
        axes[1].plot(
            log.time,
            log.base_position[:, 0],
            color=color,
            label=f"{label} x",
        )
        axes[1].plot(
            log.time,
            log.base_position[:, 1],
            color=color,
            linestyle=":",
            label=f"{label} y",
        )
        axes[1].plot(
            log.time,
            log.base_position[:, 2],
            color=color,
            linestyle="--",
            label=f"{label} z",
        )
        axes[2].plot(
            log.time,
            np.rad2deg(log.roll_pitch[:, 0]),
            color=color,
            label=f"{label} roll",
        )
        axes[2].plot(
            log.time,
            np.rad2deg(log.roll_pitch[:, 1]),
            color=color,
            linestyle="--",
            label=f"{label} pitch",
        )
        axes[3].plot(log.time, log.clearance, color=color, label=label)
        axes[4].plot(log.time, log.action_delta, color=color, label=label)
        axes[5].plot(log.time, log.update_ms, color=color, label=label)

        for transition in transition_times(log):
            axes[0].axvline(
                transition,
                color=color,
                alpha=0.18,
                linewidth=0.8,
            )

    axes[0].axhline(
        get_task(TASK)["ee_pos_thresh"],
        color="black",
        linestyle=":",
        label="goal threshold",
    )
    axes[0].set_title("End-effector position error")
    axes[0].set_ylabel("error [m]")
    axes[0].set_yscale("log")

    axes[1].set_title("Base position")
    axes[1].set_ylabel("position [m]")

    axes[2].set_title("Base attitude")
    axes[2].set_ylabel("angle [deg]")

    axes[3].axhline(
        0.0,
        color="black",
        linestyle=":",
        linewidth=0.8,
    )
    axes[3].set_title("Minimum arm-to-torso clearance")
    axes[3].set_ylabel("clearance [m]")

    axes[4].set_title("Control variation")
    axes[4].set_ylabel(r"$||u_t-u_{t-1}||_2$")

    axes[5].set_title("Controller computation time")
    axes[5].set_ylabel("update [ms]")
    axes[5].set_yscale("log")

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    axes[4].set_xlabel("simulation time [s]")
    axes[5].set_xlabel("simulation time [s]")

    figure.suptitle(
        "EE tracking: IK nominal, fixed-arm EE cost, and whole-body MPPI",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def print_summary(log: RunLog) -> None:
    """Print compact accuracy, stability, safety, and timing metrics."""
    update_hz = 1000.0 / np.mean(log.update_ms)
    base_xy_drift = np.linalg.norm(
        log.base_position[-1, :2] - log.base_position[0, :2]
    )
    max_attitude = np.max(np.abs(np.rad2deg(log.roll_pitch)))
    reached_fraction = np.mean(
        log.ee_error <= get_task(TASK)["ee_pos_thresh"]
    )
    print(
        f"{log.mode}: "
        f"final_goal={log.goal_index[-1]}, "
        f"task_success={log.task_success}, "
        f"ee_rmse={np.sqrt(np.mean(log.ee_error**2)):.4f} m, "
        f"ee_final={log.ee_error[-1]:.4f} m, "
        f"within_threshold={100.0 * reached_fraction:.1f}%, "
        f"base_xy_drift={base_xy_drift:.4f} m, "
        f"max_abs_roll_pitch={max_attitude:.2f} deg, "
        f"min_clearance={np.min(log.clearance):.4f} m, "
        f"update={update_hz:.1f} Hz, "
        f"p95={np.percentile(log.update_ms, 95):.3f} ms"
    )


def replay_trajectories(
    logs: list[RunLog],
    *,
    playback_speed: float,
    replay_fps: float,
) -> None:
    """Replay both saved trajectories sequentially in one MuJoCo viewer."""
    import mujoco_viewer

    task_data = get_task(TASK)
    model_path = PACKAGE_ROOT / "mani_mppi" / task_data["sim_path"]
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(
        model,
        data,
        hide_menus=True,
    )
    target_positions = np.asarray(task_data["ee_goal_pos"], dtype=float)
    simulation_dt = (
        float(np.median(np.diff(logs[0].time)))
        if len(logs[0].time) > 1
        else 0.01
    )
    frame_stride = max(
        1,
        round(1.0 / (simulation_dt * replay_fps)),
    )
    frame_period = frame_stride * simulation_dt / playback_speed

    try:
        for log in logs:
            for frame in range(0, len(log.time), frame_stride):
                if not viewer.is_alive:
                    return
                frame_start = perf_counter()
                data.qpos[:] = log.qpos[frame]
                data.qvel[:] = log.qvel[frame]
                data.time = log.time[frame]
                mujoco.mj_forward(model, data)

                goal_index = int(log.goal_index[frame])
                viewer.add_marker(
                    pos=target_positions[goal_index],
                    size=[0.05, 0.05, 0.05],
                    rgba=[1.0, 0.15, 0.05, 0.9],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    label=f"EE goal {goal_index}",
                )
                viewer.add_marker(
                    pos=data.qpos[:3] + np.array([0.0, 0.0, 0.65]),
                    size=[0.001, 0.001, 0.001],
                    rgba=[0.0, 0.0, 0.0, 0.0],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    label=(
                        f"{log.mode} | t={log.time[frame]:.2f}s | "
                        f"goal={goal_index}"
                    ),
                )
                viewer.render()
                remaining = frame_period - (perf_counter() - frame_start)
                if remaining > 0.0:
                    sleep(remaining)

            # Hold the final pose briefly so the mode transition is visible.
            if viewer.is_alive:
                sleep(0.5 / playback_speed)
    finally:
        viewer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run identical headless EE-tracking scenarios for three arm "
            "IK/MPPI ablations, then plot and replay them."
        )
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Simulation steps per mode (default: 2000 = 20 seconds)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Comparison PNG path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Save plots/data without opening the post-run MuJoCo replay",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="MuJoCo replay speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--replay-fps",
        type=float,
        default=50.0,
        help="Maximum replay render rate (default: 50 FPS)",
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.output.suffix.lower() != ".png":
        parser.error("--output must use a .png extension")
    if args.playback_speed <= 0.0:
        parser.error("--playback-speed must be positive")
    if args.replay_fps <= 0.0:
        parser.error("--replay-fps must be positive")
    return args


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    logs = [run_mode(mode, args.steps) for mode in MODES]
    for log in logs:
        print_summary(log)
        print(f"saved data: {save_tsv(log, output)}")
    plot_comparison(logs, output)
    print(f"saved plot: {output}")
    if not args.no_viewer:
        print(
            "starting MuJoCo replay: "
            "arm_ik_nominal -> arm_fixed_ee_cost -> whole_body_mppi"
        )
        replay_trajectories(
            logs,
            playback_speed=args.playback_speed,
            replay_fps=args.replay_fps,
        )


if __name__ == "__main__":
    main()
