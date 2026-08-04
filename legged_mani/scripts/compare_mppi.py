"""Compare arm-IK/MPPI ablations with independent target episodes.

Every ``(target, seed, mode)`` combination starts from a fresh controller and
MuJoCo simulator.  A single target remains active for the full episode, so a
previous target's terminal state cannot influence the next result.

``arm_ik_nominal``
    Fix the arm to the IK nominal and remove arm Q/R, EE, terminal, and
    collision terms.  This is a negative control, not the primary baseline.

``arm_fixed_same_cost``
    Fix the arm to the IK nominal while retaining exactly the whole-body
    objective and collision checks.  Only the arm search space is removed.

``whole_body_mppi``
    Retain the same gait/IK prior and objective, and optimize both leg and arm
    controls with MPPI.

Examples
--------
    python legged_mani/scripts/compare_mppi.py --no-viewer
    python legged_mani/scripts/compare_mppi.py --targets 0,1 --seeds 42
    python legged_mani/scripts/compare_mppi.py --seeds 0:10 --no-viewer
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, sleep

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.lines import Line2D
from tqdm import tqdm


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from mani_mppi.control.controllers.mppi_ee_tracking import MPPI
from mani_mppi.control.gait_scheduler.scheduler import Timer
from mani_mppi.interface.simulator import Simulator
from mani_mppi.utils.tasks import get_task
from mani_mppi.utils.transforms import batch_world_to_local_velocity


TASK = "ee_tracking"
ROLLOUT_MODE = "original_spline"
MODES = (
    "arm_ik_nominal",
    "arm_fixed_same_cost",
    "whole_body_mppi",
)
COLORS = {
    "arm_ik_nominal": "#E69F00",
    "arm_fixed_same_cost": "#009E73",
    "whole_body_mppi": "#0072B2",
}
LABELS = {
    "arm_ik_nominal": "Arm IK nominal (costs off)",
    "arm_fixed_same_cost": "Arm fixed (same cost)",
    "whole_body_mppi": "Whole-body MPPI",
}
DEFAULT_OUTPUT_DIR = (
    PACKAGE_ROOT
    / "mani_mppi"
    / "analysis"
    / "compare_ee_tracking"
)


@dataclass(frozen=True)
class EpisodeSpec:
    """One paired experimental condition."""

    target_index: int
    target_position: tuple[float, float, float]
    target_quaternion: tuple[float, float, float, float]
    seed: int
    mode: str


@dataclass
class EpisodeLog:
    """Time series and controller metadata from one independent episode."""

    spec: EpisodeSpec
    time: np.ndarray
    ee_error: np.ndarray
    orientation_error: np.ndarray
    base_position: np.ndarray
    roll_pitch: np.ndarray
    clearance: np.ndarray
    leg_action_delta: np.ndarray
    arm_action_delta: np.ndarray
    update_ms: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    initial_qpos: np.ndarray
    initial_qvel: np.ndarray
    initial_ee_error: float
    success: bool
    success_time_s: float
    collision_hard_distance: float
    dt: float
    n_samples: int
    horizon: int
    state_cost_weights: np.ndarray
    control_cost_weights: np.ndarray
    ee_position_weight: float
    ee_orientation_weight: float
    ee_terminal_scale: float
    collision_enabled: bool
    leg_noise_sigma: np.ndarray
    arm_noise_sigma: np.ndarray


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


def quaternion_tracking_error(
    quaternion: np.ndarray,
    target_quaternion: np.ndarray,
) -> float:
    """Return the sign-invariant quaternion error used by goal_reached."""
    dot = abs(
        float(
            np.dot(
                np.asarray(quaternion, dtype=float),
                np.asarray(target_quaternion, dtype=float),
            )
        )
    )
    return 1.0 - float(np.clip(dot, 0.0, 1.0))


def arm_actuator_indices(agent: MPPI) -> np.ndarray:
    """Resolve the four arm actuator indices from the MuJoCo model."""
    actuator_joint_ids = agent.model.actuator_trnid[:, 0]
    indices = np.flatnonzero(
        np.isin(
            actuator_joint_ids,
            agent.arm_kinematics.arm_joint_ids,
        )
    )
    if len(indices) != 4:
        raise ValueError(
            "Expected four arm actuators, found "
            f"{len(indices)} at indices {indices}"
        )
    return indices


def leg_only_cost(
    agent: MPPI,
    states: np.ndarray,
    actions: np.ndarray,
    joints_ref: np.ndarray,
    body_ref: np.ndarray,
    rollout_sensors: np.ndarray | None = None,
) -> np.ndarray:
    """Score only base and leg behavior for the negative control."""
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


def freeze_arm_actions(agent: MPPI) -> np.ndarray:
    """Remove arm exploration while preserving the configured objective."""
    arm_indices = arm_actuator_indices(agent)
    agent.base_noise_sigma[arm_indices] = 0.0
    agent.set_noise_for_gait(agent.default_gait)
    agent.gait_correction[:, arm_indices] = 0.0
    return arm_indices


def disable_arm_objective(
    agent: MPPI,
    arm_indices: np.ndarray,
) -> None:
    """Remove arm Q/R and task terms for the negative control only."""
    # Compact state layout: base pose error 6, joint q 16,
    # base velocity 6, joint dq 16.
    arm_q_weights = 6 + arm_indices
    arm_dq_weights = 28 + arm_indices
    agent.state_cost_weights[arm_q_weights] = 0.0
    agent.state_cost_weights[arm_dq_weights] = 0.0
    agent.control_cost_weights[arm_indices] = 0.0
    agent.Q = np.diag(agent.state_cost_weights)
    agent.R = np.diag(agent.control_cost_weights)

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


def configure_mode(agent: MPPI, mode: str) -> np.ndarray:
    """Apply one ablation while leaving unrelated settings unchanged."""
    if mode == "arm_ik_nominal":
        arm_indices = freeze_arm_actions(agent)
        disable_arm_objective(agent, arm_indices)
        return arm_indices
    if mode == "arm_fixed_same_cost":
        return freeze_arm_actions(agent)
    if mode == "whole_body_mppi":
        return arm_actuator_indices(agent)
    raise ValueError(f"Unknown comparison mode: {mode}")


def configure_single_target(
    agent: MPPI,
    initial_qpos: np.ndarray,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
) -> None:
    """Replace the sequential task with one fixed, independently initialized target."""
    target_position = np.asarray(target_position, dtype=float)
    target_quaternion = np.asarray(target_quaternion, dtype=float)
    agent.ee_goal_pos = target_position[None, :].copy()
    agent.ee_goal_quat = target_quaternion[None, :].copy()
    agent.goal_index = 0
    agent.task_success = False
    agent.waiting_times = [0]
    agent.timer = Timer(end_time=0)

    # Use a best-effort IK nominal even when the arm alone cannot reach the
    # target; base/body motion remains available to the tested controller.
    agent.arm_reference = agent._solve_arm_ik(
        target_position,
        np.asarray(initial_qpos, dtype=float).copy(),
        damping=agent.arm_ik_damping,
        step_size=agent.arm_ik_step_size,
        max_iterations=agent.arm_ik_max_iterations,
        tolerance=agent.arm_ik_tolerance,
        strict=False,
    )
    agent.reset_planner()
    agent._reset_gait_nominal()
    agent.last_safe_trajectory = agent.trajectory.copy()


def arm_clearance(agent: MPPI, observation: np.ndarray) -> float:
    """Evaluate minimum analytic arm-to-torso capsule clearance."""
    state = np.asarray(observation, dtype=float)[None, :]
    arm_positions, _ = agent._batch_arm_fk(state)
    clearance, _ = agent._arm_torso_clearance(state, arm_positions)
    return float(np.min(clearance))


def run_episode(
    spec: EpisodeSpec,
    *,
    episode_seconds: float,
    steps_override: int | None,
    success_hold_seconds: float,
) -> EpisodeLog:
    """Run one fixed-target episode from a fresh deterministic reset."""
    task_data = get_task(TASK)
    sim_path = PACKAGE_ROOT / "mani_mppi" / task_data["sim_path"]
    agent = MPPI(task=TASK, rollout_mode=ROLLOUT_MODE)
    dt = float(agent.model.opt.timestep)
    steps = (
        int(steps_override)
        if steps_override is not None
        else max(1, int(round(episode_seconds / dt)))
    )
    success_hold_steps = max(
        1,
        int(round(success_hold_seconds / dt)),
    )
    simulator = Simulator(
        agent=agent,
        model_path=sim_path,
        T=steps + 1,
        dt=dt,
        viewer=False,
        ctrl_rate=round(1.0 / dt),
    )

    initial_qpos = simulator.data.qpos.copy()
    initial_qvel = simulator.data.qvel.copy()
    target_position = np.asarray(spec.target_position, dtype=float)
    target_quaternion = np.asarray(spec.target_quaternion, dtype=float)
    configure_single_target(
        agent,
        initial_qpos,
        target_position,
        target_quaternion,
    )
    arm_indices = configure_mode(agent, spec.mode)
    leg_indices = np.setdiff1d(
        np.arange(agent.act_dim, dtype=int),
        arm_indices,
    )
    # Reset after every other initialization step so paired modes consume
    # identical random tensors for the same seed.
    agent.random_generator = np.random.default_rng(spec.seed)

    initial_observation = np.concatenate((initial_qpos, initial_qvel))
    initial_ee_position, _ = agent._ee_pose(initial_observation)
    initial_ee_error = float(
        np.linalg.norm(initial_ee_position - target_position)
    )

    time_log = np.empty(steps)
    ee_error_log = np.empty(steps)
    orientation_error_log = np.empty(steps)
    base_position_log = np.empty((steps, 3))
    roll_pitch_log = np.empty((steps, 2))
    clearance_log = np.empty(steps)
    leg_action_delta_log = np.empty(steps)
    arm_action_delta_log = np.empty(steps)
    update_ms_log = np.empty(steps)
    qpos_log = np.empty((steps, simulator.model.nq))
    qvel_log = np.empty((steps, simulator.model.nv))
    previous_action = None
    hold_count = 0
    success_time = float("nan")

    description = (
        f"T{spec.target_index} seed={spec.seed} {spec.mode}"
    )
    try:
        progress = tqdm(
            range(steps),
            desc=description,
            unit="step",
            dynamic_ncols=True,
        )
        for step in progress:
            observation = np.concatenate(
                (simulator.data.qpos, simulator.data.qvel)
            )

            start = perf_counter()
            if spec.mode in {
                "arm_ik_nominal",
                "arm_fixed_same_cost",
            }:
                # Remove numerical carry-over from the preceding weighted
                # update before refreshing the gait/IK nominal.
                agent.gait_correction[:, arm_indices] = 0.0
            action = agent.update(observation)
            update_ms_log[step] = 1000.0 * (perf_counter() - start)

            simulator.step(action)
            result = np.concatenate(
                (simulator.data.qpos, simulator.data.qvel)
            )
            ee_position, ee_quaternion = agent._ee_pose(result)
            ee_error = float(
                np.linalg.norm(ee_position - target_position)
            )
            orientation_error = quaternion_tracking_error(
                ee_quaternion,
                target_quaternion,
            )

            time_log[step] = simulator.data.time
            ee_error_log[step] = ee_error
            orientation_error_log[step] = orientation_error
            base_position_log[step] = simulator.data.qpos[:3]
            roll_pitch_log[step] = quaternion_to_roll_pitch(
                simulator.data.qpos[3:7]
            )
            clearance_log[step] = arm_clearance(agent, result)
            if previous_action is None:
                leg_action_delta_log[step] = 0.0
                arm_action_delta_log[step] = 0.0
            else:
                delta = action - previous_action
                leg_action_delta_log[step] = np.linalg.norm(
                    delta[leg_indices]
                )
                arm_action_delta_log[step] = np.linalg.norm(
                    delta[arm_indices]
                )
            qpos_log[step] = simulator.data.qpos
            qvel_log[step] = simulator.data.qvel
            previous_action = action.copy()

            within_threshold = (
                ee_error <= agent.ee_pos_thresh
                and orientation_error <= agent.ee_ori_thresh
            )
            hold_count = hold_count + 1 if within_threshold else 0
            if np.isnan(success_time) and hold_count >= success_hold_steps:
                # Record confirmation time, not the optimistic first sample
                # at the start of the qualifying dwell window.
                success_time = float(simulator.data.time)

            if (step + 1) % 100 == 0:
                progress.set_postfix(
                    ee_mm=f"{1000.0 * ee_error:.1f}",
                    success=not np.isnan(success_time),
                    hz=f"{1000.0 / np.mean(update_ms_log[:step + 1]):.1f}",
                )
    finally:
        agent.close()

    return EpisodeLog(
        spec=spec,
        time=time_log,
        ee_error=ee_error_log,
        orientation_error=orientation_error_log,
        base_position=base_position_log,
        roll_pitch=roll_pitch_log,
        clearance=clearance_log,
        leg_action_delta=leg_action_delta_log,
        arm_action_delta=arm_action_delta_log,
        update_ms=update_ms_log,
        qpos=qpos_log,
        qvel=qvel_log,
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        initial_ee_error=initial_ee_error,
        success=not np.isnan(success_time),
        success_time_s=success_time,
        collision_hard_distance=float(agent.collision_hard_distance),
        dt=dt,
        n_samples=int(agent.n_samples),
        horizon=int(agent.horizon),
        state_cost_weights=agent.state_cost_weights.copy(),
        control_cost_weights=agent.control_cost_weights.copy(),
        ee_position_weight=float(agent.ee_position_weight),
        ee_orientation_weight=float(agent.ee_orientation_weight),
        ee_terminal_scale=float(agent.ee_terminal_scale),
        collision_enabled=bool(agent.collision_enabled),
        # Compare the configured exploration budget, not the terminal
        # effective sigma: dynamic EE-error scaling is closed-loop and modes
        # can legitimately finish an episode at different scales.
        leg_noise_sigma=agent.base_noise_sigma[leg_indices].copy(),
        arm_noise_sigma=agent.base_noise_sigma[arm_indices].copy(),
    )


def trajectory_path(log: EpisodeLog, output_dir: Path) -> Path:
    target_dir = (
        output_dir
        / "trajectories"
        / f"target_{log.spec.target_index:02d}"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / (
        f"seed_{log.spec.seed:04d}_{log.spec.mode}.tsv"
    )


def save_trajectory(log: EpisodeLog, output_dir: Path) -> Path:
    """Save one episode time series using an unambiguous paired filename."""
    path = trajectory_path(log, output_dir)
    data = np.column_stack(
        (
            log.time,
            log.ee_error,
            log.orientation_error,
            log.base_position,
            np.rad2deg(log.roll_pitch),
            log.clearance,
            log.leg_action_delta,
            log.arm_action_delta,
            log.update_ms,
        )
    )
    np.savetxt(
        path,
        data,
        delimiter="\t",
        fmt="%.9g",
        header=(
            "time_s\tee_error_m\torientation_error\tbase_x_m\t"
            "base_y_m\tbase_z_m\troll_deg\tpitch_deg\tclearance_m\t"
            "leg_action_delta_l2\tarm_action_delta_l2\tupdate_ms"
        ),
        comments="",
    )
    return path


def episode_summary(
    log: EpisodeLog,
    *,
    steady_window_seconds: float,
    latency_warmup_steps: int,
) -> dict[str, object]:
    """Compute robust episode-level accuracy, safety, and timing metrics."""
    steady_steps = max(
        1,
        min(
            len(log.time),
            int(round(steady_window_seconds / log.dt)),
        ),
    )
    latency_start = min(latency_warmup_steps, len(log.update_ms) - 1)
    latency = log.update_ms[latency_start:]
    initial_xy = log.initial_qpos[:2]
    planar_displacement = np.linalg.norm(
        log.base_position[:, :2] - initial_xy[None, :],
        axis=1,
    )
    attitude = np.max(
        np.abs(np.rad2deg(log.roll_pitch)),
        axis=1,
    )
    auc_time = np.concatenate(([0.0], log.time))
    auc_error = np.concatenate(
        ([log.initial_ee_error], log.ee_error)
    )
    error_auc = float(np.trapezoid(auc_error, x=auc_time))
    collision_violation = (
        log.clearance < log.collision_hard_distance
    )

    return {
        "target_index": log.spec.target_index,
        "target_x_m": log.spec.target_position[0],
        "target_y_m": log.spec.target_position[1],
        "target_z_m": log.spec.target_position[2],
        "seed": log.spec.seed,
        "mode": log.spec.mode,
        "success": int(log.success),
        "success_time_s": log.success_time_s,
        "episode_duration_s": float(log.time[-1]),
        "initial_error_m": log.initial_ee_error,
        "error_rmse_m": float(
            np.sqrt(np.mean(log.ee_error**2))
        ),
        "error_mean_m": float(np.mean(log.ee_error)),
        "error_auc_m_s": error_auc,
        "steady_error_mean_m": float(
            np.mean(log.ee_error[-steady_steps:])
        ),
        "steady_error_p95_m": float(
            np.percentile(log.ee_error[-steady_steps:], 95)
        ),
        "minimum_clearance_m": float(np.min(log.clearance)),
        "collision_violation_fraction": float(
            np.mean(collision_violation)
        ),
        "final_base_xy_displacement_m": float(
            planar_displacement[-1]
        ),
        "maximum_base_xy_displacement_m": float(
            np.max(planar_displacement)
        ),
        "minimum_base_height_m": float(
            min(log.initial_qpos[2], np.min(log.base_position[:, 2]))
        ),
        "maximum_base_height_m": float(
            max(log.initial_qpos[2], np.max(log.base_position[:, 2]))
        ),
        "maximum_abs_roll_pitch_deg": float(np.max(attitude)),
        "mean_leg_action_delta": float(
            np.mean(log.leg_action_delta)
        ),
        "mean_arm_action_delta": float(
            np.mean(log.arm_action_delta)
        ),
        "latency_median_ms": float(np.median(latency)),
        "latency_p95_ms": float(np.percentile(latency, 95)),
    }


def save_episode_summary(
    rows: list[dict[str, object]],
    output_dir: Path,
) -> Path:
    path = output_dir / "episode_summary.tsv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def validate_fairness(logs: list[EpisodeLog]) -> None:
    """Fail loudly if a paired episode differs outside its intended ablation."""
    groups: dict[tuple[int, int], dict[str, EpisodeLog]] = {}
    for log in logs:
        key = (log.spec.target_index, log.spec.seed)
        groups.setdefault(key, {})[log.spec.mode] = log

    for key, by_mode in groups.items():
        missing = set(MODES) - set(by_mode)
        if missing:
            raise RuntimeError(
                f"Paired episode {key} is missing modes: {sorted(missing)}"
            )
        reference = by_mode[MODES[0]]
        for mode in MODES[1:]:
            other = by_mode[mode]
            np.testing.assert_array_equal(
                reference.initial_qpos,
                other.initial_qpos,
                err_msg=f"Initial qpos differs for {key}",
            )
            np.testing.assert_array_equal(
                reference.initial_qvel,
                other.initial_qvel,
                err_msg=f"Initial qvel differs for {key}",
            )
            if (
                reference.dt != other.dt
                or reference.n_samples != other.n_samples
                or reference.horizon != other.horizon
            ):
                raise RuntimeError(
                    f"Rollout budget differs within paired episode {key}"
                )

        fixed = by_mode["arm_fixed_same_cost"]
        whole = by_mode["whole_body_mppi"]
        np.testing.assert_array_equal(
            fixed.state_cost_weights,
            whole.state_cost_weights,
            err_msg=f"State costs differ for primary comparison {key}",
        )
        np.testing.assert_array_equal(
            fixed.control_cost_weights,
            whole.control_cost_weights,
            err_msg=f"Control costs differ for primary comparison {key}",
        )
        fixed_objective = (
            fixed.ee_position_weight,
            fixed.ee_orientation_weight,
            fixed.ee_terminal_scale,
            fixed.collision_enabled,
        )
        whole_objective = (
            whole.ee_position_weight,
            whole.ee_orientation_weight,
            whole.ee_terminal_scale,
            whole.collision_enabled,
        )
        if fixed_objective != whole_objective:
            raise RuntimeError(
                f"Task objectives differ for primary comparison {key}"
            )
        np.testing.assert_array_equal(
            fixed.leg_noise_sigma,
            whole.leg_noise_sigma,
            err_msg=f"Leg exploration differs for {key}",
        )
        if np.any(fixed.arm_noise_sigma != 0.0):
            raise RuntimeError(
                f"Fixed-arm mode has nonzero arm noise for {key}"
            )


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Return a centered rolling median with edge padding."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or window <= 1:
        return values.copy()
    window = min(int(window), len(values))
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        window,
    )
    return np.median(windows, axis=-1)


def mode_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=COLORS[mode],
            linewidth=2.5,
            label=LABELS[mode],
        )
        for mode in MODES
    ]


def finish_figure(figure, axes, title: str) -> None:
    for axis in np.asarray(axes).ravel():
        if axis.axison:
            axis.grid(True, alpha=0.22)
    figure.legend(
        handles=mode_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(MODES),
        frameon=False,
    )
    figure.suptitle(title, fontsize=15, y=0.997)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.925))


def logs_for(
    logs: list[EpisodeLog],
    target_index: int,
    mode: str,
) -> list[EpisodeLog]:
    return [
        log
        for log in logs
        if log.spec.target_index == target_index
        and log.spec.mode == mode
    ]


def rows_for(
    rows: list[dict[str, object]],
    target_index: int,
    mode: str,
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if int(row["target_index"]) == target_index
        and row["mode"] == mode
    ]


def grouped_metric_bars(
    axis,
    rows: list[dict[str, object]],
    target_indices: list[int],
    key: str,
    *,
    title: str,
    ylabel: str,
    success_rate: bool = False,
) -> None:
    """Plot seed median/IQR, or mean success rate, for each target/mode."""
    x = np.arange(len(target_indices), dtype=float)
    width = 0.24
    for mode_index, mode in enumerate(MODES):
        centers = x + (mode_index - 1) * width
        values = []
        lower = []
        upper = []
        missing = []
        for target_index in target_indices:
            selected = rows_for(rows, target_index, mode)
            samples = np.asarray(
                [float(row[key]) for row in selected],
                dtype=float,
            )
            samples = samples[np.isfinite(samples)]
            missing.append(len(samples) == 0)
            if len(samples) == 0:
                values.append(0.0)
                lower.append(0.0)
                upper.append(0.0)
            elif success_rate:
                mean = float(np.mean(samples))
                values.append(mean)
                lower.append(0.0)
                upper.append(0.0)
            else:
                median = float(np.median(samples))
                q25, q75 = np.percentile(samples, [25, 75])
                values.append(median)
                lower.append(median - float(q25))
                upper.append(float(q75) - median)
        bars = axis.bar(
            centers,
            values,
            width=width,
            color=COLORS[mode],
            alpha=0.82,
            yerr=None if success_rate else np.vstack((lower, upper)),
            capsize=2,
        )
        for bar, is_missing in zip(bars, missing):
            if is_missing:
                bar.set_alpha(0.15)
                axis.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    0.0,
                    "NR",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xticks(x, [f"T{index}" for index in target_indices])
    if success_rate:
        axis.set_ylim(0.0, 1.05)


def plot_target_error_curves(
    logs: list[EpisodeLog],
    target_indices: list[int],
    output: Path,
    smooth_window: int,
) -> None:
    """Plot seed median and IQR error curves for every independent target."""
    columns = min(2, len(target_indices))
    rows = int(np.ceil(len(target_indices) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(14, 4.6 * rows),
        squeeze=False,
    )
    flat_axes = axes.ravel()
    threshold = float(get_task(TASK)["ee_pos_thresh"])

    for axis, target_index in zip(flat_axes, target_indices):
        target_logs = [
            log
            for log in logs
            if log.spec.target_index == target_index
        ]
        target = target_logs[0].spec.target_position
        for mode in MODES:
            selected = logs_for(logs, target_index, mode)
            errors = np.stack(
                [
                    rolling_median(log.ee_error, smooth_window)
                    for log in selected
                ],
                axis=0,
            )
            median = np.median(errors, axis=0)
            q25, q75 = np.percentile(errors, [25, 75], axis=0)
            axis.fill_between(
                selected[0].time,
                np.maximum(q25, 1e-6),
                np.maximum(q75, 1e-6),
                color=COLORS[mode],
                alpha=0.14,
                linewidth=0.0,
            )
            axis.plot(
                selected[0].time,
                np.maximum(median, 1e-6),
                color=COLORS[mode],
                linewidth=2.0,
            )
        axis.axhline(
            threshold,
            color="black",
            linestyle=":",
            linewidth=1.1,
        )
        axis.set_yscale("log")
        axis.set_title(
            f"T{target_index}: "
            f"[{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]"
        )
        axis.set_xlabel("time since episode start [s]")
        axis.set_ylabel("EE position error [m]")

    for axis in flat_axes[len(target_indices):]:
        axis.axis("off")
    finish_figure(
        figure,
        axes,
        "Independent-target EE error: seed median and IQR",
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_success_and_reach_time(
    rows: list[dict[str, object]],
    target_indices: list[int],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    grouped_metric_bars(
        axes[0],
        rows,
        target_indices,
        "success",
        title="Sustained-goal success rate",
        ylabel="success fraction",
        success_rate=True,
    )
    grouped_metric_bars(
        axes[1],
        rows,
        target_indices,
        "success_time_s",
        title="Confirmed time-to-success (successful episodes only)",
        ylabel="time [s]",
    )
    episode_duration = max(
        float(row["episode_duration_s"])
        for row in rows
    )
    axes[1].set_ylim(0.0, 1.05 * episode_duration)
    finish_figure(
        figure,
        axes,
        "Target success and reach time",
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_tracking_accuracy(
    rows: list[dict[str, object]],
    target_indices: list[int],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    metrics = (
        ("error_rmse_m", "Episode RMSE", "error [m]"),
        (
            "steady_error_mean_m",
            "Final-window mean error",
            "error [m]",
        ),
        (
            "steady_error_p95_m",
            "Final-window p95 error",
            "error [m]",
        ),
    )
    threshold = float(get_task(TASK)["ee_pos_thresh"])
    for axis, (key, title, ylabel) in zip(axes, metrics):
        grouped_metric_bars(
            axis,
            rows,
            target_indices,
            key,
            title=title,
            ylabel=ylabel,
        )
        axis.axhline(
            threshold,
            color="black",
            linestyle=":",
            linewidth=1.0,
        )
    finish_figure(
        figure,
        axes,
        "Independent-target tracking accuracy",
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_stability_and_safety(
    rows: list[dict[str, object]],
    target_indices: list[int],
    collision_hard_distance: float,
    output: Path,
) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(14, 12))
    flat_axes = axes.ravel()
    metrics = (
        (
            "maximum_base_xy_displacement_m",
            "Maximum planar base displacement",
            "displacement [m]",
        ),
        (
            "minimum_base_height_m",
            "Minimum base height",
            "height [m]",
        ),
        (
            "maximum_abs_roll_pitch_deg",
            "Maximum base attitude magnitude",
            "angle [deg]",
        ),
        (
            "minimum_clearance_m",
            "Minimum arm-to-torso clearance",
            "clearance [m]",
        ),
        (
            "collision_violation_fraction",
            "Hard-clearance violation fraction",
            "fraction",
        ),
    )
    for axis, (key, title, ylabel) in zip(flat_axes, metrics):
        grouped_metric_bars(
            axis,
            rows,
            target_indices,
            key,
            title=title,
            ylabel=ylabel,
        )
    flat_axes[4].set_ylim(0.0, 1.0)

    control_axis = flat_axes[5]
    grouped_metric_bars(
        control_axis,
        rows,
        target_indices,
        "mean_leg_action_delta",
        title="Control variation: leg bars, arm diamonds",
        ylabel=r"mean $||u_t-u_{t-1}||_2$",
    )
    x = np.arange(len(target_indices), dtype=float)
    width = 0.24
    for mode_index, mode in enumerate(MODES):
        centers = x + (mode_index - 1) * width
        medians = []
        for target_index in target_indices:
            selected = rows_for(rows, target_index, mode)
            values = np.asarray(
                [
                    float(row["mean_arm_action_delta"])
                    for row in selected
                ],
                dtype=float,
            )
            medians.append(float(np.median(values)))
        control_axis.scatter(
            centers,
            medians,
            color=COLORS[mode],
            edgecolors="black",
            linewidths=0.5,
            marker="D",
            s=28,
            zorder=3,
        )

    flat_axes[3].axhline(
        collision_hard_distance,
        color="black",
        linestyle=":",
        linewidth=1.0,
    )
    finish_figure(
        figure,
        axes,
        "Independent-target stability and safety",
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_computation_time(
    rows: list[dict[str, object]],
    target_indices: list[int],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    grouped_metric_bars(
        axes[0],
        rows,
        target_indices,
        "latency_median_ms",
        title="Controller latency median",
        ylabel="update time [ms]",
    )
    grouped_metric_bars(
        axes[1],
        rows,
        target_indices,
        "latency_p95_ms",
        title="Controller latency p95",
        ylabel="update time [ms]",
    )
    finish_figure(
        figure,
        axes,
        "Controller computation time after warm-up",
    )
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def print_episode_summary(
    log: EpisodeLog,
    row: dict[str, object],
) -> None:
    success_time = (
        f"{log.success_time_s:.3f}s"
        if log.success
        else "timeout"
    )
    print(
        f"T{log.spec.target_index} seed={log.spec.seed} "
        f"{log.spec.mode}: success={log.success} ({success_time}), "
        f"rmse={float(row['error_rmse_m']):.4f} m, "
        f"steady={float(row['steady_error_mean_m']):.4f} m, "
        f"min_clearance={float(row['minimum_clearance_m']):.4f} m, "
        f"latency_p95={float(row['latency_p95_ms']):.2f} ms"
    )


def replay_episodes(
    logs: list[EpisodeLog],
    *,
    target_index: int,
    seed: int,
    playback_speed: float,
    replay_fps: float,
) -> None:
    """Replay the three modes for one selected paired condition."""
    import mujoco_viewer

    selected = [
        log
        for mode in MODES
        for log in logs
        if log.spec.target_index == target_index
        and log.spec.seed == seed
        and log.spec.mode == mode
    ]
    if len(selected) != len(MODES):
        raise ValueError(
            f"Replay condition T{target_index}, seed={seed} "
            "does not contain all modes"
        )

    task_data = get_task(TASK)
    model_path = PACKAGE_ROOT / "mani_mppi" / task_data["sim_path"]
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(
        model,
        data,
        hide_menus=True,
    )
    simulation_dt = selected[0].dt
    frame_stride = max(
        1,
        round(1.0 / (simulation_dt * replay_fps)),
    )
    frame_period = frame_stride * simulation_dt / playback_speed

    try:
        for log in selected:
            target = np.asarray(log.spec.target_position)
            for frame in range(0, len(log.time), frame_stride):
                if not viewer.is_alive:
                    return
                frame_start = perf_counter()
                data.qpos[:] = log.qpos[frame]
                data.qvel[:] = log.qvel[frame]
                data.time = log.time[frame]
                mujoco.mj_forward(model, data)
                viewer.add_marker(
                    pos=target,
                    size=[0.05, 0.05, 0.05],
                    rgba=[1.0, 0.15, 0.05, 0.9],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    label=f"Target {target_index}",
                )
                viewer.add_marker(
                    pos=data.qpos[:3] + np.array([0.0, 0.0, 0.65]),
                    size=[0.001, 0.001, 0.001],
                    rgba=[0.0, 0.0, 0.0, 0.0],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    label=(
                        f"{LABELS[log.spec.mode]} | "
                        f"seed={seed} | t={log.time[frame]:.2f}s"
                    ),
                )
                viewer.render()
                remaining = frame_period - (
                    perf_counter() - frame_start
                )
                if remaining > 0.0:
                    sleep(remaining)
            if viewer.is_alive:
                sleep(0.5 / playback_speed)
    finally:
        viewer.close()


def parse_integer_selection(
    value: str,
    *,
    name: str,
) -> list[int]:
    """Parse comma-separated integers and Python-style ``start:stop:step``."""
    result: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            result.append(int(token))
            continue
        parts = token.split(":")
        if len(parts) not in {2, 3} or not all(parts[:2]):
            raise ValueError(
                f"Invalid {name} range '{token}'; use start:stop[:step]"
            )
        start = int(parts[0])
        stop = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        if step == 0:
            raise ValueError(f"{name} range step cannot be zero")
        result.extend(range(start, stop, step))
    if not result:
        raise ValueError(f"{name} selection cannot be empty")
    return list(dict.fromkeys(result))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired, independent fixed-target EE-tracking episodes for "
            "three arm IK/MPPI ablations."
        )
    )
    parser.add_argument(
        "--episode-seconds",
        type=float,
        default=10.0,
        help="Fixed duration of every target episode (default: 10.0)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Compatibility/testing override for steps per episode; "
            "overrides --episode-seconds"
        ),
    )
    parser.add_argument(
        "--success-hold-seconds",
        type=float,
        default=0.2,
        help=(
            "Continuous threshold dwell required for success "
            "(default: 0.2)"
        ),
    )
    parser.add_argument(
        "--steady-window-seconds",
        type=float,
        default=1.0,
        help="Final window used for steady-state error (default: 1.0)",
    )
    parser.add_argument(
        "--latency-warmup-steps",
        type=int,
        default=50,
        help="Initial update samples excluded from latency (default: 50)",
    )
    parser.add_argument(
        "--targets",
        default="all",
        help=(
            "Target indices: 'all', comma list, or start:stop[:step] "
            "(default: all)"
        ),
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help=(
            "Paired seeds: comma list or start:stop[:step] "
            "(default: 42)"
        ),
    )
    parser.add_argument(
        "--episode-cooldown-seconds",
        type=float,
        default=30.0,
        help=(
            "Idle time between consecutive episodes "
            "(default: 30.0; use 0 to disable)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for plots, manifest, summary, and trajectories "
            f"(default: {DEFAULT_OUTPUT_DIR})"
        ),
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=20,
        help="Rolling-median window for error curves (default: 20)",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Do not replay the selected paired episode after plotting",
    )
    parser.add_argument(
        "--replay-target",
        type=int,
        default=None,
        help="Target to replay (default: first selected target)",
    )
    parser.add_argument(
        "--replay-seed",
        type=int,
        default=None,
        help="Seed to replay (default: first selected seed)",
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
    if args.episode_seconds <= 0.0:
        parser.error("--episode-seconds must be positive")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if args.success_hold_seconds <= 0.0:
        parser.error("--success-hold-seconds must be positive")
    if args.steady_window_seconds <= 0.0:
        parser.error("--steady-window-seconds must be positive")
    if args.latency_warmup_steps < 0:
        parser.error("--latency-warmup-steps cannot be negative")
    if args.episode_cooldown_seconds < 0.0:
        parser.error("--episode-cooldown-seconds cannot be negative")
    if args.smooth_window < 1:
        parser.error("--smooth-window must be positive")
    if args.playback_speed <= 0.0:
        parser.error("--playback-speed must be positive")
    if args.replay_fps <= 0.0:
        parser.error("--replay-fps must be positive")
    return args


def save_manifest(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    target_indices: list[int],
    targets: np.ndarray,
    target_quaternions: np.ndarray,
    seeds: list[int],
    execution_order: list[EpisodeSpec],
    logs: list[EpisodeLog],
) -> Path:
    path = output_dir / "experiment_manifest.json"
    manifest = {
        "task": TASK,
        "rollout_mode": ROLLOUT_MODE,
        "modes": {
            "arm_ik_nominal": (
                "fixed IK arm; arm Q/R, EE, terminal, collision costs off"
            ),
            "arm_fixed_same_cost": (
                "fixed IK arm; objective identical to whole-body MPPI"
            ),
            "whole_body_mppi": (
                "leg and arm controls optimized with the full objective"
            ),
        },
        "targets": [
            {
                "index": index,
                "position": targets[index].tolist(),
                "quaternion": target_quaternions[index].tolist(),
            }
            for index in target_indices
        ],
        "seeds": seeds,
        "episode_seconds": args.episode_seconds,
        "steps_override": args.steps,
        "actual_steps": len(logs[0].time),
        "dt": logs[0].dt,
        "success_hold_seconds": args.success_hold_seconds,
        "steady_window_seconds": args.steady_window_seconds,
        "latency_warmup_steps": args.latency_warmup_steps,
        "n_samples": logs[0].n_samples,
        "horizon": logs[0].horizon,
        "episode_cooldown_seconds": args.episode_cooldown_seconds,
        "execution_order_policy": "target_then_seed_then_mode",
        "execution_order": [
            {
                "target_index": spec.target_index,
                "seed": spec.seed,
                "mode": spec.mode,
            }
            for spec in execution_order
        ],
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    task_data = get_task(TASK)
    targets = np.asarray(task_data["ee_goal_pos"], dtype=float)
    target_quaternions = np.asarray(
        task_data["ee_goal_quat"],
        dtype=float,
    )
    try:
        target_indices = (
            list(range(len(targets)))
            if args.targets == "all"
            else parse_integer_selection(
                args.targets,
                name="target",
            )
        )
        seeds = parse_integer_selection(args.seeds, name="seed")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    invalid_targets = [
        index
        for index in target_indices
        if index < 0 or index >= len(targets)
    ]
    if invalid_targets:
        raise SystemExit(
            f"Target indices out of range: {invalid_targets}; "
            f"valid range is 0..{len(targets) - 1}"
        )
    if any(seed < 0 for seed in seeds):
        raise SystemExit("Seeds must be non-negative")

    specs = [
        EpisodeSpec(
            target_index=target_index,
            target_position=tuple(targets[target_index]),
            target_quaternion=tuple(
                target_quaternions[target_index]
            ),
            seed=seed,
            mode=mode,
        )
        for target_index in target_indices
        for seed in seeds
        for mode in MODES
    ]

    logs: list[EpisodeLog] = []
    for episode_index, spec in enumerate(specs):
        log = run_episode(
            spec,
            episode_seconds=args.episode_seconds,
            steps_override=args.steps,
            success_hold_seconds=args.success_hold_seconds,
        )
        logs.append(log)
        print(f"saved trajectory: {save_trajectory(log, output_dir)}")
        has_next_episode = episode_index + 1 < len(specs)
        if has_next_episode and args.episode_cooldown_seconds > 0.0:
            next_spec = specs[episode_index + 1]
            print(
                f"cooldown: {args.episode_cooldown_seconds:g}s before "
                f"T{next_spec.target_index} seed={next_spec.seed} "
                f"{next_spec.mode}"
            )
            sleep(args.episode_cooldown_seconds)

    validate_fairness(logs)
    target_order = {
        target_index: position
        for position, target_index in enumerate(target_indices)
    }
    seed_order = {
        seed: position
        for position, seed in enumerate(seeds)
    }
    mode_order = {
        mode: position
        for position, mode in enumerate(MODES)
    }
    logs.sort(
        key=lambda log: (
            target_order[log.spec.target_index],
            seed_order[log.spec.seed],
            mode_order[log.spec.mode],
        )
    )

    summary_rows = [
        episode_summary(
            log,
            steady_window_seconds=args.steady_window_seconds,
            latency_warmup_steps=args.latency_warmup_steps,
        )
        for log in logs
    ]
    for log, row in zip(logs, summary_rows):
        print_episode_summary(log, row)
    print(
        f"saved episode summary: "
        f"{save_episode_summary(summary_rows, output_dir)}"
    )
    manifest_path = save_manifest(
        output_dir=output_dir,
        args=args,
        target_indices=target_indices,
        targets=targets,
        target_quaternions=target_quaternions,
        seeds=seeds,
        execution_order=specs,
        logs=logs,
    )
    print(f"saved manifest: {manifest_path}")

    plot_paths = {
        "target error curves": output_dir / "target_error_curves.png",
        "success and reach time": (
            output_dir / "success_and_reach_time.png"
        ),
        "tracking accuracy": output_dir / "tracking_accuracy.png",
        "stability and safety": (
            output_dir / "stability_and_safety.png"
        ),
        "computation time": output_dir / "computation_time.png",
    }
    plot_target_error_curves(
        logs,
        target_indices,
        plot_paths["target error curves"],
        args.smooth_window,
    )
    plot_success_and_reach_time(
        summary_rows,
        target_indices,
        plot_paths["success and reach time"],
    )
    plot_tracking_accuracy(
        summary_rows,
        target_indices,
        plot_paths["tracking accuracy"],
    )
    plot_stability_and_safety(
        summary_rows,
        target_indices,
        logs[0].collision_hard_distance,
        plot_paths["stability and safety"],
    )
    plot_computation_time(
        summary_rows,
        target_indices,
        plot_paths["computation time"],
    )
    for name, path in plot_paths.items():
        print(f"saved {name}: {path}")

    replay_target = (
        target_indices[0]
        if args.replay_target is None
        else args.replay_target
    )
    replay_seed = (
        seeds[0]
        if args.replay_seed is None
        else args.replay_seed
    )
    if replay_target not in target_indices:
        raise SystemExit(
            f"--replay-target {replay_target} was not selected"
        )
    if replay_seed not in seeds:
        raise SystemExit(
            f"--replay-seed {replay_seed} was not selected"
        )
    if not args.no_viewer:
        print(
            f"starting MuJoCo replay: T{replay_target}, "
            f"seed={replay_seed}"
        )
        replay_episodes(
            logs,
            target_index=replay_target,
            seed=replay_seed,
            playback_speed=args.playback_speed,
            replay_fps=args.replay_fps,
        )


if __name__ == "__main__":
    main()
