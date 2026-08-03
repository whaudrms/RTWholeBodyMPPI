"""Compare CEM-guided push-box control with and without arm MPPI.

Each ``(seed, mode)`` episode starts from a fresh controller and MuJoCo
simulator. Both modes retain the same CEM base planner, adaptive gait, leg
MPPI, box objective, initial state, and random seeds. The only ablation is
whether the arm is sampled and optimized by MPPI or follows its online IK
nominal directly.

Examples
--------
    conda run -n wb-mppi python legged_mani/scripts/compare_push_box.py --no-viewer
    conda run -n wb-mppi python legged_mani/scripts/compare_push_box.py --seeds 0:5 --no-viewer
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

from mani_mppi.control.controllers.mppi_push_box import MPPI
from mani_mppi.interface.simulator import Simulator
from mani_mppi.utils.tasks import get_task
from mani_mppi.utils.transforms import batch_world_to_local_velocity


TASK = "push_box"
ROLLOUT_MODE = "safe_spline"
MODES = ("arm_ik_nominal", "whole_body_mppi")
LABELS = {
    "arm_ik_nominal": "Arm IK nominal",
    "whole_body_mppi": "Whole-body MPPI",
}
COLORS = {
    "arm_ik_nominal": "#E69F00",
    "whole_body_mppi": "#0072B2",
}
DEFAULT_OUTPUT_DIR = (
    PACKAGE_ROOT / "mani_mppi" / "analysis" / "compare_push_box"
)


@dataclass(frozen=True)
class EpisodeSpec:
    seed: int
    mode: str


@dataclass
class EpisodeLog:
    spec: EpisodeSpec
    time: np.ndarray
    box_position: np.ndarray
    box_error: np.ndarray
    box_progress: np.ndarray
    box_lateral_error: np.ndarray
    box_tilt: np.ndarray
    ee_contact_error: np.ndarray
    contact_engaged: np.ndarray
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
    initial_box_position: np.ndarray
    box_target: np.ndarray
    success: bool
    success_time_s: float
    first_contact_time_s: float
    initial_plan_xy: np.ndarray
    initial_plan_height: float
    cem_seed: int
    cem_plan_count: int
    cem_compute_seconds: float
    collision_hard_distance: float
    dt: float
    n_samples: int
    horizon: int
    base_noise_sigma: np.ndarray
    state_cost_weights: np.ndarray
    control_cost_weights: np.ndarray
    box_cost_weights: np.ndarray
    box_orientation_weight: float
    box_max_tilt: float


def parse_integer_selection(value: str, *, name: str) -> list[int]:
    """Parse comma-separated integers and ``start:stop[:step]`` ranges."""
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


def box_tilt_angle(quaternion: np.ndarray) -> float:
    """Return box tilt from world-up while ignoring its yaw."""
    quaternion = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(quaternion)
    if norm <= 1e-12:
        return float("nan")
    _, x, y, _ = quaternion / norm
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return float(np.arccos(np.clip(up_z, -1.0, 1.0)))


def arm_actuator_indices(agent: MPPI) -> np.ndarray:
    actuator_joint_ids = agent.model.actuator_trnid[:, 0]
    indices = np.flatnonzero(
        np.isin(actuator_joint_ids, agent.arm_kinematics.arm_joint_ids)
    )
    if len(indices) != 4:
        raise ValueError(f"Expected four arm actuators, got {indices}")
    return indices


def arm_clearance(agent: MPPI, observation: np.ndarray) -> float:
    state = np.asarray(observation, dtype=float)[None, :]
    arm_positions, _ = agent._batch_arm_fk(state)
    clearance, _ = agent._arm_torso_clearance(state, arm_positions)
    return float(np.min(clearance))


def push_cost_without_arm_objective(
    agent: MPPI,
    states: np.ndarray,
    actions: np.ndarray,
    joints_ref: np.ndarray,
    body_ref: np.ndarray,
    rollout_sensors: np.ndarray | None = None,
) -> np.ndarray:
    """Keep leg/base and box objectives while omitting every arm objective."""
    del rollout_sensors
    num_samples, horizon = states.shape[:2]
    qpos = states[:, :, :agent.model.nq]
    qvel = states[:, :, agent.model.nq:]
    robot_states = np.concatenate(
        (
            qpos[:, :, agent.robot_qpos_indices],
            qvel[:, :, agent.robot_dof_indices],
        ),
        axis=2,
    )
    box_states = np.concatenate(
        (
            qpos[:, :, agent.box_qpos_adr:agent.box_qpos_adr + 7],
            qvel[:, :, agent.box_dof_adr:agent.box_dof_adr + 6],
        ),
        axis=2,
    )
    flat_robot_states = robot_states.reshape(
        -1, robot_states.shape[-1]
    ).copy()
    flat_box_states = box_states.reshape(-1, box_states.shape[-1])
    flat_actions = actions.reshape(-1, actions.shape[-1])

    body_refs = np.repeat(body_ref[None, :], len(flat_robot_states), axis=0)
    tiled_joints = np.tile(
        joints_ref.T, (num_samples, 1, 1)
    ).reshape(-1, joints_ref.shape[0])
    base_velocity_ref = np.zeros((len(flat_robot_states), 6), dtype=float)
    base_velocity_ref[:, :2] = body_refs[:, 7:9]
    robot_ref = np.concatenate(
        (
            body_refs[:, :7],
            tiled_joints[:, :agent.act_dim],
            base_velocity_ref,
            tiled_joints[:, agent.act_dim:],
        ),
        axis=1,
    )
    flat_robot_states[:, 23:26] = batch_world_to_local_velocity(
        flat_robot_states[:, 3:7], flat_robot_states[:, 23:26]
    )

    robot_cost = agent.quadruped_cost_np(
        flat_robot_states, flat_actions, robot_ref
    )
    box_cost = agent._box_position_cost(flat_box_states)
    box_orientation_cost = agent._box_orientation_cost(flat_box_states)
    agent.collision_valid_rollouts = np.ones(num_samples, dtype=bool)
    agent.collision_min_clearance = np.full(num_samples, np.inf)
    agent.collision_exact_evaluations = 0
    return (
        robot_cost + box_cost + box_orientation_cost
    ).reshape(num_samples, horizon).sum(axis=1)


def configure_mode(agent: MPPI, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Apply only the arm-search ablation and return arm/leg actuator IDs."""
    arm_indices = arm_actuator_indices(agent)
    leg_indices = np.setdiff1d(
        np.arange(agent.act_dim, dtype=int), arm_indices
    )
    if mode == "whole_body_mppi":
        return arm_indices, leg_indices
    if mode != "arm_ik_nominal":
        raise ValueError(f"Unknown comparison mode: {mode}")

    agent.base_noise_sigma[arm_indices] = 0.0
    agent.set_noise_for_gait(agent.default_gait)
    agent.gait_correction[:, arm_indices] = 0.0

    arm_q_weights = 6 + arm_indices
    arm_dq_weights = 28 + arm_indices
    agent.state_cost_weights[arm_q_weights] = 0.0
    agent.state_cost_weights[arm_dq_weights] = 0.0
    agent.control_cost_weights[arm_indices] = 0.0
    agent.ee_position_weight = 0.0
    agent.ee_orientation_weight = 0.0
    agent.ee_terminal_scale = 0.0
    agent.collision_enabled = False
    agent.capture_rollout_sensors = False
    agent.sensor_rollouts = None
    agent.calculate_total_cost = lambda *args, **kwargs: (
        push_cost_without_arm_objective(agent, *args, **kwargs)
    )
    return arm_indices, leg_indices


def complete_pending_cem(
    agent: MPPI,
    observation: np.ndarray,
) -> tuple[float, bool] | None:
    """Apply a pending CEM result at a deterministic control boundary."""
    future = agent._planner_future
    if future is None:
        return None
    completed = future.result()
    elapsed = float(completed[-1])
    applied = bool(agent._apply_completed_cem_plan(observation))
    return elapsed, applied


def initialize_cem_plan(
    agent: MPPI,
    observation: np.ndarray,
) -> tuple[float, bool]:
    """Finish and apply the paired initial CEM plan before simulation time."""
    if not agent._submit_cem_plan(observation):
        raise RuntimeError("Failed to submit the initial push-box CEM plan")
    result = complete_pending_cem(agent, observation)
    if result is None or not agent._has_active_plan:
        raise RuntimeError("Initial push-box CEM plan was not applied")
    return result


def moving_contact_error(agent: MPPI, observation: np.ndarray) -> float:
    _, box_state = agent._split_observation(observation)
    target = agent._batch_contact_targets(box_state[None, :])[0]
    ee_position, _ = agent._ee_pose(observation)
    return float(np.linalg.norm(ee_position - target))


def run_episode(
    spec: EpisodeSpec,
    *,
    episode_seconds: float,
    steps_override: int | None,
) -> EpisodeLog:
    task_data = get_task(TASK)
    sim_path = PACKAGE_ROOT / "mani_mppi" / task_data["sim_path"]
    agent = MPPI(task=TASK, rollout_mode=ROLLOUT_MODE)
    dt = float(agent.model.opt.timestep)
    steps = (
        int(steps_override)
        if steps_override is not None
        else max(1, int(round(episode_seconds / dt)))
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
    initial_observation = np.concatenate((initial_qpos, initial_qvel))
    _, initial_box_state = agent._split_observation(initial_observation)
    initial_box_position = initial_box_state[:3].copy()
    box_target = agent.x_box_ref.copy()
    push_delta = box_target[:2] - initial_box_position[:2]
    push_distance = float(np.linalg.norm(push_delta))
    push_direction = (
        push_delta / push_distance
        if push_distance > 1e-12
        else np.array([1.0, 0.0])
    )

    arm_indices, leg_indices = configure_mode(agent, spec.mode)
    agent.random_generator = np.random.default_rng(spec.seed)
    cem_seed = spec.seed + 1000
    agent.base_pose_planner.optimizer.random_generator = (
        np.random.default_rng(cem_seed)
    )
    initial_cem_elapsed, initial_cem_applied = initialize_cem_plan(
        agent, initial_observation
    )
    initial_plan_xy = agent.planned_base_xy.copy()
    initial_plan_height = float(agent.planned_base_height)
    cem_compute_seconds = initial_cem_elapsed
    cem_plan_count = int(initial_cem_applied)

    time_log = np.empty(steps)
    box_position_log = np.empty((steps, 3))
    box_error_log = np.empty(steps)
    box_progress_log = np.empty(steps)
    box_lateral_error_log = np.empty(steps)
    box_tilt_log = np.empty(steps)
    ee_contact_error_log = np.empty(steps)
    contact_engaged_log = np.empty(steps, dtype=bool)
    base_position_log = np.empty((steps, 3))
    roll_pitch_log = np.empty((steps, 2))
    clearance_log = np.empty(steps)
    leg_action_delta_log = np.empty(steps)
    arm_action_delta_log = np.empty(steps)
    update_ms_log = np.empty(steps)
    qpos_log = np.empty((steps, simulator.model.nq))
    qvel_log = np.empty((steps, simulator.model.nv))
    previous_action = None
    success_time = float("nan")
    first_contact_time = float("nan")

    description = f"seed={spec.seed} {spec.mode}"
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
            completed = complete_pending_cem(agent, observation)
            if completed is not None:
                elapsed, applied = completed
                cem_compute_seconds += elapsed
                cem_plan_count += int(applied)

            if spec.mode == "arm_ik_nominal":
                agent.gait_correction[:, arm_indices] = 0.0
            update_start = perf_counter()
            action = agent.update(observation)
            update_ms_log[step] = 1000.0 * (
                perf_counter() - update_start
            )
            simulator.step(action)

            result = np.concatenate(
                (simulator.data.qpos, simulator.data.qvel)
            )
            _, box_state = agent._split_observation(result)
            box_position = box_state[:3]
            box_error = float(
                np.linalg.norm(box_position[:2] - box_target[:2])
            )
            displacement = box_position[:2] - initial_box_position[:2]
            progress_distance = float(np.dot(displacement, push_direction))
            lateral_vector = displacement - progress_distance * push_direction

            time_log[step] = simulator.data.time
            box_position_log[step] = box_position
            box_error_log[step] = box_error
            box_progress_log[step] = progress_distance
            box_lateral_error_log[step] = np.linalg.norm(lateral_vector)
            box_tilt_log[step] = box_tilt_angle(box_state[3:7])
            ee_contact_error_log[step] = moving_contact_error(agent, result)
            contact_engaged_log[step] = agent.contact_engaged
            base_position_log[step] = simulator.data.qpos[:3]
            roll_pitch_log[step] = quaternion_to_roll_pitch(
                simulator.data.qpos[3:7]
            )
            clearance_log[step] = arm_clearance(agent, result)
            if previous_action is None:
                leg_action_delta_log[step] = 0.0
                arm_action_delta_log[step] = 0.0
            else:
                action_delta = action - previous_action
                leg_action_delta_log[step] = np.linalg.norm(
                    action_delta[leg_indices]
                )
                arm_action_delta_log[step] = np.linalg.norm(
                    action_delta[arm_indices]
                )
            previous_action = action.copy()
            qpos_log[step] = simulator.data.qpos
            qvel_log[step] = simulator.data.qvel

            if agent.contact_engaged and np.isnan(first_contact_time):
                first_contact_time = float(simulator.data.time)
            if agent.goal_reached(result):
                if np.isnan(success_time):
                    success_time = float(simulator.data.time)
                agent.next_goal()

            if (step + 1) % 100 == 0:
                progress.set_postfix(
                    box_cm=f"{100.0 * box_error:.1f}",
                    contact=bool(agent.contact_engaged),
                    hz=f"{1000.0 / np.mean(update_ms_log[:step + 1]):.1f}",
                )

        final_observation = np.concatenate(
            (simulator.data.qpos, simulator.data.qvel)
        )
        completed = complete_pending_cem(agent, final_observation)
        if completed is not None:
            elapsed, applied = completed
            cem_compute_seconds += elapsed
            cem_plan_count += int(applied)
    finally:
        agent.close()

    return EpisodeLog(
        spec=spec,
        time=time_log,
        box_position=box_position_log,
        box_error=box_error_log,
        box_progress=box_progress_log,
        box_lateral_error=box_lateral_error_log,
        box_tilt=box_tilt_log,
        ee_contact_error=ee_contact_error_log,
        contact_engaged=contact_engaged_log,
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
        initial_box_position=initial_box_position,
        box_target=box_target,
        success=not np.isnan(success_time),
        success_time_s=success_time,
        first_contact_time_s=first_contact_time,
        initial_plan_xy=initial_plan_xy,
        initial_plan_height=initial_plan_height,
        cem_seed=cem_seed,
        cem_plan_count=cem_plan_count,
        cem_compute_seconds=cem_compute_seconds,
        collision_hard_distance=float(agent.collision_hard_distance),
        dt=dt,
        n_samples=int(agent.n_samples),
        horizon=int(agent.horizon),
        base_noise_sigma=agent.base_noise_sigma.copy(),
        state_cost_weights=agent.state_cost_weights.copy(),
        control_cost_weights=agent.control_cost_weights.copy(),
        box_cost_weights=agent.box_cost_weights.copy(),
        box_orientation_weight=float(agent.box_orientation_weight),
        box_max_tilt=float(agent.box_max_tilt),
    )


def validate_fairness(logs: list[EpisodeLog]) -> None:
    """Verify paired runs differ only in their intended arm ablation."""
    groups: dict[int, dict[str, EpisodeLog]] = {}
    for log in logs:
        groups.setdefault(log.spec.seed, {})[log.spec.mode] = log
    for seed, by_mode in groups.items():
        missing = set(MODES) - set(by_mode)
        if missing:
            raise RuntimeError(f"Seed {seed} is missing modes: {sorted(missing)}")
        nominal = by_mode["arm_ik_nominal"]
        whole = by_mode["whole_body_mppi"]
        for name in (
            "initial_qpos",
            "initial_qvel",
            "initial_box_position",
            "box_target",
            "initial_plan_xy",
            "box_cost_weights",
        ):
            np.testing.assert_array_equal(
                getattr(nominal, name),
                getattr(whole, name),
                err_msg=f"{name} differs for paired seed {seed}",
            )
        if (
            nominal.initial_plan_height != whole.initial_plan_height
            or nominal.cem_seed != whole.cem_seed
            or nominal.dt != whole.dt
            or nominal.n_samples != whole.n_samples
            or nominal.horizon != whole.horizon
            or nominal.box_orientation_weight
            != whole.box_orientation_weight
            or nominal.box_max_tilt != whole.box_max_tilt
        ):
            raise RuntimeError(f"Common configuration differs for seed {seed}")

        arm_indices = np.arange(len(nominal.base_noise_sigma) - 4,
                                len(nominal.base_noise_sigma))
        leg_indices = np.arange(len(nominal.base_noise_sigma) - 4)
        np.testing.assert_array_equal(
            nominal.base_noise_sigma[leg_indices],
            whole.base_noise_sigma[leg_indices],
            err_msg=f"Leg exploration differs for seed {seed}",
        )
        if np.any(nominal.base_noise_sigma[arm_indices] != 0.0):
            raise RuntimeError(f"Nominal arm has nonzero noise for seed {seed}")
        np.testing.assert_array_equal(
            nominal.control_cost_weights[leg_indices],
            whole.control_cost_weights[leg_indices],
            err_msg=f"Leg control objective differs for seed {seed}",
        )


def trajectory_path(log: EpisodeLog, output_dir: Path) -> Path:
    target_dir = output_dir / "trajectories"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"seed_{log.spec.seed:04d}_{log.spec.mode}.tsv"


def save_trajectory(log: EpisodeLog, output_dir: Path) -> Path:
    path = trajectory_path(log, output_dir)
    data = np.column_stack(
        (
            log.time,
            log.box_position,
            log.box_error,
            log.box_progress,
            log.box_lateral_error,
            np.rad2deg(log.box_tilt),
            log.ee_contact_error,
            log.contact_engaged.astype(int),
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
            "time_s\tbox_x_m\tbox_y_m\tbox_z_m\tbox_error_m\t"
            "box_progress_m\tbox_lateral_error_m\tbox_tilt_deg\t"
            "ee_contact_error_m\tcontact_engaged\tbase_x_m\tbase_y_m\t"
            "base_z_m\troll_deg\tpitch_deg\tclearance_m\t"
            "leg_action_delta_l2\tarm_action_delta_l2\tupdate_ms"
        ),
        comments="",
    )
    return path


def episode_summary(
    log: EpisodeLog,
    *,
    latency_warmup_steps: int,
) -> dict[str, object]:
    latency_start = min(latency_warmup_steps, len(log.update_ms) - 1)
    latency = log.update_ms[latency_start:]
    attitude = np.max(np.abs(np.rad2deg(log.roll_pitch)), axis=1)
    initial_error = float(
        np.linalg.norm(
            log.initial_box_position[:2] - log.box_target[:2]
        )
    )
    contact_errors = log.ee_contact_error[log.contact_engaged]
    contact_rmse = (
        float(np.sqrt(np.mean(contact_errors**2)))
        if len(contact_errors)
        else float("nan")
    )
    return {
        "seed": log.spec.seed,
        "mode": log.spec.mode,
        "success": int(log.success),
        "success_time_s": log.success_time_s,
        "episode_duration_s": float(log.time[-1]),
        "initial_box_error_m": initial_error,
        "final_box_error_m": float(log.box_error[-1]),
        "minimum_box_error_m": float(np.min(log.box_error)),
        "box_error_auc_m_s": float(
            np.trapezoid(
                np.concatenate(([initial_error], log.box_error)),
                x=np.concatenate(([0.0], log.time)),
            )
        ),
        "final_box_progress_m": float(log.box_progress[-1]),
        "maximum_lateral_error_m": float(np.max(log.box_lateral_error)),
        "maximum_box_tilt_deg": float(np.max(np.rad2deg(log.box_tilt))),
        "first_contact_time_s": log.first_contact_time_s,
        "contact_fraction": float(np.mean(log.contact_engaged)),
        "contact_ee_rmse_m": contact_rmse,
        "minimum_clearance_m": float(np.min(log.clearance)),
        "clearance_violation_fraction": float(
            np.mean(log.clearance < log.collision_hard_distance)
        ),
        "minimum_base_height_m": float(
            min(log.initial_qpos[2], np.min(log.base_position[:, 2]))
        ),
        "maximum_abs_roll_pitch_deg": float(np.max(attitude)),
        "mean_leg_action_delta": float(np.mean(log.leg_action_delta)),
        "mean_arm_action_delta": float(np.mean(log.arm_action_delta)),
        "latency_median_ms": float(np.median(latency)),
        "latency_p95_ms": float(np.percentile(latency, 95)),
        "cem_plan_count": log.cem_plan_count,
        "cem_compute_seconds": log.cem_compute_seconds,
    }


def save_summary(rows: list[dict[str, object]], output_dir: Path) -> Path:
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


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or window <= 1:
        return values.copy()
    window = min(int(window), len(values))
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, window
    )
    return np.median(windows, axis=-1)


def mode_handles() -> list[Line2D]:
    return [
        Line2D(
            [0], [0], color=COLORS[mode], linewidth=2.5,
            label=LABELS[mode],
        )
        for mode in MODES
    ]


def plot_time_series(
    logs: list[EpisodeLog],
    output: Path,
    smooth_window: int,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    series = (
        ("box_error", "Box XY error", "error [m]"),
        ("ee_contact_error", "EE-to-moving-contact target", "error [m]"),
        ("box_tilt", "Box tilt", "tilt [deg]"),
        ("clearance", "Arm-to-torso clearance", "clearance [m]"),
    )
    for axis, (attribute, title, ylabel) in zip(axes.ravel(), series):
        for mode in MODES:
            mode_logs = [log for log in logs if log.spec.mode == mode]
            curves = []
            for log in mode_logs:
                values = getattr(log, attribute)
                if attribute == "box_tilt":
                    values = np.rad2deg(values)
                values = rolling_median(values, smooth_window)
                curves.append(values)
                axis.plot(
                    log.time, values, color=COLORS[mode],
                    alpha=0.16, linewidth=0.9,
                )
            median = np.median(np.stack(curves), axis=0)
            axis.plot(
                mode_logs[0].time, median, color=COLORS[mode],
                linewidth=2.5,
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("simulation time [s]")
    axes[1, 1].set_xlabel("simulation time [s]")
    figure.legend(
        handles=mode_handles(), loc="upper center", ncol=2,
        frameon=False, bbox_to_anchor=(0.5, 0.97),
    )
    figure.suptitle(
        "CEM-guided push-box: IK nominal vs whole-body MPPI",
        fontsize=15, y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_box_paths(logs: list[EpisodeLog], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    for mode in MODES:
        for log in (item for item in logs if item.spec.mode == mode):
            axis.plot(
                log.box_position[:, 0], log.box_position[:, 1],
                color=COLORS[mode], alpha=0.25, linewidth=1.1,
            )
        mode_logs = [log for log in logs if log.spec.mode == mode]
        median_path = np.median(
            np.stack([log.box_position[:, :2] for log in mode_logs]),
            axis=0,
        )
        axis.plot(
            median_path[:, 0], median_path[:, 1],
            color=COLORS[mode], linewidth=2.6,
        )
    reference = logs[0]
    axis.scatter(
        reference.initial_box_position[0],
        reference.initial_box_position[1],
        marker="o", s=90, color="black", label="Box start", zorder=5,
    )
    axis.scatter(
        reference.box_target[0], reference.box_target[1],
        marker="*", s=180, color="#D55E00", label="Box goal", zorder=5,
    )
    handles = mode_handles() + [
        Line2D([0], [0], marker="o", color="black", linestyle="None",
               label="Box start"),
        Line2D([0], [0], marker="*", color="#D55E00", linestyle="None",
               markersize=11, label="Box goal"),
    ]
    axis.legend(handles=handles, frameon=False)
    axis.set_title("Box XY trajectories")
    axis.set_xlabel("world x [m]")
    axis.set_ylabel("world y [m]")
    axis.axis("equal")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_summary(rows: list[dict[str, object]], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = (
        ("success", "Success rate", "fraction"),
        ("final_box_error_m", "Final box error", "error [m]"),
        ("contact_ee_rmse_m", "Contact EE RMSE", "error [m]"),
        ("latency_median_ms", "Controller latency", "median [ms]"),
    )
    x = np.arange(len(MODES))
    for axis, (key, title, ylabel) in zip(axes.ravel(), metrics):
        means = []
        spreads = []
        for mode in MODES:
            values = np.asarray(
                [float(row[key]) for row in rows if row["mode"] == mode],
                dtype=float,
            )
            finite = values[np.isfinite(values)]
            means.append(float(np.mean(finite)) if len(finite) else np.nan)
            spreads.append(float(np.std(finite)) if len(finite) > 1 else 0.0)
        axis.bar(
            x, means, yerr=spreads,
            color=[COLORS[mode] for mode in MODES],
            alpha=0.85, capsize=4,
        )
        axis.set_xticks(x, [LABELS[mode] for mode in MODES], rotation=8)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Paired push-box episode summary", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_manifest(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    seeds: list[int],
    specs: list[EpisodeSpec],
    logs: list[EpisodeLog],
) -> Path:
    path = output_dir / "experiment_manifest.json"
    task_data = get_task(TASK)
    manifest = {
        "task": TASK,
        "box_target": np.asarray(task_data["box_pos_ref"], dtype=float).tolist(),
        "modes": {
            "arm_ik_nominal": (
                "CEM and leg/base MPPI retained; arm is online IK nominal "
                "with arm exploration and arm objectives disabled"
            ),
            "whole_body_mppi": (
                "CEM and full leg/arm whole-body MPPI objective"
            ),
        },
        "rollout_mode": ROLLOUT_MODE,
        "seeds": seeds,
        "mppi_seed_policy": "episode seed",
        "cem_seed_policy": "episode seed + 1000",
        "cem_timing_policy": (
            "initial plan before t=0; later plans applied at next control boundary"
        ),
        "episode_seconds": args.episode_seconds,
        "steps_override": args.steps,
        "actual_steps": len(logs[0].time),
        "dt": logs[0].dt,
        "n_samples": logs[0].n_samples,
        "horizon": logs[0].horizon,
        "latency_warmup_steps": args.latency_warmup_steps,
        "episode_cooldown_seconds": args.episode_cooldown_seconds,
        "execution_order_policy": "seed_then_mode",
        "execution_order": [
            {"seed": spec.seed, "mode": spec.mode} for spec in specs
        ],
        "paired_initial_cem_plans": [
            {
                "seed": log.spec.seed,
                "mode": log.spec.mode,
                "base_xy": log.initial_plan_xy.tolist(),
                "base_height": log.initial_plan_height,
            }
            for log in logs
        ],
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return path


def replay_episodes(
    logs: list[EpisodeLog],
    *,
    seed: int,
    playback_speed: float,
    replay_fps: float,
) -> None:
    import mujoco_viewer

    selected = [
        log for mode in MODES for log in logs
        if log.spec.seed == seed and log.spec.mode == mode
    ]
    if len(selected) != len(MODES):
        raise ValueError(f"Replay seed {seed} does not contain both modes")
    task_data = get_task(TASK)
    model_path = PACKAGE_ROOT / "mani_mppi" / task_data["sim_path"]
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=True)
    frame_stride = max(
        1, round(1.0 / (selected[0].dt * replay_fps))
    )
    frame_period = frame_stride * selected[0].dt / playback_speed
    try:
        for log in selected:
            for frame in range(0, len(log.time), frame_stride):
                if not viewer.is_alive:
                    return
                frame_start = perf_counter()
                data.qpos[:] = log.qpos[frame]
                data.qvel[:] = log.qvel[frame]
                data.time = log.time[frame]
                mujoco.mj_forward(model, data)
                viewer.add_marker(
                    pos=log.box_target,
                    size=[0.06, 0.06, 0.06],
                    rgba=[1.0, 0.15, 0.05, 0.9],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    label="Box goal",
                )
                viewer.add_marker(
                    pos=data.qpos[:3] + np.array([0.0, 0.0, 0.7]),
                    size=[0.001, 0.001, 0.001],
                    rgba=[0.0, 0.0, 0.0, 0.0],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    label=(
                        f"{LABELS[log.spec.mode]} | seed={seed} | "
                        f"t={log.time[frame]:.2f}s"
                    ),
                )
                viewer.render()
                remaining = frame_period - (perf_counter() - frame_start)
                if remaining > 0.0:
                    sleep(remaining)
            if viewer.is_alive:
                sleep(0.5 / playback_speed)
    finally:
        viewer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired CEM-guided push-box episodes for arm IK nominal "
            "and whole-body MPPI."
        )
    )
    parser.add_argument(
        "--episode-seconds", type=float, default=30.0,
        help="Fixed duration of every mode episode (default: 30.0)",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Testing override for steps; overrides --episode-seconds",
    )
    parser.add_argument(
        "--seeds", default="42",
        help="Paired seeds: comma list or start:stop[:step] (default: 42)",
    )
    parser.add_argument(
        "--episode-cooldown-seconds", type=float, default=30.0,
        help=(
            "Idle time between consecutive episodes "
            "(default: 30.0; use 0 to disable)"
        ),
    )
    parser.add_argument(
        "--latency-warmup-steps", type=int, default=50,
        help="Initial update samples excluded from latency (default: 50)",
    )
    parser.add_argument(
        "--smooth-window", type=int, default=20,
        help="Rolling-median window for time-series plots (default: 20)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-viewer", action="store_true",
        help="Do not replay a paired seed after plotting",
    )
    parser.add_argument(
        "--replay-seed", type=int, default=None,
        help="Seed to replay (default: first selected seed)",
    )
    parser.add_argument(
        "--playback-speed", type=float, default=1.0,
        help="MuJoCo replay speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--replay-fps", type=float, default=50.0,
        help="Maximum replay render rate (default: 50)",
    )
    args = parser.parse_args()
    if args.episode_seconds <= 0.0:
        parser.error("--episode-seconds must be positive")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if args.episode_cooldown_seconds < 0.0:
        parser.error("--episode-cooldown-seconds cannot be negative")
    if args.latency_warmup_steps < 0:
        parser.error("--latency-warmup-steps cannot be negative")
    if args.smooth_window < 1:
        parser.error("--smooth-window must be positive")
    if args.playback_speed <= 0.0 or args.replay_fps <= 0.0:
        parser.error("Replay speed and FPS must be positive")
    return args


def main() -> None:
    args = parse_args()
    try:
        seeds = parse_integer_selection(args.seeds, name="seed")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if any(seed < 0 for seed in seeds):
        raise SystemExit("Seeds must be non-negative")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        EpisodeSpec(seed=seed, mode=mode)
        for seed in seeds
        for mode in MODES
    ]
    logs: list[EpisodeLog] = []
    for episode_index, spec in enumerate(specs):
        log = run_episode(
            spec,
            episode_seconds=args.episode_seconds,
            steps_override=args.steps,
        )
        logs.append(log)
        print(f"saved trajectory: {save_trajectory(log, output_dir)}")
        if (
            episode_index + 1 < len(specs)
            and args.episode_cooldown_seconds > 0.0
        ):
            next_spec = specs[episode_index + 1]
            print(
                f"cooldown: {args.episode_cooldown_seconds:g}s before "
                f"seed={next_spec.seed} {next_spec.mode}"
            )
            sleep(args.episode_cooldown_seconds)

    validate_fairness(logs)
    rows = [
        episode_summary(
            log, latency_warmup_steps=args.latency_warmup_steps
        )
        for log in logs
    ]
    for row in rows:
        success_time = (
            f"{float(row['success_time_s']):.2f}s"
            if int(row["success"])
            else "timeout"
        )
        print(
            f"seed={row['seed']} {row['mode']}: "
            f"success={success_time}, "
            f"final_box_error={float(row['final_box_error_m']):.3f}m, "
            f"progress={float(row['final_box_progress_m']):.3f}m, "
            f"contact={100.0 * float(row['contact_fraction']):.1f}%, "
            f"latency_p95={float(row['latency_p95_ms']):.1f}ms"
        )
    print(f"saved summary: {save_summary(rows, output_dir)}")
    print(
        "saved manifest: "
        f"{save_manifest(output_dir=output_dir, args=args, seeds=seeds, specs=specs, logs=logs)}"
    )

    plot_paths = {
        "time series": output_dir / "push_box_time_series.png",
        "box paths": output_dir / "push_box_xy_paths.png",
        "summary": output_dir / "push_box_summary.png",
    }
    plot_time_series(logs, plot_paths["time series"], args.smooth_window)
    plot_box_paths(logs, plot_paths["box paths"])
    plot_summary(rows, plot_paths["summary"])
    for name, path in plot_paths.items():
        print(f"saved {name}: {path}")

    replay_seed = seeds[0] if args.replay_seed is None else args.replay_seed
    if replay_seed not in seeds:
        raise SystemExit(f"--replay-seed {replay_seed} was not selected")
    if not args.no_viewer:
        replay_episodes(
            logs,
            seed=replay_seed,
            playback_speed=args.playback_speed,
            replay_fps=args.replay_fps,
        )


if __name__ == "__main__":
    main()
