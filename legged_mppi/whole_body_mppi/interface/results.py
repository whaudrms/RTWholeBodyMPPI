"""Per-run numeric artifacts and headless plots for controller comparisons."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


OUTPUT_ROOT = Path(__file__).resolve().parents[1] / 'outputs'


def statistics(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    return dict(mean=float(values.mean()), rms=float(np.sqrt(np.mean(values**2))),
                p95=float(np.percentile(values, 95)), max=float(values.max()), final=float(values[-1]))


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value)}')


def save_results(sim, controller, output_root=None):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder = Path(output_root or OUTPUT_ROOT) / controller / sim.agent.task / f'{stamp}_{sim.robot}'
    folder.mkdir(parents=True, exist_ok=False)
    n = sim.recorded_steps
    t = sim.time[:n]
    arrays = {name: getattr(sim, name)[..., :n] for name in (
        'time', 'qpos', 'qvel', 'ctrl', 'cost', 'solve_seconds', 'cost_seconds',
        'plan_age_seconds', 'plan_application_delay_seconds', 'sync_lag_seconds', 'plan_exhausted',
        'body_targets', 'body_velocity_commands', 'body_velocity_errors', 'body_position_errors',
        'ee_errors', 'ee_positions', 'ee_targets', 'ee_world_targets', 'ee_hold_status',
        'ee_waypoint_indices', 'ee_velocities', 'ee_velocity_commands', 'command_phase_indices') if getattr(sim, name, None) is not None}
    if sim.velocity_task:
        arrays.pop('body_targets')
        arrays.pop('body_position_errors')
        arrays['body_height_targets'] = sim.body_targets[2, :n]
    np.savez_compressed(folder / 'trajectory.npz', **arrays)
    columns = dict(time_s=t, solve_ms=sim.solve_seconds[:n]*1000,
                   cost_evaluation_ms=sim.cost_seconds[:n]*1000,
                   predicted_horizon_cost=sim.cost[0, :n], body_position_error_m=sim.body_position_errors[:n],
                   body_velocity_error_m_s=np.linalg.norm(sim.body_velocity_errors[:, :n], axis=0))
    columns.update(plan_age_ms=sim.plan_age_seconds[:n]*1000,
                   plan_application_delay_ms=sim.plan_application_delay_seconds[:n]*1000,
                   sync_lag_ms=sim.sync_lag_seconds[:n]*1000,
                   plan_exhausted=sim.plan_exhausted[:n].astype(int))
    if sim.velocity_task:
        columns.pop('body_position_error_m')
        columns['command_phase_index'] = sim.command_phase_indices[:n]
    if sim.ee_errors is not None:
        columns.update(ee_position_error_m=sim.ee_errors[:n], ee_holding=sim.ee_hold_status[:n].astype(int),
                       ee_waypoint_index=sim.ee_waypoint_indices[:n])
    if sim.ee_velocities is not None:
        columns['ee_velocity_error_m_s'] = np.linalg.norm(sim.ee_velocities[:, :n]-sim.ee_velocity_commands[:, :n], axis=0)
    with (folder / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))
    durations = sim.solve_seconds[:n]
    finite = durations[np.isfinite(durations)]
    summary = dict(controller=controller, task=sim.agent.task, robot=sim.robot,
                   simulation_seconds=float(t[-1]-t[0]), recorded_states=n,
                   wall_seconds=sim.run_wall_seconds, solve_total_seconds=float(finite.sum()),
                   solve_seconds=statistics(finite), control_period_seconds=1/sim.ctrl_rate,
                   deadline_miss_fraction=float(np.mean(finite > 1/sim.ctrl_rate)) if len(finite) else None,
                   cost_evaluation_total_seconds=float(np.nansum(sim.cost_seconds[:n])),
                   predicted_horizon_cost=statistics(sim.cost[0, :n]),
                   body_position_error_m=statistics(sim.body_position_errors[:n]),
                   body_velocity_error_m_s=statistics(columns['body_velocity_error_m_s']),
                   mujoco_warnings=int(sim.data.warning.number.sum()),
                   cost_definition='Selected plan predicted cumulative horizon objective, not realized stage cost; held between solves.',
                   timing_definition='solve_seconds times agent.update, excluding rendering, diagnostic cost evaluation and file/plot export.')
    summary.update(sync=sim.sync,
                   plan_age_seconds=statistics(sim.plan_age_seconds[:n]),
                   plan_application_delay_seconds=statistics(sim.plan_application_delay_seconds[:n]),
                   sync_lag_seconds=statistics(sim.sync_lag_seconds[:n]),
                   plan_exhausted_fraction=float(sim.plan_exhausted[:n].mean()),
                   latency_definition='Serial measured-latency emulation, not concurrent execution. Solve plus diagnostic cost evaluation rounded up to a physics step. Old plan executes until arrival; new plan indexed from observation time; exhausted plans hold last command.')
    if sim.velocity_task:
        summary.pop('body_position_error_m')
        summary.update(control_mode='cmd_vel', body_velocity_frame='body', ee_velocity_frame='world',
                       command_transition='Simulation time; final zero command remains active.')
    if sim.ee_errors is not None:
        summary['ee_position_error_m'] = statistics(sim.ee_errors[:n])
        summary['ee_final_holding'] = bool(sim.ee_hold_status[n-1])
        summary['ee_error_frame'] = sim.agent.ee_objective.frame
        final_waypoint = sim.ee_waypoint_indices[:n] == len(sim.agent.waypoints)-1
        tail = final_waypoint & (t >= t[-1]-5.)
        summary['ee_final_waypoint_last_5s_error_m'] = statistics(sim.ee_errors[:n][tail])
        summary['ee_final_waypoint_last_5s_samples'] = int(tail.sum())
    if sim.ee_velocities is not None:
        summary['ee_velocity_error_m_s'] = statistics(columns['ee_velocity_error_m_s'])
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    config = dict(task=sim.agent.task_data, parameters=sim.agent.params,
                  execution=dict(sync=sim.sync),
                  rollout_model=getattr(sim.agent, 'rollout_model', None),
                  effective_solver=dict(n_samples=sim.agent.n_samples, n_iterations=sim.agent.n_iterations,
                                        horizon=sim.agent.horizon, temperature=sim.agent.temperature,
                                        noise_sigma=sim.agent.noise_sigma))
    (folder / 'config.json').write_text(json.dumps(config, indent=2, default=json_value), encoding='utf-8')
    fig = Figure(figsize=(12, 10), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.subplots(3, 2)
    ax[0, 0].plot(t, columns['solve_ms'], '.', markersize=3)
    ax[0, 0].axhline(1000/sim.ctrl_rate, color='r', ls='--', label='Control period')
    if sim.sync:
        ax[0, 0].plot(t, columns['plan_application_delay_ms'], 'x', label='Plan arrival delay')
        ax[0, 0].plot(t, columns['plan_age_ms'], alpha=.5, label='Active plan age')
    ax[0, 0].set(ylabel='Solve time (ms)', title='Controller computation')
    ax[0, 0].legend()
    ax[0, 1].plot(t, sim.cost[0, :n])
    ax[0, 1].set(ylabel='Predicted horizon cost', title='Selected plan cost')
    if sim.velocity_task:
        actual = sim.body_velocity_commands[:, :n]+sim.body_velocity_errors[:, :n]
        for i, label in enumerate('xy'):
            ax[1, 0].plot(t, actual[i], label=f'Body {label} actual')
            ax[1, 0].plot(t, sim.body_velocity_commands[i, :n], '--', label=f'Body {label} command')
    else:
        ax[1, 0].plot(t, sim.body_position_errors[:n], label='Body position')
    if sim.ee_errors is not None:
        ax[1, 0].plot(t, sim.ee_errors[:n], label='EE position')
    ax[1, 0].set(ylabel='Velocity (m/s)' if sim.velocity_task else 'Error (m)',
                 title='Body-local velocity tracking' if sim.velocity_task else 'Position tracking')
    ax[1, 0].legend()
    ax[1, 1].plot(t, columns['body_velocity_error_m_s'], label='Body velocity')
    if sim.ee_velocities is not None:
        ax[1, 1].plot(t, columns['ee_velocity_error_m_s'], label='EE velocity')
    ax[1, 1].set(ylabel='Error (m/s)', title='Velocity tracking')
    ax[1, 1].legend()
    for i, label in enumerate('xyz'):
        ax[2, 0].plot(t, sim.qpos[i, :n], label=label)
    ax[2, 0].set(ylabel='Position (m)', title='Body position')
    ax[2, 0].legend()
    ax[2, 1].plot(t, sim.ctrl[:, :n].T, linewidth=.6)
    ax[2, 1].set(ylabel='Joint command (rad)', title='Actuator commands')
    for axis in ax.flat:
        axis.set_xlabel('Simulation time (s)')
        axis.grid(alpha=.25)
    fig.suptitle(f'{controller} / {sim.agent.task} / {sim.robot}')
    fig.savefig(folder / 'metrics.png', dpi=150)
    if sim.ee_errors is not None:
        ee = Figure(figsize=(12, 7), constrained_layout=True)
        FigureCanvasAgg(ee)
        axes = ee.subplots(2, 3)
        for i, label in enumerate('xyz'):
            axes[0, i].plot(t, sim.ee_positions[i, :n], label='Actual')
            axes[0, i].plot(t, sim.ee_world_targets[i, :n], '--', label='Target')
            axes[0, i].set(ylabel=f'World {label} (m)')
            if sim.ee_velocities is not None:
                axes[1, i].plot(t, sim.ee_velocities[i, :n], label='Actual')
                axes[1, i].plot(t, sim.ee_velocity_commands[i, :n], '--', label='Command')
                axes[1, i].set(ylabel=f'World v{label} (m/s)')
            else:
                axes[1, i].plot(t, sim.ee_positions[i, :n]-sim.ee_world_targets[i, :n])
                axes[1, i].set(ylabel=f'World {label} error (m)')
        for axis in axes.flat:
            axis.set_xlabel('Simulation time (s)')
            axis.grid(alpha=.25)
            if axis.get_legend_handles_labels()[0]:
                axis.legend()
        ee.savefig(folder / 'ee_tracking.png', dpi=150)
    if sim.velocity_task:
        ee = Figure(figsize=(12, 7), constrained_layout=True)
        FigureCanvasAgg(ee)
        axes = ee.subplots(2, 3)
        for i, label in enumerate('xyz'):
            axes[0, i].plot(t, sim.ee_positions[i, :n], label='Actual position (no target)')
            axes[0, i].set(ylabel=f'World {label} (m)')
            axes[1, i].plot(t, sim.ee_velocities[i, :n], label='Actual')
            axes[1, i].plot(t, sim.ee_velocity_commands[i, :n], '--', label='Command')
            axes[1, i].set(ylabel=f'World v{label} (m/s)')
        for axis in axes.flat:
            axis.set_xlabel('Simulation time (s)')
            axis.grid(alpha=.25)
            axis.legend()
        ee.savefig(folder / 'ee_tracking.png', dpi=150)
    return folder
