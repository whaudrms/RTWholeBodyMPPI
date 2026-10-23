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
        'cbf_margin', 'cbf_ee_margin', 'cbf_ee_radius', 'cbf_verified_residual', 'cbf_correction_norm', 'cbf_infeasible',
        'cbf_extra_rollouts', 'cbf_shield_seconds',
        'cbf_velocity_margins', 'cbf_body_speed', 'cbf_ee_speed',
        'cbf_body_margin', 'cbf_leg_correction_norm', 'cbf_arm_correction_norm',
        'cbf_body_guard_rejected', 'cbf_ee_correction_deferred',
        'cbf_nominal_body_residual', 'cbf_verified_body_residual',
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
    columns.update(cbf_margin=sim.cbf_margin[:n],
                   cbf_ee_margin=sim.cbf_ee_margin[:n], cbf_ee_radius_m=sim.cbf_ee_radius[:n],
                   cbf_verified_residual=sim.cbf_verified_residual[:n],
                   cbf_correction_rad=sim.cbf_correction_norm[:n],
                   cbf_infeasible=sim.cbf_infeasible[:n],
                   cbf_extra_rollouts=sim.cbf_extra_rollouts[:n],
                   cbf_shield_ms=sim.cbf_shield_seconds[:n]*1000)
    velocity_margin_names = ('body_speed', 'ee_speed', 'body_velocity_error', 'ee_velocity_error')
    columns.update({f'cbf_{name}_margin': sim.cbf_velocity_margins[i, :n]
                    for i, name in enumerate(velocity_margin_names)})
    columns.update(cbf_body_speed_m_s=sim.cbf_body_speed[:n], cbf_ee_speed_m_s=sim.cbf_ee_speed[:n])
    for name in ('body_margin', 'leg_correction_norm', 'arm_correction_norm',
                 'body_guard_rejected', 'ee_correction_deferred',
                 'nominal_body_residual', 'verified_body_residual'):
        columns[f'cbf_{name}'] = getattr(sim, f'cbf_{name}')[:n]
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
    cbf = getattr(sim.agent, 'cbf', None)
    margins = sim.cbf_margin[:n]
    margins = margins[np.isfinite(margins)]
    summary['cbf'] = dict(enabled=bool(cbf is not None and cbf.enabled),
                          margin=statistics(sim.cbf_margin[:n]),
                          ee_margin=statistics(sim.cbf_ee_margin[:n]),
                          ee_radius_m=statistics(sim.cbf_ee_radius[:n]),
                          minimum_margin=float(margins.min()) if len(margins) else None,
                          margin_violation_fraction=float(np.mean(margins < 0)) if len(margins) else None,
                          verified_residual=statistics(sim.cbf_verified_residual[:n]),
                          correction_rad=statistics(sim.cbf_correction_norm[:n]),
                          shield_seconds=statistics(sim.cbf_shield_seconds[:n]),
                          extra_rollouts_total=int(np.nansum(sim.cbf_extra_rollouts[:n])),
                          infeasible_solves=int(np.nansum(sim.cbf_infeasible[:n])),
                          intervention_solves=int(np.sum(sim.cbf_correction_norm[:n] > 1e-8)),
                          timing='Included in solve_seconds; extra_rollouts are additional to solver n_samples.',
                          verification='At solve observation time only; sync-delayed execution is not re-certified.')
    body_margin = sim.cbf_body_margin[:n]
    body_margin = body_margin[np.isfinite(body_margin)]
    summary['cbf']['body_priority'] = dict(
        margin=statistics(body_margin),
        minimum_margin=float(body_margin.min()) if len(body_margin) else None,
        violation_fraction=float(np.mean(body_margin < 0)) if len(body_margin) else None,
        nominal_residual=statistics(sim.cbf_nominal_body_residual[:n]),
        verified_residual=statistics(sim.cbf_verified_body_residual[:n]),
        leg_correction_rad=statistics(sim.cbf_leg_correction_norm[:n]),
        arm_correction_rad=statistics(sim.cbf_arm_correction_norm[:n]),
        rejected_solves=int(np.nansum(sim.cbf_body_guard_rejected[:n])),
        ee_deferred_solves=int(np.nansum(sim.cbf_ee_correction_deferred[:n])))
    summary['cbf']['velocity'] = dict(body_speed_m_s=statistics(sim.cbf_body_speed[:n]),
                                     ee_speed_m_s=statistics(sim.cbf_ee_speed[:n]))
    for i, name in enumerate(velocity_margin_names):
        values = sim.cbf_velocity_margins[i, :n]
        values = values[np.isfinite(values)]
        summary['cbf']['velocity'][name] = dict(margin=statistics(values),
                                                minimum_margin=float(values.min()) if len(values) else None,
                                                violation_fraction=float(np.mean(values < 0)) if len(values) else None)
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
                  execution=dict(sync=sim.sync, cbf=cbf.configuration() if cbf is not None else dict(enabled=False)),
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
    fig.suptitle(f'{controller} / {sim.agent.task} / {sim.robot} / CBF {"on" if cbf is not None and cbf.enabled else "off"}')
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
