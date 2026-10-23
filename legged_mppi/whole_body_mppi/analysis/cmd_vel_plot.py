"""Save command tracking figures without an interactive plotting window."""
from pathlib import Path
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from whole_body_mppi.control.costs.cmd_vel import heading_velocity


def save_cmd_vel_plot(simulator, output_dir=None):
    agent = simulator.agent
    count = simulator.recorded_steps
    time = simulator.time[:count]
    velocity, yaw_rate, _ = heading_velocity(
        simulator.qpos[3:7, :count].T,
        simulator.qvel[:3, :count].T,
        simulator.qvel[3:6, :count].T)
    actual = np.column_stack((velocity[:, :2], yaw_rate))
    command = simulator.command_history[:count]
    target = simulator.command_target_history[:count]
    error = actual - command
    rmse = np.sqrt(np.mean(error**2, axis=0))
    mode = agent.mppi_mode or ('mppi' if agent.n_iterations == 1 else 'dial-mpc (configured)')
    safety = getattr(agent, 'safety_mode', None)
    started = simulator.run_started_at
    fig = Figure(figsize=(14, 10), layout='constrained')
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 2, sharex=True)
    fig.suptitle(
        f'{agent.task} | robot={simulator.robot} | solver={mode} | safety={safety or "none"}'
        f'{" | STOPPED: filter failure" if getattr(simulator, "filter_failure", None) else ""}\n'
        f'Start: {started:%Y-%m-%d %H:%M:%S %Z} | '
        f'N={agent.n_samples}, H={agent.horizon}, iterations={agent.n_iterations}, '
        f'lambda={agent.temperature:g}, dt={agent.h:g}s\n'
        f'sampling={agent.sample_type}, knots={agent.n_knots}, seed={agent.params["seed"]}, '
        f'workers={agent.num_workers} | simulated={time[-1]-time[0]:.2f}s', fontsize=12)
    for i, (name, unit) in enumerate((('Heading vx', 'm/s'), ('Heading vy', 'm/s'), ('Yaw rate', 'rad/s'))):
        ax, err_ax = axes[i]
        ax.plot(time, target[:, i], ':', color='0.45', label='Requested command')
        ax.step(time, command[:, i], where='post', color='tab:orange', label='Rate-limited reference')
        ax.plot(time, actual[:, i], color='tab:blue', label='Measured')
        ax.set_title(name)
        ax.set_ylabel(unit)
        err_ax.plot(time, error[:, i], color='tab:red')
        err_ax.axhline(0, color='0.4', linewidth=.7)
        err_ax.set_title(f'{name} error | RMSE={rmse[i]:.4f} {unit}')
        err_ax.set_ylabel(unit)
        for panel in (ax, err_ax):
            panel.grid(True, alpha=.25)
    axes[0, 0].legend(loc='best', fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel('Simulation time (s)')
    directory = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parents[1] / 'outputs'
    directory.mkdir(parents=True, exist_ok=True)
    filename_mode = agent.mppi_mode or ('mppi' if agent.n_iterations == 1 else 'dial-mpc-configured')
    if safety is not None:
        filename_mode += '+' + safety
    path = directory / f'{agent.task}_{filename_mode}_{started:%Y%m%d_%H%M%S_%f}_KST.png'
    fig.savefig(path, dpi=160)
    title = fig._suptitle.get_text()
    fig.clear()
    if getattr(getattr(agent, 'ee_objective', None), 'velocity_tracking', False):
        ee_fig = Figure(figsize=(14, 10), layout='constrained')
        FigureCanvasAgg(ee_fig)
        ee_axes = ee_fig.subplots(3, 2, sharex=True)
        ee_fig.suptitle(title + f'\nEE linear velocity | frame={agent.ee_objective.frame}', fontsize=11)
        actual_ee = simulator.ee_velocities[:count]
        command_ee = simulator.ee_commands[:count]
        for i, axis in enumerate(('x', 'y', 'z')):
            error_ee = actual_ee[:, i]-command_ee[:, i]
            ee_axes[i,0].step(time, command_ee[:,i], where='post', label='EE command')
            ee_axes[i,0].plot(time, actual_ee[:,i], label='EE measured')
            ee_axes[i,0].set_title(f'EE v{axis}')
            ee_axes[i,0].legend()
            ee_axes[i,1].plot(time, error_ee, color='tab:red')
            ee_axes[i,1].set_title(f'Error | RMSE={np.sqrt(np.mean(error_ee**2)):.4f} m/s')
            for panel in ee_axes[i]:
                panel.set_ylabel('m/s')
                panel.grid(True, alpha=.25)
        for panel in ee_axes[-1]:
            panel.set_xlabel('Simulation time (s)')
        simulator.ee_velocity_plot_path = path.with_name(path.stem + '_ee_velocity.png')
        ee_fig.savefig(simulator.ee_velocity_plot_path, dpi=160)
        ee_fig.clear()
        print(f'EE velocity plot: {simulator.ee_velocity_plot_path}')
    return path
