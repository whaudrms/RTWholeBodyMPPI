"""Run locomotion or config-based whole-body tasks with Go1 and Go1 + DM."""
import argparse
from pathlib import Path
import sys

# Also support running directly from a source checkout without pip install -e.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whole_body_mppi.interface.simulator import Simulator
from whole_body_mppi.control.tasks.locomotion import LocomotionController
from whole_body_mppi.control.tasks.base import PACKAGE_ROOT
from whole_body_mppi.control.tasks.definitions import get_task
from whole_body_mppi.control.solvers.options import add_mppi_mode_arguments, apply_mppi_mode

VALID_TASKS = ['walk_straight_cmd_vel', 'stairs', 'stand', 'walk_octagon', 'walk_straight', 'big_box',
               'walk_octagon_hw', 'walk_straight_hw', 'stand_hw', 'climb_box_hw',
               'ee_reach_cmd_vel', 'ee_walk_cmd_vel']


def main(task, mppi_mode=None, robot='go1', *, headless=False, steps=2000, plot=True, rollout_model='full', initial_key=None, safety_mode=None):
    """steps counts 10 ms control periods for either robot."""
    task_data = get_task(task)
    if task_data.get('controller') != 'whole_body':
        agent = LocomotionController(task=task, robot=robot, rollout_model=rollout_model)
    else:
        from whole_body_mppi.control.tasks.whole_body import WholeBodyController
        agent = WholeBodyController(task, robot, rollout_model)
    try:
        apply_mppi_mode(agent, mppi_mode, safety_mode)
        simulator = Simulator(
            agent=agent, viewer=not headless, T=steps + 1,
            dt=0.01, timeconst=0.02, dampingratio=1.0,
            model_path=PACKAGE_ROOT / task_data['sim_path'], ctrl_rate=100,
            robot=robot, initial_key=(initial_key if initial_key is not None else
                task_data.get('initial_key', 'home')), model_variant=rollout_model)
        if hasattr(agent, 'ee_objective'):
            import numpy as np
            initial_error = np.linalg.norm(agent.ee_objective.velocities(simulator.data.qpos, simulator.data.qvel)-agent.ee_objective.command_at(simulator.data.time))
            print(f'EE initial velocity error: {initial_error:.4f} m/s')
        simulator.run()
        print(f'robot={robot}, mode={agent.mppi_mode or "configured"}, safety={agent.safety_mode or "none"}, '
              f't={simulator.data.time:.2f}s, base={simulator.data.qpos[:3]}, '
              f'goal={agent.goal_index}, warnings={simulator.data.warning.number.sum()}')
        if hasattr(agent, 'ee_objective'):
            final_error = np.linalg.norm(agent.ee_objective.velocities(simulator.data.qpos, simulator.data.qvel)-agent.ee_objective.command_at(simulator.data.time))
            print(f'EE final velocity error: {final_error:.4f} m/s ({agent.ee_objective.frame} frame)')
        if getattr(agent, 'shield', None) is not None:
            import csv
            history = agent.shield.history
            diagnostics = simulator.cmd_vel_plot_path.with_suffix('.shield.csv')
            with diagnostics.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['step'] + list(history[0]))
                writer.writeheader()
                for step, row in enumerate(history):
                    writer.writerow(dict(step=step, **row))
            failed = sum(not row['feasible'] for row in history)
            print(f'Shield: residual violations={failed}/{len(history)}, diagnostics={diagnostics}')
        if plot and simulator.filter_failure is None:
            simulator.plot_trajectory()
        return simulator
    finally:
        agent.shutdown()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', choices=('go1', 'go1_dm'), default='go1',
                        help='Go1 with DM; EE tasks jointly optimize the legs and arm.')
    parser.add_argument('--rollout-model', choices=('simple', 'full'), default='full',
                        help='DM physics/rollout model: simple=rigid payload, full=6-axis primitive arm (default); display always uses detailed meshes.')
    parser.add_argument('--task', required=True, choices=VALID_TASKS)
    parser.add_argument('--headless', action='store_true', help='Disable interactive windows; cmd_vel tracking PNG is still saved.')
    parser.add_argument('--no-plot', action='store_true', help='Disable interactive trajectory plots; cmd_vel tracking PNG is still saved.')
    parser.add_argument('--initial-key', choices=('home', 'sit', 'tune'), default=None,
                        help='Override start pose; defaults to the task initial key or standing home.')
    parser.add_argument('--steps', type=int, default=2000, help='Number of 10 ms control periods (default: 2000).')
    add_mppi_mode_arguments(parser, include_safety=True)
    args = parser.parse_args()
    task_data = get_task(args.task)
    if task_data.get('required_robot') and args.robot != task_data['required_robot']:
        parser.error(f'{args.task} requires --robot {task_data["required_robot"]}')
    if task_data.get('controller') == 'whole_body':
        if args.robot != task_data['required_robot'] or args.rollout_model != task_data['required_rollout_model']:
            parser.error(f'{args.task} requires --robot go1_dm --rollout-model full')
    if args.steps < 1:
        parser.error('--steps must be positive')
    return args


if __name__ == '__main__':
    args = parse_args()
    result = main(args.task, args.mppi_mode, args.robot,
         rollout_model=args.rollout_model, headless=args.headless, steps=args.steps, plot=not (args.headless or args.no_plot), initial_key=args.initial_key, safety_mode=args.safety_mode)

    if result.filter_failure is not None:
        sys.exit(2)
