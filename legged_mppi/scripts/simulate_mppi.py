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

VALID_TASKS = ['stairs', 'stand', 'walk_octagon', 'walk_straight', 'big_box',
               'walk_octagon_hw', 'walk_straight_hw', 'stand_hw', 'climb_box_hw',
               'ee_reach', 'ee_walk']


def main(task, mppi_mode=None, robot='go1', *, headless=False, steps=2000, plot=True, rollout_model='full'):
    """steps counts 10 ms control periods for either robot."""
    task_data = get_task(task)
    if task_data.get('controller') != 'whole_body':
        agent = LocomotionController(task=task, robot=robot, rollout_model=rollout_model)
    else:
        from whole_body_mppi.control.tasks.whole_body import WholeBodyController
        agent = WholeBodyController(task, robot, rollout_model)
    try:
        apply_mppi_mode(agent, mppi_mode)
        simulator = Simulator(
            agent=agent, viewer=not headless, T=steps + 1,
            dt=0.01, timeconst=0.02, dampingratio=1.0,
            model_path=PACKAGE_ROOT / task_data['sim_path'], ctrl_rate=100,
            robot=robot, initial_key=task_data.get('initial_key', 1), model_variant=rollout_model)
        if hasattr(agent, 'ee_objective'):
            import numpy as np
            initial_error = np.linalg.norm(agent.ee_objective.positions(simulator.data.qpos)-agent.ee_objective.target)
            print(f'EE initial error: {initial_error:.4f} m')
        simulator.run()
        print(f'robot={robot}, mode={mppi_mode or "configured"}, '
              f't={simulator.data.time:.2f}s, base={simulator.data.qpos[:3]}, '
              f'goal={agent.goal_index}, warnings={simulator.data.warning.number.sum()}')
        if hasattr(agent, 'ee_objective'):
            final_error = np.linalg.norm(agent.ee_objective.positions(simulator.data.qpos)-agent.ee_objective.target)
            print(f'EE final error: {final_error:.4f} m ({agent.ee_objective.frame} frame)')
        if plot:
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
    parser.add_argument('--headless', action='store_true', help='Disable the simulation window and plots.')
    parser.add_argument('--no-plot', action='store_true', help='Disable end-of-run trajectory plots.')
    parser.add_argument('--steps', type=int, default=2000, help='Number of 10 ms control periods (default: 2000).')
    add_mppi_mode_arguments(parser)
    args = parser.parse_args()
    task_data = get_task(args.task)
    if task_data.get('controller') == 'whole_body':
        if args.robot != task_data['required_robot'] or args.rollout_model != task_data['required_rollout_model']:
            parser.error(f'{args.task} requires --robot go1_dm --rollout-model full')
    if args.steps < 1:
        parser.error('--steps must be positive')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args.task, args.mppi_mode, args.robot,
         rollout_model=args.rollout_model, headless=args.headless, steps=args.steps, plot=not (args.headless or args.no_plot))
