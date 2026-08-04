import os
import sys
from pathlib import Path

package_root = str(Path(__file__).resolve().parents[1])
if package_root not in sys.path:
    sys.path.insert(0, package_root)

from whole_body_mppi.interface.simulator import Simulator
from whole_body_mppi.utils.tasks import get_task

import argparse

def main(
    task,
    planner_rate,
    backend="cpu",
    samples=None,
    horizon=None,
    steps=2000,
    viewer=True,
    plot=True,
):
    T = steps
    VIEWER = viewer

    SIMULATION_STEP = 0.01
    CTRL_UPDATE_RATE = planner_rate
    # Soft contact model paramters
    TIMECONST = 0.02
    DAMPINGRATIO = 1.0
    

    # Get task data
    task_data = get_task(task)
    sim_path = os.path.join(os.path.dirname(__file__), "../whole_body_mppi", task_data["sim_path"])

    # Initialize agent and simulator
    if task == 'push_box':
        from whole_body_mppi.control.controllers.mppi_locomanipulation import (
            MPPI_box_push,
        )
        agent = MPPI_box_push(task=task, backend=backend)
    else:
        from whole_body_mppi.control.controllers.mppi_locomotion import MPPI
        agent = MPPI(task=task, backend=backend)
    if samples is not None or horizon is not None:
        agent.set_params(
            horizon=agent.horizon if horizon is None else horizon,
            lambda_=agent.temperature,
            N=agent.n_samples if samples is None else samples,
        )
    simulator = Simulator(agent=agent, viewer=VIEWER, T=T, dt=SIMULATION_STEP, timeconst=TIMECONST,
                          dampingratio=DAMPINGRATIO, model_path=sim_path, ctrl_rate=CTRL_UPDATE_RATE)
    
    # Run simulation
    try:
        simulator.run()
        if plot:
            simulator.plot_trajectory()
    finally:
        agent.shutdown()

if __name__ == "__main__":
    # Define valid tasks
    VALID_TASKS = ['stairs', 'stand', 'walk_octagon', 'walk_straight', 'big_box', 'push_box',
                   'walk_octagon_hw', 'walk_straight_hw', 'stand_hw', 'climb_box_hw']

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run simulation with a specified task.")
    parser.add_argument('--task', type=str, required=True, choices=VALID_TASKS, 
                        help=f"Name of the task. Must be one of {VALID_TASKS}.")
    parser.add_argument(
        '--planner-rate', type=float, default=25,
        help="MPPI update rate in Hz (default: 25).",
    )
    parser.add_argument(
        '--backend', choices=('cpu', 'warp'), default='cpu',
        help="MPPI compute backend; warp keeps rollout and all costs on CUDA.",
    )
    parser.add_argument(
        '--samples', type=int, default=None,
        help="Override the trajectory sample count from the task config.",
    )
    parser.add_argument(
        '--horizon', type=int, default=None,
        help="Override the prediction horizon from the task config.",
    )
    parser.add_argument(
        '--steps', type=int, default=2000,
        help="Number of 100 Hz simulation steps (default: 2000).",
    )
    parser.add_argument(
        '--headless', action='store_true',
        help="Run without opening the MuJoCo viewer.",
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help="Do not create/show the trajectory plot after simulation.",
    )
    args = parser.parse_args()

    # Run main with the provided task
    main(
        args.task,
        args.planner_rate,
        backend=args.backend,
        samples=args.samples,
        horizon=args.horizon,
        steps=args.steps,
        viewer=not args.headless,
        plot=not args.no_plot,
    )
