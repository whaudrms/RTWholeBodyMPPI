import numpy as np
import os
import sys
from pathlib import Path

package_root = str(Path(__file__).resolve().parents[1])
if package_root not in sys.path:
    sys.path.insert(0, package_root)

from mani_mppi.interface.simulator import Simulator
from mani_mppi.utils.tasks import get_task

import argparse

TASK_ALIASES = {
    "ee_tracking": "locomani",
}


def main(task, viewer_render_rate=30.0, rollout_mode=None):
    T = 2000  # 20 seconds
    VIEWER = True

    SIMULATION_STEP = 0.01
    CTRL_UPDATE_RATE = 100
    CTRL_HORIZON = 40
    CTRL_LAMBDA = 0.1
    CTRL_N_SAMPLES = 30

    # Soft contact model paramters
    TIMECONST = 0.02
    DAMPINGRATIO = 1.0
    

    # Get task data
    task_name = TASK_ALIASES.get(task, task)
    task_data = get_task(task_name)
    sim_path = os.path.join(os.path.dirname(__file__), "../mani_mppi", task_data["sim_path"])

    # Initialize agent and simulator
    if task in {"locomani", "ee_tracking", "push_box"}:
        if rollout_mode is not None:
            raise ValueError(
                "--rollout-mode is currently available for locomotion tasks only"
            )
    if task == "locomani":
        from mani_mppi.control.controllers.mppi_locomani import MPPI
        agent = MPPI(task=task_name)
    elif task == "ee_tracking":
        from mani_mppi.control.controllers.mppi_ee_tracking import MPPI
        agent = MPPI(task=task_name)
    elif task == "push_box":
        from mani_mppi.control.controllers.mppi_push_box import MPPI
        agent = MPPI(task=task_name)
    else:
        from mani_mppi.control.controllers.mppi_locomotion import MPPI
        agent = MPPI(task=task_name, rollout_mode=rollout_mode)
    # agent.set_params(horizon=CTRL_HORIZON, lambda_=CTRL_LAMBDA, N=CTRL_N_SAMPLES)
    if viewer_render_rate <= 0:
        raise ValueError("viewer_render_rate must be positive")
    render_every = max(1, round(1.0 / (SIMULATION_STEP * viewer_render_rate)))
    simulator = Simulator(agent=agent, viewer=VIEWER, T=T, dt=SIMULATION_STEP, timeconst=TIMECONST,
                          dampingratio=DAMPINGRATIO, model_path=sim_path, ctrl_rate=CTRL_UPDATE_RATE,
                          render_every=render_every)
    
    # Run simulation
    simulator.run()
    simulator.plot_trajectory()

if __name__ == "__main__":
    # Define valid tasks
    VALID_TASKS = [
        'stand',
        'walk_straight',
        'big_box',
        'locomani',
        'ee_tracking',
        'push_box',
    ]

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run simulation with a specified task.")
    parser.add_argument('--task', type=str, required=True, choices=VALID_TASKS, 
                        help=f"Name of the task. Must be one of {VALID_TASKS}.")
    parser.add_argument(
        '--render-rate', type=float, default=60.0,
        help='Viewer frames per simulated second (default: 30).',
    )
    parser.add_argument(
        '--rollout-mode', choices=('gait', 'original'), default=None,
        help=(
            'Override locomotion rollout sampling: gait uses the current '
            'gait-residual method; original uses previous-solution absolute '
            'cubic sampling.'
        ),
    )
    args = parser.parse_args()

    # Run main with the provided task
    main(
        args.task,
        viewer_render_rate=args.render_rate,
        rollout_mode=args.rollout_mode,
    )
