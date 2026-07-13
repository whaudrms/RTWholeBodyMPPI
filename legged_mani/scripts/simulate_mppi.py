"""Run and visualize every registered B2-Z1 MPPI task from one CLI."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import mujoco.viewer
import numpy as np

from legged_mani.control import B2Z1LocomotionMPPI, B2Z1PoseHoldMPPI
from legged_mani.interface import B2Z1Env
from legged_mani.utils.tasks import TASKS, TaskSpec, get_task, list_tasks


@dataclass
class RunResult:
    observations: np.ndarray
    controls: np.ndarray
    costs: np.ndarray
    simulated_time: float
    wall_time: float
    mean_update_time: float


def print_tasks() -> None:
    print("Available B2-Z1 tasks:")
    for task in list_tasks():
        print(f"  {task.name:<12} {task.description}")


def create_controller(task: TaskSpec):
    if task.controller == "hold":
        return B2Z1PoseHoldMPPI(config_path=task.config_path)
    if task.controller == "locomotion":
        return B2Z1LocomotionMPPI(task=task.name, config_path=task.config_path)
    raise ValueError(f"Unsupported controller type: {task.controller}")


def run_task(
    task: TaskSpec,
    duration: float,
    control_decimation: int,
    viewer_enabled: bool,
    realtime: bool,
    max_steps: int | None = None,
) -> RunResult:
    env = B2Z1Env()
    observation = env.reset(task.keyframe)
    observations = [observation.copy()]
    controls = []
    costs = []
    update_times = []
    simulation_start = env.data.time
    controller = create_controller(task)
    wall_start = 0.0

    with controller:
        # Model/controller loading is setup time, not part of simulation pacing.
        wall_start = time.perf_counter()
        def loop(viewer=None) -> None:
            nonlocal observation
            step_count = 0
            while viewer is None or viewer.is_running():
                simulated = env.data.time - simulation_start
                if duration > 0.0 and simulated >= duration:
                    break
                if max_steps is not None and step_count >= max_steps:
                    break

                update_start = time.perf_counter()
                controller.update(observation)
                update_times.append(time.perf_counter() - update_start)
                current_cost = controller.eval_best_trajectory()
                executed_steps = 0

                for index in range(control_decimation):
                    if viewer is not None and not viewer.is_running():
                        break
                    if max_steps is not None and step_count >= max_steps:
                        break
                    if duration > 0.0 and env.data.time - simulation_start >= duration:
                        break

                    action = controller.selected_trajectory[
                        min(index, controller.horizon - 1)
                    ]
                    observation = env.step(action)

                    # Preserve the original simulator's task-progression
                    # contract. Locomotion uses this to update its timer,
                    # waiting state, gait selection, and exploration policy.
                    if (
                        hasattr(controller, "next_goal")
                        and hasattr(controller, "goal_thresh")
                    ):
                        goal_error = np.linalg.norm(
                            controller.body_ref[:3] - observation[:3]
                        )
                        if goal_error < controller.goal_thresh[controller.goal_index]:
                            controller.next_goal()

                    observations.append(observation.copy())
                    controls.append(action.copy())
                    costs.append(current_cost)
                    step_count += 1
                    executed_steps += 1

                    if viewer is not None:
                        viewer.sync()
                    if realtime:
                        target = wall_start + (env.data.time - simulation_start)
                        remaining = target - time.perf_counter()
                        if remaining > 0.0:
                            time.sleep(remaining)

                if hasattr(controller, "advance_reference"):
                    controller.advance_reference(executed_steps)

        if viewer_enabled:
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.35])
                viewer.cam.distance = 1.8
                viewer.cam.azimuth = 135.0
                viewer.cam.elevation = -20.0
                loop(viewer)
        else:
            loop()

    wall_time = time.perf_counter() - wall_start
    simulated_time = env.data.time - simulation_start
    return RunResult(
        observations=np.asarray(observations),
        controls=np.asarray(controls),
        costs=np.asarray(costs),
        simulated_time=simulated_time,
        wall_time=wall_time,
        mean_update_time=float(np.mean(update_times)) if update_times else 0.0,
    )


def plot_result(task: TaskSpec, result: RunResult, timestep: float = 0.01) -> None:
    import matplotlib.pyplot as plt

    state_time = np.arange(len(result.observations)) * timestep
    control_time = np.arange(len(result.controls)) * timestep
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    axes[0].plot(state_time, result.observations[:, :3])
    axes[0].set_ylabel("base position [m]")
    axes[0].legend(("x", "y", "z"), loc="best")
    axes[1].plot(control_time, result.controls[:, :12])
    axes[1].set_ylabel("leg target [rad]")
    axes[2].plot(control_time, result.costs)
    axes[2].set_ylabel("best rollout cost")
    axes[2].set_xlabel("simulation time [s]")
    figure.suptitle(f"B2-Z1 MPPI: {task.name}")
    figure.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a registered B2-Z1 MPPI task and visualize it."
    )
    parser.add_argument("--task", choices=tuple(TASKS), help="task to run")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None, help="simulation-step limit")
    parser.add_argument("--control-decimation", type=int, default=None)
    parser.add_argument("--headless", action="store_true", help="disable MuJoCo viewer")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--plot", action="store_true", help="plot states and cost after run")
    args = parser.parse_args()
    if not args.list_tasks and args.task is None:
        parser.error("--task is required unless --list-tasks is used")
    if args.control_decimation is not None and args.control_decimation < 1:
        parser.error("--control-decimation must be at least 1")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    if args.list_tasks:
        print_tasks()
        return

    task = get_task(args.task)
    duration = task.default_duration if args.duration is None else args.duration
    decimation = task.control_decimation if args.control_decimation is None else args.control_decimation
    if args.headless and duration <= 0.0 and args.steps is None:
        raise ValueError("Headless runs need a positive --duration or --steps")

    print(f"task: {task.name} ({task.description})")
    print(
        f"control period: {0.01 * decimation:.3f} s "
        f"({100.0 / decimation:.1f} Hz MPPI update)"
    )
    result = run_task(
        task=task,
        duration=duration,
        control_decimation=decimation,
        viewer_enabled=not args.headless,
        realtime=not args.no_realtime and not args.headless,
        max_steps=args.steps,
    )
    factor = result.simulated_time / result.wall_time if result.wall_time > 0 else 0.0
    final = result.observations[-1]
    print(f"simulated: {result.simulated_time:.2f} s, wall: {result.wall_time:.2f} s")
    print(f"mean MPPI update: {1000.0 * result.mean_update_time:.1f} ms")
    print(f"real-time factor: {factor:.2f}x")
    print(f"final base position: {np.array2string(final[:3], precision=5)}")
    if args.plot:
        plot_result(task, result)


if __name__ == "__main__":
    main()
