"""Task registry for the standalone B2-Z1 simulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    description: str
    keyframe: str
    config_path: Path
    controller: str = "hold"
    default_duration: float = 30.0
    control_decimation: int = 5
    goal_positions: tuple[tuple[float, float, float], ...] = ()
    desired_linear_velocities: tuple[tuple[float, float, float], ...] = ()
    desired_gaits: tuple[str, ...] = ()
    goal_thresholds: tuple[float, ...] = ()
    waiting_times: tuple[int, ...] = ()


TASKS = {
    "sit_hold": TaskSpec(
        name="sit_hold",
        description="Maintain the B2-Z1 seated pose",
        keyframe="sit",
        config_path=PACKAGE_DIR / "configs" / "mppi_sit_hold.yml",
    ),
    "stand_hold": TaskSpec(
        name="stand_hold",
        description="Maintain the B2-Z1 standing pose",
        keyframe="stand",
        config_path=PACKAGE_DIR / "configs" / "mppi_stand_hold.yml",
    ),
    "in_place": TaskSpec(
        name="in_place",
        description="Stand from sit and run the original-style FAST in-place gait",
        keyframe="sit",
        config_path=PACKAGE_DIR / "configs" / "mppi_locomotion_in_place.yml",
        controller="locomotion",
        control_decimation=1,
        goal_positions=((0.0, 0.0, 0.55),),
        desired_linear_velocities=((0.0, 0.0, 0.0),),
        desired_gaits=("in_place",),
        goal_thresholds=(0.2,),
        waiting_times=(0,),
    ),
    "walk_straight": TaskSpec(
        name="walk_straight",
        description="Follow the original three-stage straight-line waypoint task",
        keyframe="sit",
        config_path=PACKAGE_DIR / "configs" / "mppi_locomotion_in_place.yml",
        controller="locomotion",
        default_duration=20.0,
        control_decimation=1,
        goal_positions=(
            (0.0, 0.0, 0.55),
            (1.0, 0.0, 0.55),
            (1.0, 0.0, 0.55),
        ),
        desired_linear_velocities=(
            (0.0, 0.0, 0.0),
            (0.2, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        desired_gaits=("in_place", "walk_fast", "in_place"),
        goal_thresholds=(0.2, 0.2, 0.2),
        waiting_times=(0, 0, 0),
    ),
}


def get_task(name: str) -> TaskSpec:
    try:
        return TASKS[name]
    except KeyError as error:
        available = ", ".join(TASKS)
        raise ValueError(f"Unknown task '{name}'. Available tasks: {available}") from error


def list_tasks() -> tuple[TaskSpec, ...]:
    return tuple(TASKS.values())
