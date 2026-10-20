# Control code structure

All paths below are relative to `legged_mppi/whole_body_mppi/control/`.

| Directory | Responsibility |
| --- | --- |
| `tasks/` | Task registry, goal transitions, body/joint references, Go1 setup and state layouts |
| `costs/` | Locomotion and locomanipulation cost calculations and quaternion errors |
| `solvers/` | MPPI/DIAL/CMA/MPOPI/MBD optimization, Gaussian/cubic sampling, covariance and warm-start state |
| `rollouts/` | MuJoCo batch integration and worker pool lifecycle |
| `configs/` | Existing task YAML settings, moved without changing their values |
| `gait_scheduler/` | Existing gait scheduling and gait assets |
| `controllers/` | Compatibility imports for existing notebooks and external scripts |

## Task controllers

```python
from whole_body_mppi.control.tasks.locomotion import LocomotionController
from whole_body_mppi.control.tasks.locomanipulation import LocomanipulationController
from whole_body_mppi.control.solvers.options import apply_mppi_mode

controller = LocomotionController('walk_straight')
apply_mppi_mode(controller, 'cma-mppi')
# action = controller.update(obs)
controller.shutdown()
```

Task definitions live in `tasks/definitions.py`. Its `config_path` values are
relative to `control/`; model and simulation paths are relative to
`whole_body_mppi/`. Edit weights and solver settings in `configs/`.

Task controllers inherit the shared `BaseTaskController` and their cost mixin.
The task controller prepares the current references and implements
`task_cost(states, actions)` for its state layout. `BaseTaskController.solve_task`
calls the solver, advances the gait once, and commits the final plan once.
Selected-plan evaluation uses the same task-specific state layout as planning.

## Solver boundary

`MPPISolver` receives an already configured MuJoCo model, solver parameters,
action limits, and an initial action. It has no Go1 limits, gait files, task
registry, or reference/cost generation. `optimize_actions(obs, evaluate_cost)`
accepts an objective callback and dispatches to `solvers/mppi.py` or
`solvers/cma_mppi.py`, `solvers/mpopi.py`, or `solvers/mbd.py`. The callback receives candidate actions and can use
`solver.state_rollouts`. Nominal and DIAL share the iterative update with
mode-specific noise parameters. CMA uses its adaptive covariance update.

`commit_plan(actions)` stores the selected trajectory and shifts the mean
warm start by one step. CMA's covariance shift remains inside its update;
reset and resize clear the adaptation state. `rollout_func` delegates batch
integration to a separate `MujocoRollout` instance.

## Compatibility and execution

Existing script names and `--mppi`, `--dial-mpc`, `--cma-mppi`, `--mpopi`, `--mbd` options are
unchanged. First-party scripts/tests use the new modules. Old imports of
`controllers.mppi_locomotion.MPPI`,
`controllers.mppi_locomanipulation.MPPI_box_push`, and `utils.tasks` remain
available through thin compatibility modules. YAML files are now under
`control/configs/`, rather than `control/controllers/configs/`.

```bash
PYTHONPATH=legged_mppi python legged_mppi/scripts/simulate_mppi.py --task walk_straight --cma-mppi
PYTHONPATH=legged_mppi python -m unittest discover -s legged_mppi/tests -v
```

MPOPI elite proposal adaptation is described in [mpopi.md](mpopi.md).

Model-based diffusion is described in [mbd.md](mbd.md).

## Shared rollout budget

`n_samples` is the total number of candidate trajectories per control update
for MPPI, DIAL, CMA-MPPI, MPOPI and MBD. `solvers/budget.py` partitions it
across iterations: 10/3 -> [4, 3, 3], 30/3 -> [10, 10, 10]. MPPI uses one
iteration and the full budget. Require at least two candidates per iteration.
The task cost sees only the active rollout buffer at each iteration.
`iteration_sample_counts` records the allocation. Changing iteration count
keeps total candidate evaluations and physics steps (`n_samples * horizon`)
constant. Stage barriers and contact-dependent runtime can still differ.
Selected-plan diagnostic evaluation in the simulator remains one extra
trajectory outside this optimization budget, equally for every mode.
