# MPOPI

From the repository root with the project Python environment active:

```bash
PYTHONPATH=legged_mppi python legged_mppi/scripts/simulate_mppi.py --task walk_straight --mpopi
```

The same `--mpopi` flag is available in both ROS controller scripts and is
mutually exclusive with `--mppi`, `--dial-mpc`, and `--cma-mppi`.
The solver is `whole_body_mppi/control/solvers/mpopi.py`.

## Adaptation

This ports the optimizer in `/home/tony/9_mppi_study/mppi/mpopi/controller.py`
to the existing task cost callback. References, task costs, physics, gait
advancement and action limits remain unchanged. No smoothing or control-noise
cross cost from the study implementation is added.

Each cycle starts from the shifted trajectory. Intermediate iterations select
the lowest-cost elites and update the proposal mean and covariance using
normalized logarithmic rank weights. The covariance moment is centered on the
OLD proposal mean. Covariance is represented in coordinates normalized by
`noise_sigma`, symmetrized, and stabilized with an eigenvalue floor.

Cubic sampling adapts a full action-axis covariance at each knot, then
interpolates bounded knot candidates and clips dense actions. Normal sampling
adapts at every horizon step. Covariance captures joint correlations, but not
cross-time correlations. The current proposal is included as candidate zero.

Only the last iteration's candidates enter the final exponential cost-weighted
mean. Covariance resets to base covariance at the start of each control cycle;
it does not persist like CMA-MPPI covariance. DIAL annealing is not applied.

## Settings

Optional settings in the task YAML under `whole_body_mppi/control/configs/`:

```yaml
mpopi:
  elite_fraction: 0.2
  learning_rate: 0.5
  covariance_floor: 0.001
```

`elite_fraction` is in (0, 1], `learning_rate` in [0, 1], and
`covariance_floor` in (0, 1]. `noise_sigma` must contain positive finite
values. At least two candidates per iteration are required. With one
iteration the solver skips elite adaptation and performs final MPPI weighting.

The total rollout budget is `n_samples`, matching every solver here and
the study repository. It is split across iterations, e.g. 10/3 uses [4, 3, 3].
Require `n_samples >= 2 * n_iterations`. Final weights use `exp(-(cost-min_cost)/lambda)`, matching the
study MPOPI weighting rather than MPPI/DIAL min-max normalization or CMA
standard-deviation normalization. Existing temperatures may need tuning for
Go1 cost magnitudes; implementation tests do not establish locomotion quality.

Diagnostics on the solver/controller:

- `mpopi_covariance_history`: normalized covariance used to sample each iteration.
- `mpopi_proposal_history`: proposal knot/step positions used in each iteration.
- `mpopi_final_actions`, `mpopi_final_costs`: final iteration candidates and costs.
- `exp_weights`: normalized final weights.
- `iteration_best_costs`: minimum finite cost in each iteration.

Planner reset and resizing clear these diagnostics. Non-finite candidates get
zero weight, and an all-invalid batch raises `FloatingPointError`.
