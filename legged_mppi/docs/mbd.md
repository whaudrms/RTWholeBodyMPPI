# Model-based diffusion (MBD)

```bash
PYTHONPATH=legged_mppi python legged_mppi/scripts/simulate_mppi.py --task walk_straight --mbd
```

The same `--mbd` option is available in both ROS controller scripts. It is
mutually exclusive with the other solver flags. Implementation:
`whole_body_mppi/control/solvers/mbd.py`.

This ports the local study implementation at
`/home/tony/9_mppi_study/mppi/mbd/controller.py` to the current rollout/cost
interface and cubic knot representation. Task references, task costs, joint
limits, gait advancement, and physics remain unchanged. No smoothing or
control-noise costs are added.

## Settings

Optional task YAML section under `whole_body_mppi/control/configs/`:

```yaml
mbd:
  alpha_bar_first: 0.98
  alpha_bar_last: 0.2
```

Require `0 < alpha_bar_last <= alpha_bar_first < 1`. The number of diffusion
stages is `n_iterations`. A single stage uses `alpha_bar_last`.
`noise_sigma` must be positive and finite for each joint; it is the base
standard deviation in radians, rather than a fraction of joint range.

The warm-start knot/step positions are normalized as
`y = (q - center) / noise_sigma`, with center equal to the midpoint of the
joint limits. Forward diffusion shrinks the normalized sequence and adds
Gaussian noise using a geometric cumulative-alpha schedule. It performs no
model rollouts.

Backward stages run from the most corrupted level to the clean level. Each
stage samples clean candidates around `latent / sqrt(alpha_bar)` with
normalized standard deviation `sqrt(1/alpha_bar - 1)`. Candidate zero is the
proposal mean. Candidates are projected to joint limits, interpolated for
cubic sampling, clipped, and evaluated through the existing task cost.
Exponential cost weights estimate a clean sequence and its score. A
deterministic Monte Carlo score-ascent reverse step updates the latent.
No reverse-SDE noise is added. The final clean knots are interpolated and
clipped to obtain the selected plan.

This is a bounded knot/action-space adaptation, not full state-trajectory
diffusion or exact unconstrained inference. Clipping is a practical constraint
approximation. DIAL annealing and CMA/MPOPI covariance adaptation are not used.

Weights use `exp(-(cost-min_cost)/lambda)`, as in study MBD and current MPOPI.
This differs from MPPI/DIAL min-max normalization and CMA standard-deviation
normalization. Existing `lambda` may need tuning for locomotion cost magnitudes.

The total control-cycle budget is `n_samples`, split across reverse stages
with the remainder assigned to earlier executed stages. For example, 10/3
uses [4, 3, 3]. Require `n_samples >= 2 * n_iterations`, ensuring at least
two candidates per stage. This matches the study total-budget convention. Forward diffusion adds
computation but no rollout evaluations. The final plan is committed/shifted
once by the existing task integration.

## Diagnostics

- `mbd_alpha_bars`: cumulative alpha, shape `(n_iterations + 1,)`, starting at 1.
- `mbd_alphas`: per-stage alpha, shape `(n_iterations,)`.
- `mbd_forward_trace`: normalized forward knots, clean to corrupted.
- `mbd_backward_trace`: normalized reverse knots, corrupted to clean.
- `mbd_score_trace`: reverse-stage score estimates.
- `mbd_final_actions`, `mbd_final_costs`: final-stage evaluated candidates/costs.
- `exp_weights`: final-stage normalized weights.
- `iteration_best_costs`: stage minimum finite costs, in reverse execution order.

Forward/backward traces have shape `(n_iterations + 1, points, action_dim)`;
score traces have shape `(n_iterations, points, action_dim)`. `points` is
`n_knots` for cubic sampling and `horizon` for normal sampling. Planner reset
and resizing clear traces. Invalid costs get zero weight; all-invalid batches
raise `FloatingPointError`. Short numerical checks do not establish long-term
walking quality or real-time deadlines.
