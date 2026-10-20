# CMA-MPPI for gait-based controllers

Run from the repository root with the wb-mppi environment active:

```bash
export PYTHONPATH="$PWD/legged_mppi${PYTHONPATH:+:$PYTHONPATH}"
python legged_mppi/scripts/simulate_mppi.py --task walk_straight --cma-mppi
```

The same mutually exclusive `--cma-mppi`, `--dial-mpc`, and `--mppi`
options are available in the ROS locomotion and locomanipulation scripts.
With no option the task YAML settings remain in effect.

CMA-MPPI keeps the existing task references, cost functions, action limits,
and rollout dynamics. `n_samples` is the total control-cycle budget, split
across `n_iterations` with the remainder assigned to earlier iterations.
For example, 10 samples and 3 iterations use [4, 3, 3] candidates.
CMA mode uses adaptive diagonal covariance instead of DIAL annealing.

Task YAML files are in `legged_mppi/whole_body_mppi/control/configs/`.
The implementation lives in `control/solvers/cma_mppi.py`; task integration
is described in [control_structure.md](control_structure.md).

Optional task YAML settings (defaults shown):

```yaml
cma_mppi:
  covariance_learning_rate: 0.5
  sigma_min: 0.05  # Standard deviation in joint position radians.
  sigma_max: 3.0
  elite_ratio: 0.0  # Zero uses all finite-cost candidates.
  reward_normalization: true
```

Weights follow the study CMA implementation: exponential cost differences
with temperature scaled by the cost standard deviation when
`reward_normalization` is true. No smoothing or control cost terms from
the study repository are added. The existing `lambda` may need tuning.

For cubic sampling, candidates and diagonal covariance live at the existing
knots. Bounded knots are interpolated, then dense actions are clipped again.
Weighted dense actions define the returned plan; weighted variance at knots
updates covariance using an exponential moving average. For normal sampling,
the same update applies independently at each horizon step.

Covariance persists between control updates, shifts by one simulation step
(linear interpolation of knot variances for cubic sampling), and restores
the final variance to the clipped base noise variance. Planner reset clears
adaptation. This is a knot-space adaptation of the local study implementation,
not full CMA-ES and not a full covariance method.
