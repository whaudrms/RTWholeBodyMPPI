# Experimental Shield-MPPI for Go1

```bash
python legged_mppi/scripts/simulate_mppi.py --task walk_straight_cmd_vel --shield-mppi --headless --steps 300
```

Supported scope: Go1 `walk_straight_cmd_vel`, and Go1+DM `ee_reach_cmd_vel` / `ee_walk_cmd_vel` (full model). The independent safety option adds DCBF violation costs to
sample evaluation, then repairs the first `repair_steps` controls. It does not
add CLF or a QP and is not enabled for hardware or other tasks.

Reference: https://arxiv.org/abs/2302.11719, equations 17/22 and Algorithm 2.
The local adaptation uses normalized Go1 height and tilt barriers instead of
vehicle track boundaries:

- `h_height = (z - min_height) / min_height`
- `h_tilt = (R_zz - cos(max_tilt)) / (1 - cos(max_tilt))`
- residual `h(x_next) - alpha*h(x_previous)`; nonnegative is satisfied.

The measured initial state is included in the first transition. All sampled
transitions receive `cost_weight * sum(max(-residual, 0))`. Final repair minimizes
the same unweighted violation sum over a short MuJoCo rollout, initialized from
the nominal plan. It skips optimization if the nominal prefix already satisfies
the conditions. No nominal-distance objective is added. Like Algorithm 2, the
nominal sequence supplies the next shifted warm start; the executed/selected plan
contains the repaired prefix.

Unlike the paper's BFGS description, SciPy L-BFGS-B is used to enforce joint
command bounds. Gradients use forward finite differences; both iterations and
objective evaluations are limited. These limits are NOT wall-clock deadlines.
The best evaluated bounded prefix is returned. If residual constraints remain
violated, the result is marked infeasible; there is no certified fallback or
unconditional safety guarantee. Nor do geometric height/tilt margins establish
control invariance for a legged robot with contacts. Defaults (0.18 m, 30 degrees)
are experimental and configurable in `control/configs/mppi_cmd_vel.yml`.

The command tracking PNG uses `shield-mppi` in its title/filename. A companion
`.shield.csv` records each control update's pre/post violation sum, minimum
residual, feasibility, evaluation count, optimizer status, and repair latency.
Optimizer convergence is distinct from barrier feasibility. Repair only checks
the short prefix; the remaining nominal plan is not certified. The console
reports residual violation counts even when MuJoCo reports no warnings.


Optimizer and safety are independent CLI groups:

```bash
python legged_mppi/scripts/simulate_mppi.py --task walk_straight_cmd_vel --mpopi --clf-cbf-qp
python legged_mppi/scripts/simulate_mppi.py --task walk_straight_cmd_vel --mbd --shield-mppi
```

Choose one optimizer (`--mppi`, `--dial-mpc`, `--cma-mppi`, `--mpopi`, `--mbd`)
and at most one safety layer (`--shield-mppi`, `--clf-cbf-qp`). Safety alone defaults
to one-pass MPPI for compatibility; no flags retain YAML optimizer settings.
Safety does not change an explicitly selected optimizer's iterations or noise
schedule. Safety flags are exposed only in the simulation CLI. Plot titles,
filenames and QP JSON distinguish the optimizer from the safety layer. Shield
retains its nominal-plan warm-start policy; optimizer covariance state is not
reset by the per-cycle repair. Successful composition is not a performance or
safety guarantee; cost scaling and covariance adaptation still require tuning.

EE tasks optimize and repair 16 commands (12 legs + arm J1–J4); J5/J6 retain home PD targets in every rollout and applied command. Height/tilt CBFs remain unchanged. Residual violations are logged and execution continues, matching Go1. See [QP and EE details](clf_cbf_qp.md).
