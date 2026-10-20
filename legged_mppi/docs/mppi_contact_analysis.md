# Contact-dependent rollout timing analysis

Historical measurements before the shared-budget change: these results used
`n_samples` per iteration. Current code uses it as a total control-cycle
budget; the old 3x budget comparison below no longer describes current code.

Measured on the current host using MuJoCo 3.1.6 and the current
`walk_straight` settings: dt=0.01, horizon=40, n_samples=10, n_iterations=3,
n_workers=5, seed=42. Nominal overrides iterations to one. No solver, cost,
reference, model or configuration code was changed for this analysis.
Raw measurements: [mppi_contact_profile.json](mppi_contact_profile.json).

## Findings

1. Nominal evaluates 10 trajectories / 400 physics steps. Each iterative
   solver evaluates 30 trajectories / 1200 physics steps per update. Stages
   depend on preceding stages and run sequentially. The default comparison
   therefore starts with a threefold rollout budget difference.
2. All modes use the same `MujocoRollout` backend. There is no mode-specific
   slow physics path for identical models, states and candidate controls.
3. Contact/constraint difficulty can multiply the cost of each physics step.
   The model uses pyramidal cones and foot condim=6, which corresponds to ten
   constraint dimensions per active foot contact. Other contacts have different
   dimensions. The meaningful workload indicator is nefc plus solver effort,
   not ncon alone. Measured model options: Newton, max iterations=100,
   line-search max=50, tolerance=1e-8, auto Jacobian, impratio=100.
4. Statically partitioned worker chunks introduce straggler latency. With
   N=10 and W=5 each worker gets two complete trajectories. Every stage waits
   for every worker. One difficult candidate can hold up a stage; three
   stages provide three such barriers. This is a structural risk, not a
   measured attribution of a particular timing spike.
5. Changes in candidate distributions and actual controlled states can change
   per-step physics difficulty. CMA adapts variance, MPOPI learns joint
   correlation, and MBD initially samples at twice base standard deviation
   with alpha_bar_last=.2. These can change contact histories, but the fixed
   state measurements did not show a systematic per-stage penalty for these
   methods. DIAL noise shrinks, so broader exploration cannot explain DIAL
   slowdown in general.

## Fixed-state complete-update measurements

A crouched keyframe was given the indicated base height with unchanged joints
and velocities. These are synthetic stress states, including floor penetration,
not recorded user failure states. Each mode/state was instantiated five times
with the same seed and warm start. Worker startup and a warm-up rollout were
outside timing. Values are medians of summed native batch rollout wall time;
cost evaluation and proposal updates were timed separately. Candidate traces
from one repetition were replayed with Python mj_step for constraint statistics,
separately from the native batch timings.

| Initial state | Nominal | DIAL | CMA | MPOPI | MBD |
| --- | ---: | ---: | ---: | ---: | ---: |
| z=.27, initial ncon=0 | 5.76 ms | 16.83 ms | 16.36 ms | 17.03 ms | 16.00 ms |
| z=.07, initial ncon=6 | 6.53 ms | 18.99 ms | 18.91 ms | 18.88 ms | 18.01 ms |
| z=0, initial ncon=74 | 6.63 ms | 18.68 ms | 18.02 ms | 18.87 ms | 18.35 ms |
| z=-.05, initial ncon=108 | 5.98 ms | 18.04 ms | 17.99 ms | 17.13 ms | 18.00 ms |

The ratio is roughly 2.7–3.0 rather than a new large contact-dependent multiplier
specific to iterative algorithms. Initial high contact counts are transient:
for z=0, nominal's horizon-average ncon is 6.67, not 74. In the deeper-penetration
case the robot is rapidly ejected from contact. Thus maximum/initial contact
count is not a reliable proxy for the integrated horizon workload.

## Isolating the cost of a contact-rich step

Reset the same crouched state before every step, set the same initial action,
and measure only mj_step. Discard ten warm-up steps and take the median of
240 measurements. This deliberately prevents the state from escaping contact;
it is a computational stress test, not a physically valid trajectory.

| ncon | nefc | Newton iterations | mj_step median |
| ---: | ---: | ---: | ---: |
| 0 | 16 | 1 | 18.1 us |
| 6 | 64 | 6 | 51.0 us |
| 74 | 336 | 27 | 429.1 us |
| 108 | 472 | 22 | 524.7 us |

Going from 6 to 74 contacts costs 8.4x per step in this constructed state.
A trajectory sustaining such states can make a rollout much more expensive.
The relationship is not a fixed power law: active sets, friction, solver
convergence and factorization work also matter. Non-contact joint friction
and limit constraints contribute to nefc as well.

## Closed-loop comparison

150 updates per mode using the actual simulation scene and crouched initial
keyframe, same 10 ms simulation step and task goal advancement; no viewer,
plotting, or selected-plan evaluation. Each mode follows its own resulting
state trajectory, so these are not controlled equal-state comparisons.
The first frame includes worker startup and is retained in the maximum.

| Mode | Rollout median | Rollout p95 | Complete update median |
| --- | ---: | ---: | ---: |
| Nominal | 3.60 ms | 5.91 ms | 5.26 ms |
| DIAL | 16.42 ms | 20.94 ms | 22.42 ms |
| CMA | 17.91 ms | 24.97 ms | 24.60 ms |
| MPOPI | 16.79 ms | 21.70 ms | 23.46 ms |
| MBD | 15.43 ms | 20.15 ms | 22.13 ms |

Rollout median ratios rise to 4.3–5.0x when each controller generates its own
trajectory. This demonstrates that the threefold sample count alone is not
sufficient to predict closed-loop wall time. It does not establish that all
additional time is due to contact count: state, candidate histories, scheduling,
and solver convergence differ. No persistent extreme-contact runtime spike
was reproduced in this short test. A snapshot of the user's actual slow state
and candidates is needed to attribute that particular spike.

## Other sources of apparent rollout time

`Simulator.store_trajectory` calls `eval_best_trajectory` every simulation frame.
That function performs one additional horizon rollout outside controller
update. It is a diagnostic evaluation, common to all modes, and can be costly
in contact-rich states. Timing the viewer loop also includes this evaluation,
plant stepping, rendering and frame saving. None is included in the native
rollout columns above. The locomotion objective itself has no per-contact loop.
MPOPI eigendecomposition and MBD proposal/score work run outside native batch
rollout; they affect complete-update timing rather than physics-only timing.

## Recommended investigation order

- Compare equal total trajectory budgets first, e.g. nominal N=30 versus
  iterative N=10, I=3. The current N=10 versus N=10,I=3 comparison is not equal
  compute. Thread and barrier overhead still differs at equal budget.
- Record per-stage rollout/cost/proposal time, and per-worker duration for the
  actual slow state. Replay captured candidate sequences for ncon, nefc,
  solver_niter, penetration depth and body contacts. Use their horizon profiles,
  not only the plant's current contact counter.
- Test finer-grained trajectory scheduling to reduce chunk stragglers, while
  checking dispatch overhead. Preserve dynamics/costs during this experiment.
- Profile selected-plan diagnostic evaluation separately and reduce its
  frequency if plotting fidelity permits.
- Only after identifying physics bottlenecks, benchmark solver tolerance,
  iteration caps, cone/condim, and contact geometry. These change numerical
  accuracy or contact physics and should not be silently applied as timing
  optimizations. Use iteration statistics before assuming the cap of 100
  is actually reached; this test typically needed substantially fewer.

MuJoCo 3.1.6 references:
[contact dimensions and constraint solver](https://mujoco.readthedocs.io/en/3.1.6/computation/index.html),
[solver selection and performance](https://mujoco.readthedocs.io/en/3.1.6/modeling.html).
