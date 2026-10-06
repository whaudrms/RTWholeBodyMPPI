# B2-Z1 standalone Whole-Body MPPI

`legged_mani` is a ROS-free MuJoCo + MPPI package for the Unitree B2 with a
reduced 4-DoF Z1 arm. It does not import code or runtime assets from
`legged_mppi`, `/home/tony/robots`, or `wb-mpc-locoman`.

The first integrated task is `sit_hold`: maintain the supplied seated pose in
a closed MuJoCo loop. The locomotion path now also includes an adapted dynamic
in-place gait; end-effector objectives remain a later stage.

Registered tasks are defined in `utils/tasks.py`:

- `sit_hold`: maintain the seated keyframe.
- `stand_hold`: maintain the standing keyframe with its own exploration and
  state-cost tuning.
- `in_place`: run a cyclic 32-row dynamic joint reference adapted from the
  original SLOW 0_0 10 cm gait and mean-aligned to the B2 `stand` keyframe.

The hold tasks use the locomotion-controller cost form without a gait scheduler.
At the first MPPI update, the complete current 45-D observation is copied,
its velocity targets are set to zero, and it remains the hold reference. The stage cost combines
quaternion/state error, an L1 base-position term, and the original virtual PD
effort term `kp * (u - q) - kd * dq`, expanded to all 16 actuated joints.

Only tasks with an implemented B2-Z1 reference and cost are registered. Future
walking and manipulation tasks can be added to the same registry when their
references are available.

## Directory layout

```text
legged_mani/
├── configs/                 MPPI sampling and cost parameters
├── control/                 modified base_controller and B2-Z1 task cost
├── interface/               MuJoCo environment and closed-loop simulator
├── models/                  self-contained MJCF and mesh assets
├── scripts/                 headless run and model viewer entry points
├── tests/                   environment and MPPI integration checks
├── utils/                   named B2-Z1 state layout
├── environment.py           compatibility import for earlier code
├── smoke_test.py            compatibility test entry point
└── viewer.py                compatibility viewer entry point
```

The sampling, threaded rollout, and receding-horizon structure in
`control/base_controller.py` is adapted directly from
`legged_mppi/whole_body_mppi/control/controllers/base_controller.py`. Go1-only
dimensions and limits were removed; `nu`, actuator control ranges, and initial
controls now come directly from the B2-Z1 MuJoCo model.

`control/mppi_locomotion.py` retains the original source-level `MPPI` class and
method flow (`update`, `quadruped_cost_np`, `calculate_total_cost`, and
`eval_best_trajectory`). Its Go1-specific imports and 12-joint state slices are
adapted to the local B2-Z1 model, 16 controls, 45 states, and 32-row scheduler.
The original transform helpers are retained in `utils/transforms.py` with
shape validation and a zero-distance orientation guard.

## Model and solver contract

- Active joints: 12 B2 leg joints + Z1 `joint1` through `joint4`.
- Fixed joints: Z1 `joint5`, `joint6`, and gripper.
- Action: 16 desired joint angles, not raw torques.
- Observation: `concat(qpos, qvel)`, shape `(45,)`.
- FULLPHYSICS rollout: `[time, qpos, qvel]`, shape `(46,)`.
- Default keyframe: `sit`, base height `0.1502 m` and leg angles
  `(hip, thigh, calf) = (0, 1.28, -2.8)`.
- MPPI batch: 30 samples × 40 steps at `dt=0.01 s` by default.

Action order:

```text
FR_hip, FR_thigh, FR_calf,
FL_hip, FL_thigh, FL_calf,
RR_hip, RR_thigh, RR_calf,
RL_hip, RL_thigh, RL_calf,
joint1, joint2, joint3, joint4
```

State order:

```text
0:3    base position          23:26  base linear velocity
3:7    base quaternion        26:29  base angular velocity
7:19   leg positions          29:41  leg velocities
19:23  arm positions          41:45  arm velocities
```

The arm remains part of the 16-D action and 45-D state cost. Its exploration
noise is initially zero in `configs/mppi_sit_hold.yml`, so the first solver
integration optimizes leg targets while holding the supplied arm pose. Arm
noise can be enabled after adding an end-effector task cost.

## Run

From the `RTWholeBodyMPPI` repository root:

```bash
python3 -m legged_mani.tests.test_environment
python3 -m legged_mani.tests.test_mppi
python3 -m legged_mani.tests.test_locomotion
python3 -m legged_mani.scripts.view_model
python3 -m legged_mani.scripts.visualize_gait_trajectory
python3 -m legged_mani.scripts.simulate_mppi --list-tasks
python3 -m legged_mani.scripts.simulate_mppi --task sit_hold
python3 -m legged_mani.scripts.simulate_mppi --task stand_hold
python3 -m legged_mani.scripts.simulate_mppi --task in_place
python3 -m legged_mani.scripts.simulate_mppi --task big_box
```

To validate a 32-row gait reference without controller or contact-dynamics
effects, replay its joint angles with the floating base fixed at the `stand`
keyframe. The viewer draws the complete foot-site paths in four colors and a
z=0 reference grid. Press Space to pause, use Left/Right to step through gait
samples, and press R to return to the first sample.

```bash
python3 -m legged_mani.scripts.visualize_gait_trajectory \
  --model legged_mani/mani_mppi/models/b2_z1_4dof.xml \
  --gait legged_mani/mani_mppi/control/gait_scheduler/gaits/FAST/b2_z1/walking_gait_raibert_FAST_0_0_10cm_80hz.tsv \
  --rate 80

# Numerical validation only, with optional exported xyz foot trajectories.
python3 -m legged_mani.scripts.visualize_gait_trajectory \
  --headless --save-trajectory /tmp/b2_foot_trajectory.tsv
```

`simulate_mppi` runs MuJoCo at 100 Hz. Hold tasks update MPPI every five
simulation steps (20 Hz), while `in_place` defaults to one step (100 Hz) to
match the original gait and controller timing. Scheduler phase and MPPI
warm-start advance by the number of controls actually applied, including when
`--control-decimation` overrides the task default.

The source-equivalent 100 Hz `in_place` timing is computationally slower than
wall-clock real time with the current 30 × 40 batch (about 32 ms per update on
the validated CPU). For real-time visualization at reduced optimization rate,
use `--control-decimation 5`; this preserves phase alignment but performs one
MPPI optimization per five physics steps. Pass `--duration 0` to run until the
viewer window is closed.

```bash
# Source-equivalent 100 Hz MPPI timing (default, slower wall-clock playback)
python3 -m legged_mani.scripts.simulate_mppi --task in_place

# 20 Hz MPPI / 100 Hz physics visualization with aligned phase and warm-start
python3 -m legged_mani.scripts.simulate_mppi \
  --task in_place --control-decimation 5
```

Run without a viewer for automated checks, or display plots after a run:

```bash
python3 -m legged_mani.scripts.simulate_mppi \
  --task sit_hold --headless --steps 100
python3 -m legged_mani.scripts.simulate_mppi \
  --task stand_hold --duration 10 --plot
```

Install the local Python dependencies if needed:

```bash
python3 -m pip install -r legged_mani/requirements.txt
```

## Data provenance

- B2 dynamics and OBJ meshes: `/home/tony/robots/b2_description_mujoco`
- Z1 inertial, kinematic, joint-limit, and mesh data:
  `/home/tony/robots/z1_description`
- B2-Z1 mount transform and reduced arm reference:
  `/home/tony/wb_mpc_ws/wb-mpc-locoman/robots/b2_z1_description`

All required runtime assets were copied beneath `models/assets`.

## Next integration stages

1. Replace the adapted Go1-derived in-place reference with a gait generated or
   optimized directly for B2 dynamics.
2. Add `sensordata` rollout output and an end-effector pose cost using the
   existing `ee_pos` and `ee_quat` sensors.
3. Enable nonzero arm exploration noise and tune whole-body weights.
4. Reduce sample count/horizon or optimize the cost path if a 100 Hz control
   loop is required; the current validated CPU setup is approximately 26 Hz.

## Modeling assumptions

- The B2 base uses the valid 35.86 kg principal inertia from Unitree's
  simulation-ready B2 MJCF.
- Position-PD gains are initial values from the supplied Gazebo controller
  configurations and still require task-level tuning.
- The default world contains only the robot and floor; the optional
  `big_box` task adds the matching box geometry to simulator and MPPI models.

## Locomani Hermite targets and annealing

From the repository root, with the `wb-mppi` environment active:

```bash
# Hermite position/velocity targets + three refinement iterations
python legged_mani/scripts/simulate_mppi.py --task locomani \
  --rollout-mode hermite --anneal-iterations 3 --plot

# Hermite only (no iteration or horizon annealing)
python legged_mani/scripts/simulate_mppi.py --task locomani \
  --rollout-mode hermite --anneal-iterations 1 --horizon-noise-factor 1

# Original cubic single-pass baseline
python legged_mani/scripts/simulate_mppi.py --task locomani \
  --rollout-mode original_spline --anneal-iterations 1 --horizon-noise-factor 1

# Cubic + annealing
python legged_mani/scripts/simulate_mppi.py --task locomani \
  --rollout-mode original_spline --anneal-iterations 3
```

These options currently apply to `locomani`, not the push-box or EE-tracking
controllers. `--headless` disables the viewer. The existing gait, IK, CEM and
cost settings are retained. The default rollout mode remains `original_spline`;
locomani YAML now defaults to three annealing iterations.

The sampler uses cubic Hermite position and derivative nodes with separate
velocity noise. Derivative bounds keep the equivalent Bezier control points
inside the joint-target ranges, bounding the entire interpolated position curve.
This does not enforce intermediate joint-velocity or acceleration constraints.
The standard deviation at iteration i and node j is scaled by
`anneal_factor**i * horizon_noise_factor**(K-1-j)`. YAML defaults are 0.5 and
0.9; `hermite_velocity_noise_scale: 4.0` converts the position noise scale to
velocity-node noise (1/s). The previous best candidate is included in each
subsequent iteration, and the weighted mean is separately rolled out before
it can replace the best execution plan.

For existing affine MuJoCo PD actuators, velocity feedforward is encoded as
`ctrl = q_ref + (Kd/Kp)*dq_ref`. Thus the actuator produces
`Kp*(q_ref-q) + Kd*(dq_ref-dq)`, subject to its existing force limits. The
controller and Simulator both disable clipping of this **encoded** ctrl to the
position-target range. Actual position references remain spline-bounded.
`selected_position_targets` and `selected_velocity_targets` expose the physical
PD targets; `selected_trajectory` and `update()` contain encoded actuator controls.
A hardware bridge must send the two physical targets with matching gains, not
interpret the encoded controls as desired joint positions.

`last_timing` aggregates each phase across all iterations;
`iteration_diagnostics` records noise scale, selected cost and selection source.
Thirty candidates and three iterations require 90 batch trajectory evaluations
plus three mean validations per update. A configured 100 Hz loop is not a
measured real-time guarantee; compare wall-clock latency as well as task error.

### CPU planning budget (20 ms target)

Measured results are recorded in [benchmarks/rt20_cpu_results.json](benchmarks/rt20_cpu_results.json).
On this i7-12700H, consecutive 500-update comparisons (first 10 excluded) gave:

| Configuration | Median | p95 | Scenario |
| --- | ---: | ---: | --- |
| Before these optimizations, 30 samples x 3 iterations | 112.7 ms | 135.4 ms | 500 steps, goal index 4 |
| Optimized default, same search settings | 101.6 ms | 117.5 ms | 500 steps, goal index 4 |

The separate budgeted rt20 trial measured 36.3 ms median / 42.3 ms p95 and
stopped at step 1608 with no valid candidate. The 20 ms target is **not met**.
These are single-run measurements; asynchronous CEM and CPU scheduling cause
trajectory and latency variation.

The `rt20` profile is experimental. Long scenario runs on the tested CPU did
not satisfy both the deadline and the original task completion performance;
the fixed-iteration configuration remains the default for scenario replay.

```bash
# Default fixed-iteration search with collision/call-overhead optimizations
python legged_mani/scripts/simulate_mppi.py --task locomani --rollout-mode hermite

# Scenario replay, Hermite + budgeted refinement
python legged_mani/scripts/simulate_mppi.py --task locomani \
  --rollout-mode hermite --performance-profile rt20

# Headless solve-latency and scenario-quality measurements
python legged_mani/scripts/benchmark_locomani.py --performance-profile rt20 \
  --steps 2000 --output /tmp/locomani_rt20.json
python legged_mani/scripts/benchmark_locomani.py \
  --steps 2000 --output /tmp/locomani_default.json
```

`rt20` is an opt-in **soft planning budget**, not a guarantee that every solve
finishes in 20 ms. It requires Hermite and retains 30 candidates, all 40 horizon
steps (0.4 s), and validated weighted-mean execution. Smaller candidate sets
were rejected during tuning because scenario progress and validity degraded.
The profile uses 10 persistent rollout workers with one trajectory per job.
It enables a conservative capsule-AABB broad phase: pairs proven clear avoid
fine sampling; ambiguous pairs retain the existing exact-distance refinement.
Clearance diagnostics for broad-phase accepted pairs are lower bounds.
Single-plan validation runs inline with private thread-local MuJoCo data,
avoiding thread-pool submission overhead.

At least one full batch is evaluated from the current observation on every
update. Further annealing iterations, up to `--anneal-iterations` (default 3),
start only if elapsed time plus 1.5 times the recent iteration estimate plus
3 ms fits the 20 ms budget. The estimate includes weighted-mean validation.
An already-running MuJoCo batch cannot be interrupted by this budget, so even
the first iteration can exceed it. `budget_limited`, `deadline_missed`,
`last_timing`, and `iteration_diagnostics` expose the actual behavior.

Without the profile, the existing fixed-iteration search remains the default;
the conservative broad phase is enabled in locomani YAML and inline single-plan
validation also applies.
YAML `planning_budget_ms: 0` disables the budget. `validate_mean: true` preserves
separate mean verification. For ablations, `validate_mean: false` executes only
evaluated samples and includes the previous mean in the next regular batch;
an untested average is never executed. The rt20 profile keeps validation on.

The benchmark reports effective settings, initialization time, all-update and
steady-state median/p95/p99/max, deadline misses, completed goals, EE error,
base height/tilt, and failures. Steady-state statistics omit the first 10 updates;
raw per-update records are retained. Reported latency measures `agent.update()`;
it excludes GUI, physics stepping, and goal-transition handling. CEM work that
overlaps a solve can still affect its latency. The benchmark and scenario are
synchronous simulations with 10 ms simulated steps; this change does not turn
them into a wall-clock 50 Hz controller or compensate for observation delay.
Check long-run scenario quality as well as timing before choosing a profile.
On the tested i7-12700H, Linux CPU IDs 0-11 are P-core threads. `taskset -c 0-11`
can be prepended to a benchmark command for a separate affinity experiment;
this alone did not meet the long-run target. CPU numbering is machine-specific,
and the controller never changes system power settings or process affinity.

Regression tests:

```bash
PYTHONPATH="$PWD/legged_mani${PYTHONPATH:+:$PYTHONPATH}" \
  python -m unittest discover -s legged_mani/tests -p 'test_hermite_annealing.py' -v

PYTHONPATH="$PWD/legged_mani${PYTHONPATH:+:$PYTHONPATH}" \
  python -m unittest discover -s legged_mani/tests -p 'test_realtime_search.py' -v
```
