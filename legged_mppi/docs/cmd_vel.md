# Go1 command-velocity task

Run from the repository root in the `wb-mppi` environment:

```bash
python legged_mppi/scripts/simulate_mppi.py --robot go1 --task walk_straight_cmd_vel --mpopi
python legged_mppi/scripts/simulate_mppi.py --robot go1 --task walk_straight_cmd_vel --mppi --headless --steps 300
```

This is a continuous Go1-only task, not a waypoint task. The default target is
`[vx, vy, yaw_rate] = [0.2, 0.0, 0.0]` (m/s, m/s, rad/s). It continues until the
simulation step limit or viewer closure; proximity to the origin never completes
it. Absolute x/y position and absolute yaw are not tracked. Height (0.27 m),
yaw-free tilt, joint gait reference, and effort costs remain active.

Linear commands use a yaw-only heading frame: forward/left on the horizontal
plane, unaffected by body roll/pitch. Angular command means Euler yaw derivative,
computed from the free joint's body-frame angular velocity, not simply its z
component when tilted. Euler yaw becomes singular at vertical pitch; cost
calculation clamps the denominator for fallen samples. This is not a fall
recovery controller.

Edit the task entry in `control/tasks/definitions.py` to change the default target
or acceleration limits. The command starts at zero and ramps at
`[0.5, 0.5, 1.0]` (m/s², m/s², rad/s²). A controller client may call
`agent.set_cmd_vel(vx, vy, yaw_rate)`; the latest target takes effect with the same
rate limit on subsequent updates. Within each solve, the current ramped command
is held constant across the prediction horizon. Gait remains `walk_fast`, even
for a zero command; automatic gait switching is not implemented.

`control/configs/mppi_cmd_vel.yml` is an independent configuration. Initial
horizontal velocity weights are 1000 each and yaw-rate weight is 100; these are
starting values, not validated tracking or stability guarantees. Existing walk
configuration and waypoint task behavior are retained. The script supports the
existing solver flags. This task adds no CLF, CBF, QP, or hardware launch path.

Validation: command-cost translation/yaw invariance, non-mutating state handling,
velocity-error sensitivity, command slew limits, continuous task lifecycle,
tilt-aware yaw rate, and rejection of Go1+DM are covered in `tests/test_cmd_vel.py`.
A 300-step MPOPI smoke run completed with zero MuJoCo warnings; tracking quality
and 100 Hz wall-clock performance have not been established.

After every completed run (including early viewer closure), a tracking PNG is
saved automatically to `whole_body_mppi/outputs/`, also with `--headless` or
`--no-plot`. It shows requested commands, rate-limited references, measured
heading vx/vy and Euler yaw rate, plus tracking errors and sample RMSE. Titles
include task, robot, solver, sample budget, horizon, iterations, temperature,
timestep, sampling type, knots, seed, workers, simulated duration, and start time
in Asia/Seoul (KST). Timestamped filenames avoid overwriting previous runs.

Use `--shield-mppi` for experimental DCBF costs and bounded local repair; see
[Shield-MPPI implementation and limitations](shield_mppi.md).

Use `--clf-cbf-qp` for the separate Python/OSQP current-input filter. QP mode
starts standing and stops on any failed solve or nonlinear verification; see
[CLF-CBF-QP design, settings and failure behavior](clf_cbf_qp.md).

All simulation solver modes start standing (`home`) by default, with the nominal
plan initialized to the same pose controls. Use `--initial-key sit` to explicitly
start seated. No separate standing hold interval is inserted before walking.


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
