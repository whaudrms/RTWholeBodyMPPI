# Go1 CLF-CBF-QP input filter

```bash
conda activate wb-mppi
python legged_mppi/scripts/simulate_mppi.py --task walk_straight_cmd_vel --clf-cbf-qp --headless --steps 300
```

Python OSQP 1.x is required (`pip install 'osqp>=1.0,<2'`). This independent safety option filters only
the selected optimizer’s current 12 joint position commands. No CBF/CLF sample cost is added. Future
commands are unchanged. Existing `--shield-mppi` retains its own algorithm.

The combined CLF is `V = 0.5 * sum((error / velocity_scale)**2)`, using heading
vx/vy and Euler yaw rate. The current rate-limited command is frozen for the
one-step check. It requests `V_next <= exp(-decay_rate*dt)*V_current + slack`.
The single nonnegative slack is penalized, not bounded. This is a CLF candidate
for velocity tracking, not a certificate of full robot stability.

Height/tilt barriers and alpha are read directly from `shield_mppi` in
`control/configs/mppi_cmd_vel.yml`; no duplicated CBF limits. At implementation
time these were 0.18 m and 10 degrees. These limits remain experimental.

Variables: 12 joint command corrections and one CLF slack. Objective:
`0.5 * sum((delta_u/max_correction)**2) + 0.5*slack_weight*slack**2`.
MuJoCo one-step transitions are linearized about the nominal MPPI command using
bounded central finite differences (one-sided at actuator limits). Constraints:

- Linearized aggregate CLF with slack.
- Two hard linearized DCBF constraints.
- Actuator command bounds and per-joint correction trust region.
- Nonnegative slack.

The prototype sets up an OSQP problem each cycle and warm-starts its primal
variables from the previous accepted solution. There are 25 batched prediction
rollouts for the nominal/Jacobian and one nonlinear verification rollout.
Only OSQP `solved` status is accepted; linear constraints, finite values, and
nonlinear CBF residuals and next-state barrier margins are checked
within the configured normalized tolerance. Numerical bound clipping precedes
nonlinear re-evaluation. No retries, relaxed CBF fallback, torque/rate constraints,
or wall-clock deadline are implemented. The correction bound is NOT an input
change-rate limit relative to the previous applied command.

CLF is soft: nonlinear CLF mismatch does not reject the command. Logs retain
`slack`/`qp_slack` and additionally record `actual_required_slack`,
`clf_slack_excess`, `clf_current`, `clf_next`, `clf_decay_satisfied` (with tolerance)
and `clf_decreased` (actual non-increase). These diagnostics do not certify CLF
decay. QP constraints still include and penalize the CLF slack.

**Solver, input constraint, nonfinite prediction or CBF failure stops the
simulation before applying the rejected command.** The CLI
exits with code 2. The partial tracking PNG and `.qp.json` are saved in
`whole_body_mppi/outputs/`. JSON includes history, configuration, failed current
state, nominal/candidate command, command reference, solver status and residuals.
No previous or nominal command is silently substituted. This simulation stop is
not a hardware emergency-stop controller; the option supports the Go1 command task and Go1+DM EE velocity tasks in simulation.

All simulation modes default to the standing `home` keyframe unless the task
specifies another initial key. The nominal planner is initialized from the
selected pose controls.
`--initial-key sit` tests failure from the existing 0.08 m sitting pose, below the
0.18 m barrier. Use `--initial-key home` explicitly for pose-matched comparisons
with other modes.

Historical validation before separating soft CLF diagnostics from hard safety
checks: standing initialization accepted the first
command, then stopped at 0.01 s because the nonlinear CLF residual exceeded its
QP slack by approximately 0.000341 (tolerance 0.0001); CBF residuals were positive.
Sitting initialization produced primal infeasibility and stopped at 0 s. These
are intentionally reported failures, not evidence of successful sustained
tracking. Trust region, linearization and CLF settings require further study;
the checks were not loosened to obtain a longer run. Unit tests exercise an
accepted solve, unchanged future controls, infeasible CBF, failed nonlinear
verification, aggregate CLF values and nonfinite input rejection.

OSQP Python API reference: https://osqp.org/docs/interfaces/python.html

After separating soft CLF diagnostics from hard CBF acceptance, a 300-step
standing-start run completed 3.0 simulated seconds with all 300 commands accepted
and zero MuJoCo warnings. This confirms the previous immediate-stop regression
is resolved; it does not establish sustained tracking accuracy or CLF decay.


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

## Go1+DM EE velocity tasks

`ee_reach_cmd_vel` and `ee_walk_cmd_vel` support both safety options with every optimizer.
MPPI and the safety layer optimize 16 commands: 12 legs and arm J1–J4.
J5/J6 commands are fixed at home throughout sampling, prediction and application.
The physical model retains 18 actuators; fixed wrist commands are PD targets, not welded joints.

QP variables are 16 corrections plus independent body and EE nonnegative slacks.
The objective adds `0.5*ee_slack_weight*ee_slack**2` to the body slack and correction penalties.
Each CLF has its own linearized decay constraint. The EE CLF is
`0.5*sum(((v_ee-command)/ee_velocity_scale)**2)` in the task's velocity frame
(world for reach; trunk-relative for walk). Its command is frozen for the current check.
There are 33 nominal/Jacobian predictions plus one nonlinear verification.
The two hard height/tilt CBFs and failure policy are unchanged. No arm obstacle CBF is added.
Each EE YAML supplies its own `shield_mppi` limits and `clf_cbf_qp` settings;
initial values match Go1. EE diagnostics use the `ee_` prefix.

```bash
python legged_mppi/scripts/simulate_mppi.py --robot go1_dm --task ee_walk_cmd_vel --mpopi --clf-cbf-qp
python legged_mppi/scripts/simulate_mppi.py --robot go1_dm --task ee_reach_cmd_vel --dial-mpc --shield-mppi
```
