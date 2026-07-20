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
