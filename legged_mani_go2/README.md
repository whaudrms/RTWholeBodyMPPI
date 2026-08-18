# Go2 + SO-ARM100 Whole-Body MPPI

`legged_mani_go2` is the Unitree Go2 + SO-ARM100 variant of the standalone
MuJoCo whole-body MPPI package. The gripper and gripper-roll axes are excluded.

## Robot contract

- Floating-base Unitree Go2
- 12 controlled Go2 leg joints
- 4 controlled SO-ARM100 joints: `Rotation`, `Pitch`, `Elbow`, `Wrist_Pitch`
- No controlled `Wrist_Roll`, `Jaw`, jaw fingers, or jaw actuator
- The wrist-roll motor housing is fixed at the original home angle and visible
- 16 position-target controls in total
- `nq=23`, `nv=22`, `nu=16`
- Observation: `concat(qpos, qvel)`, shape `(45,)`
- Default standing base height: `0.27 m`

Action and gait order:

```text
FR_hip, FR_thigh, FR_calf,
FL_hip, FL_thigh, FL_calf,
RR_hip, RR_thigh, RR_calf,
RL_hip, RL_thigh, RL_calf,
Rotation, Pitch, Elbow, Wrist_Pitch
```

MuJoCo body-tree order differs from this order. Controllers resolve joint
addresses by name and canonicalize rollout states before evaluating costs.

## Models

- `models/go2/go2.xml`: integrated Go2 + four-axis SO-ARM100 model
- `models/go2_so_arm_base.xml`: flat rollout model
- `models/go2_so_arm_base_big_box.xml`: big-box rollout model
- `models/go2_so_arm_base_push_box.xml`: dynamic push-box rollout model
- `models/scene*.xml`: matching simulator scenes
- `models/so_arm100/`: SO-ARM100 meshes and Apache-2.0 license

The active arm assets originate from
`/home/tony/mujoco_menagerie/trs_so_arm100`. Jaw finger meshes are not
referenced; only the wrist-roll motor housing is retained as a fixed visual.

## Gaits

Standard 32-row references live under `gaits/*/go2_so_arm/` (16 joint
positions followed by 16 joint velocities). Reachability-aware locomani uses
height-conditioned references at `0.18`, `0.225`, and `0.27 m` under
`gaits/FAST/go2_so_arm_height_conditioned/`.

Regenerate all padded and height-conditioned references with:

```bash
/home/tony/miniforge3/envs/wb-mppi/bin/python \
  legged_mani_go2/scripts/convert_gait_reference.py --mode padding --all
/home/tony/miniforge3/envs/wb-mppi/bin/python \
  legged_mani_go2/scripts/generate_locomani_gait_bank.py
```

## Run

From the repository root:

```bash
python3 legged_mani_go2/scripts/simulate_mppi.py --task stand
python3 legged_mani_go2/scripts/simulate_mppi.py --task ee_tracking
python3 legged_mani_go2/scripts/simulate_mppi.py --task locomani
python3 legged_mani_go2/scripts/simulate_mppi.py --task push_box
```

Add `--headless` for execution without the MuJoCo viewer.

## Test

```bash
PYTHONPATH=legged_mani_go2 \
  /home/tony/miniforge3/envs/wb-mppi/bin/python -m unittest discover \
  -s legged_mani_go2/tests -p 'test_*.py' -v
```
