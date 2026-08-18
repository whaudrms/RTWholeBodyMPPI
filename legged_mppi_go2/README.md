# legged_mppi_go2

Go2-specific copy of the quadruped MPPI package. All simulation tasks resolve
to MJCF scenes under `whole_body_mppi/models/go2/`; the retained `models/go1/`
directory is archival and is not referenced by active task definitions.

## Simulation

```bash
python3 scripts/simulate_mppi.py --task stand
python3 scripts/simulate_mppi.py --task walk_straight
```

Available Go2 scenes:

- `scene.xml`: flat floor
- `scene_big_box.xml`: static large box
- `scene_stairs.xml`: staircase
- `scene_push_box.xml`: dynamic push box
- `scene_climb_box.xml`: hardware-style climb box

## Model conventions

MPPI actions use `FR, FL, RR, RL`, with hip, thigh, and calf for each leg.
MuJoCo stores the Go2 body tree as `FL, FR, RL, RR`; the controller maps rollout
states back into MPPI actuator order before evaluating costs.

The bundled 24-row gait TSV files are the legacy gait bank. The Go2 MJCF keeps
compatible joint coordinates and limits so these references can be used as an
initial baseline. Go2-optimized gait references should replace them before
performance tuning or hardware deployment.

## Hardware note

The ROS scripts default to `/mocap_node/Go2_body/Odom`, but the actual Go2
low-level state/command bridge and mocap topic must match the deployment.
