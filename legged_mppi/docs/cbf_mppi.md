# Waypoint·cmd_vel task의 CBF on/off 비교

`simulate_mppi.py --cbf on|off`로 기존 솔버 위에 실험용 DCBF 계층을 켜고 끕니다.
기본값은 `off`입니다. Go1, Go1-DM full/simple의 기존 locomotion 위치 task와
Go1-DM full의 `ee_reach`, `ee_walk`, `ee_reach_cmd_vel`, `ee_walk_cmd_vel`를 지원합니다.
cmd_vel task에서는 EE 위치 구 조건 대신 몸체·EE 속력과 명령 속도 오차 조건을 적용합니다.
이 옵션은 시뮬레이션 진입점에 추가되며 ROS 하드웨어 실행 스크립트에는 적용되지 않습니다.

## 실행

레포 루트에서 같은 robot/task/solver/seed 설정으로 비교합니다.

```bash
python legged_mppi/scripts/simulate_mppi.py --robot go1 --task walk_straight --mpopi --cbf off --headless --steps 2000
python legged_mppi/scripts/simulate_mppi.py --robot go1 --task walk_straight --mpopi --cbf on  --headless --steps 2000

python legged_mppi/scripts/simulate_mppi.py --robot go1_dm --task walk_straight --mpopi --cbf off --headless --steps 2000
python legged_mppi/scripts/simulate_mppi.py --robot go1_dm --task walk_straight --mpopi --cbf on  --headless --steps 2000

python legged_mppi/scripts/simulate_mppi.py --robot go1_dm --task ee_walk --mpopi --cbf off --headless --steps 2000
python legged_mppi/scripts/simulate_mppi.py --robot go1_dm --task ee_walk --mpopi --cbf on  --headless --steps 2000

python legged_mppi/scripts/simulate_mppi.py --robot go1_dm --task ee_walk_cmd_vel --mpopi --cbf off --headless --steps 2000
python legged_mppi/scripts/simulate_mppi.py --robot go1_dm --task ee_walk_cmd_vel --mpopi --cbf on  --headless --steps 2000
```

`--mppi`, `--dial-mpc`, `--cma-mppi`, `--mpopi`, `--mbd` 모두 지원합니다.
결과는 기존 `outputs/<solver>/<task>/<timestamp>_<robot>/`에 저장합니다.
`summary.json`의 `cbf.enabled`와 `config.json`의 `execution.cbf`로
on/off와 실제 설정을 식별할 수 있습니다.

## 구현 범위

1. 기존 전체 horizon rollout에 유한값의 DCBF 위반 penalty를 더합니다.
2. 최종 가중평균 plan의 선두 최대 8 step을 다시 예측합니다.
3. 위반 시에만 actuator별 유한차분을 batch rollout으로 계산하고,
   입력 보정량과 slack을 변수로 하는 선형 제약 QP를 SLSQP로 한 번 풉니다.
4. 보정 후 비선형 MuJoCo 예측을 다시 실행합니다. 최악 위반이 개선된 경우에만
   보정을 채택하고, 보정한 plan을 실행·저장·warm start에 공통으로 사용합니다.
5. 제약을 만족시키지 못하면 원래 plan과 보정 plan 중 최악 위반이 작은 것을 선택하고
   `cbf_infeasible=1`로 기록합니다. 이것이 안전한 fallback이라는 보장은 없습니다.

이 구현은 **DCBF cost + 예측 입력 필터**입니다. 각 sample의 매 physics step에
몸통 feedback이나 QP를 적용하는 closed-loop rollout은 포함하지 않습니다.
Sampling distribution 변경이나 importance-weight 보정도 하지 않습니다.
샘플 수 감소·정밀도 향상·실시간 성능은 비교 실험으로 평가해야 합니다.

안전 함수 후보는 몸통 z축과 world z축의 내적으로 계산하는 45도 tilt 상한,
world z = 0.12 m의 몸통 높이 하한, 모델의 actuated hinge joint 가동 범위입니다.
높이는 지형 대비 clearance가 아닙니다. 계단 위 전도, 단차 충돌,
발 접촉·마찰·동적 균형, 장애물 회피에 대한 보장은 포함하지 않습니다.
DM의 고정 팔 관절은 보정 변수에서 제외하고 기존 command bounds를 유지합니다.

각 함수를 정규화하고 위치 여유 `h`와 속도를 포함한 `b = h + tau * h_dot`를 계산합니다.
예측 상태의 `h >= 0`, `b >= 0`, 그리고
`b_next - exp(-decay_rate * dt) * b >= 0`를 검사합니다.
최초 transition에 현재 관측 상태를 포함합니다. 이는 임의의 접촉 상태에서 실행 가능성이
증명된 HOCBF가 아니라, 속도를 반영하는 실험용 barrier 후보입니다.
MuJoCo의 soft joint limit과 초기 sit 자세 때문에 초기부터 위반할 수 있습니다.
Slack을 가진 QP가 성공해도 비선형 검증에 실패하면 infeasible로 보고합니다.

`--sync`의 검증은 solve 시점 관측 상태 기준입니다. 지연 후 도착한 plan을 실행 시점의
상태에서 다시 인증하지 않습니다. 먼저 비 sync 조건에서 비교한 다음,
sync로 계산 지연까지 포함한 동작을 평가하세요.

## EE waypoint 조건

`ee_reach`, `ee_walk`에서는 TCP 위치의 목표 오차에 대한 조건도 함께 적용합니다.
몸통·관절 조건에 더해, 현재 EE waypoint를 중심으로 줄어드는 구 내부에 TCP를 유지하도록 합니다.
초기 목표 오차가 큰 상태를 처음부터 위반으로 만들지 않도록 반경은
`rho0 = max(rho_final, 초기 EE 오차 + 0.05 m)`로 설정합니다.

```text
rho(t) = rho_final + (rho0 - rho_final) * exp(-0.5 * 경과시간)
h_ee = (rho(t)^2 - ||p_tcp - p_target||^2) / rho0^2
h_dot_ee = (2*rho*rho_dot - 2*(p_tcp - p_target)·v_tcp) / rho0^2
b_ee = h_ee + 0.05 * h_dot_ee
```

최종 반경은 task의 `ee_position_tolerance`를 사용하며 현재 기본 task에서는 0.03 m입니다.
위의 `h >= 0`, `b >= 0`, DCBF 감소율 조건을 EE에도 동일하게 적용합니다.
위치뿐 아니라 목표에서 멀어지는 속도와 허용 반경의 축소 속도를 함께 반영합니다.
이 위치 waypoint 조건에는 절대 EE 속도 상한이나 EE orientation, 장애물 충돌 조건은 포함하지 않습니다.
cmd_vel의 속도 상한은 아래 별도 조건으로 적용합니다.

좌표계는 `ee_frame`을 따릅니다. World frame에서는 몸통 운동을 포함한 절대 TCP 속도를,
base frame에서는 몸통에 대한 TCP 상대 속도를 사용합니다. 예측의 각 미래 step에는 해당
미래 시점의 반경을 적용합니다. 시간은 시뮬레이션 physics time을 사용하므로 sync에서
기존 plan을 실행하는 동안에도 반경이 계속 줄어듭니다.

반경은 solve 때마다 초기화하지 않고 실제 waypoint index·좌표·frame이 바뀔 때만
측정된 EE 오차로 재설정합니다. 직접 controller.update만 호출하는 경우에는 한 호출당
control dt를 경과시간으로 사용합니다. `off`에서는 같은 반경과 margin을 관측용으로만
기록하고 비용·입력을 변경하지 않습니다. EE task가 아닌 Go1/DM 보행에는 EE 조건을 추가하지 않습니다.

목표 도달과 반경 준수를 강제 보장하는 구현은 아닙니다. 특히 도달 불가능한 목표나
너무 빠른 반경 축소는 infeasible로 남을 수 있습니다. 기존의 settling 판정과
waypoint 전환 조건은 그대로 사용합니다.

## cmd_vel 속도 조건

`ee_reach_cmd_vel`, `ee_walk_cmd_vel`에서도 몸통 기울기·높이·관절 범위 조건을 적용하며,
다음 네 가지 속도 barrier를 추가합니다. 모두 축별 상한이 아닌 벡터의 L2 norm 기준입니다.

| 조건 | 기본 상한 | 좌표계 |
|---|---|---|
| 몸체 속력 | 0.6 m/s | body-local XY |
| EE 속력 | 0.6 m/s | world XYZ, 몸체 운동 포함 |
| 몸체 명령 속도 오차 | 0.25 m/s | body-local XY |
| EE 명령 속도 오차 | 0.25 m/s | world XYZ |

속도 `v`와 상한 `v_max`에 대해 `h_speed = 1 - ||v||²/v_max²`,
명령 `v_cmd`와 오차 상한 `e_max`에 대해
`h_error = 1 - ||v - v_cmd||²/e_max²`를 사용합니다.
속도 수준 조건에는 가상의 가속도 미분을 넣지 않고 `b = h`로 두어,
MuJoCo가 예측한 다음 속도로 동일한 DCBF 감소율 조건을 검사합니다.
몸체 z축 선속도·각속도에 별도 속력 상한은 추가하지 않았습니다.

현재 시점뿐 아니라 예측 미래 시점의 command phase를 조회합니다. 명령이 불연속으로
바뀌는 경계에서는 오차 barrier가 급변하므로 DCBF 조건이 infeasible할 수 있습니다.
이는 slack·비선형 재검증·기존 infeasible 지표로 처리하며, 명령을 자동으로 변경하지 않습니다.
상한은 실험용 설정이며 로봇의 인증된 속도 한계가 아닙니다.

cmd_vel에 위치 목표·위치 feedback·EE 위치 반경은 추가하지 않습니다. 정지 명령도
속도 오차 기준이며 이전 위치로 복귀시키지 않습니다. 기존 속도 추종 비용은 계속 작동합니다.
`off`는 동일한 속도 margin을 관측용으로만 기록합니다.

## 파라미터

Task YAML에 선택적으로 `cbf` mapping을 추가할 수 있습니다. 생략한 값의 기본값은 다음과 같습니다.

```yaml
cbf:
  tilt_degrees: 45.0
  min_world_height: 0.12
  decay_rate: 10.0
  lookahead_seconds: 0.05
  joint_lookahead_seconds: 0.02
  penalty_weight: 2000.0
  shield_steps: 8
  correction_limit: 0.12  # actuator별 보정 상한 [rad]
  finite_difference: 0.002
  slack_weight: 10000.0
  tolerance: 0.00001
  ee_radius: null            # null: task의 ee_position_tolerance 사용 [m]
  ee_initial_margin: 0.05    # 초기 목표 오차에 더하는 여유 [m]
  ee_convergence_rate: 0.5   # 반경 축소율 [1/s]
  ee_lookahead_seconds: 0.05
  body_speed_max: 0.6           # cmd_vel: body-local XY norm [m/s]
  ee_speed_max: 0.6             # cmd_vel: world XYZ norm [m/s]
  body_velocity_error_max: 0.25 # cmd_vel: XY command-error norm [m/s]
  ee_velocity_error_max: 0.25   # cmd_vel: XYZ command-error norm [m/s]
```

## 비교 지표

- 기존 몸통·EE tracking error, solve time, deadline miss를 비교합니다.
- `cbf_margin`: 실제 상태에서 정규화한 `h`, `b`의 최솟값입니다. on/off 모두 기록합니다.
- `cbf_ee_margin`: EE 조건만의 `h`, `b` 최솟값입니다. 음수이면 해당 EE 조건 위반입니다.
- `cbf_ee_radius_m`: 해당 시점 EE 목표 오차의 허용 반경입니다.
  NPZ에서는 `cbf_ee_radius`라는 이름으로 저장하며, EE 위치 waypoint task가 아니면 NaN입니다.
- `cbf_body_speed_margin`, `cbf_ee_speed_margin`: cmd_vel의 속력 barrier 값입니다.
- `cbf_body_velocity_error_margin`, `cbf_ee_velocity_error_margin`: 명령 속도 오차 barrier 값입니다.
  네 값을 NPZ의 `cbf_velocity_margins`에 위 순서대로 저장합니다. 음수이면 위반입니다.
- `cbf_body_speed_m_s`, `cbf_ee_speed_m_s`: 실제 속력입니다.
  NPZ에서는 각각 `cbf_body_speed`, `cbf_ee_speed`로 저장하며,
  summary의 `cbf.velocity`에는 속력 통계와 조건별 최소 margin·위반 비율을 저장합니다.
- `cbf_verified_residual`: solve 시점 단기 예측의 최소 제약값입니다.
- `cbf_correction_rad`: 첫 입력 보정 벡터의 L2 norm입니다.
- `cbf_infeasible`: 비선형 검증 후에도 제약 위반이 남은 solve입니다.
- `cbf_extra_rollouts`: 솔버의 `n_samples` **외에** 추가한 예측 본수입니다.
  Nominal이 조건을 만족하면 1개, 위반 시 보통 `자유 actuator 수 + 2`개입니다.
- `cbf_shield_ms`: 필터 계산시간입니다. 모든 CBF 계산은 `solve_ms`에 포함됩니다.
- Solve 단위 지표는 solve가 없는 시점과 off일 때 NaN입니다.

위 지표는 `metrics.csv`, `trajectory.npz`에 저장하고 `summary.json`의 `cbf`에 집계합니다.
기존 `predicted_horizon_cost`는 비교를 위해 task cost만 표시하며 sampling 시 추가한 CBF penalty를
포함하지 않습니다. 같은 `n_samples`에서도 on은 추가 계산을 하므로 solve time과 추가 rollout 수를
반드시 함께 비교하세요. 이 초기 버전은 샘플 예산을 자동으로 조절하지 않습니다.

```bash
PYTHONPATH=legged_mppi python -m unittest discover -s legged_mppi/tests -p 'test_cbf.py'
```
