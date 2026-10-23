# Go1 + reBot B601-DM

기존 Go1 task에 DM 팔을 부착한 MuJoCo 모델입니다. `--rollout-model full`은 팔 6축을 유지하는 기존 단순 도형 모델이고, `simple`은 팔을 home 자세의 고정 하중으로 합친 더 가벼운 모델입니다. 기본값은 `full`입니다. 두 옵션은 예측·물리 시뮬레이션에 적용되고, 화면에는 두 옵션 모두 원본 상세 DM 메시를 표시합니다. MPPI 최적화 대상은 기존 다리 12축이며 샘플 수·비용함수·horizon은 그대로입니다.

| 옵션 | 팔 표현 | nq / nv / nu | 용도 |
|---|---|---|---|
| `full` (기본) | 팔 6축 + 고정 손가락, 단순 도형 | 25 / 24 / 18 | 팔의 관절 운동과 반작용 포함 |
| `simple` | home 자세의 하나의 고정 하중 | 19 / 18 / 12 | 팔을 움직이지 않는 보행 |

여기서 `full`은 원본 상세 메시 모델을 의미하지 않습니다. 원본 상세 모델은 아래 독립 preview로 확인할 수 있습니다.

## 실행

MuJoCo가 설치된 환경에서 레포 루트 기준:

```bash
python legged_mppi/scripts/simulate_mppi.py --robot go1_dm --task walk_straight --mpopi --rollout-model full
python legged_mppi/scripts/simulate_mppi.py --robot go1_dm --task walk_straight --mpopi --rollout-model simple
python legged_mppi/scripts/simulate_mppi.py --robot go1_dm --task walk_straight --mpopi --headless --steps 1000
PYTHONPATH=legged_mppi python -m unittest discover -s legged_mppi/tests -p 'test_go1_dm*.py' -v
```

`--robot` 기본값은 `go1`입니다. 기존 locomotion task의 지형·목표·gait·MPPI 설정을 보존하며 `--mppi`, `--dial-mpc`, `--cma-mppi`, `--mpopi`, `--mbd`를 사용할 수 있습니다. task별 성공을 보장하는 추가 튜닝은 포함하지 않습니다.

`--steps`는 10 ms 제어 주기 수로 기본 2000은 20초입니다. `--headless`는 viewer와 최종 plot을 끄고, `--no-plot`은 최종 plot만 끕니다.

## Go1과 동일하게 맞춘 실행 조건

| 항목 | go1 | go1_dm 기본값 |
|---|---|---|
| 예측 timestep | task YAML의 dt, 현재 10 ms | 동일 |
| 시뮬레이션 timestep | 10 ms | 10 ms |
| 제어 주기당 적분 | 1회 | 1회 |
| 적분기 | 원래 task의 Go1 설정 (Euler) | 동일 |
| 렌더링 호출 | 제어 주기당 1회 | 동일 |
| 초기 베이스·다리 자세 | sit keyframe | 동일 |
| planner 초기 다리 명령 | 기존 Go1 설정 | 동일 |
| 샘플 수·horizon·반복 수·비용함수 | task 설정 | 동일 |
| DM 물리 충돌 | 해당 없음 | capsule/box/cylinder |
| DM 화면 표시 | 해당 없음 | 원본 상세 메시 |

DM proxy의 접촉 기본값도 Go1 기본값을 사용합니다 (friction 0.6, margin 0.001, condim 1). 기존 task에서 의도한 예측/시뮬레이션의 다리 접촉 모델 차이 등은 Go1과 동일하게 보존합니다. 질량·관성·팔 6축 자유도가 추가되므로 계산 시간까지 동일해지는 것은 아닙니다.

`rollout_model.py`의 공통 로더를 예측과 시뮬레이션 양쪽에서 사용합니다. 두 옵션 모두 10 ms 적분·substep 1회·원래 Go1의 적분기와 접촉 기본값을 사용합니다. 물리 모델의 DM 형상은 단순 도형이며, viewer는 별도의 원본 상세 메시 모델을 사용합니다. `visualization.py`가 베이스·다리·팔 관절각을 복사하고 kinematics/COM 변환만 갱신합니다. 상세 모델에 물리 적분·충돌 검출·구속조건 계산을 실행하지 않으며, headless 실행에서는 상세 표시 모델을 로딩하지 않습니다. full은 실제 팔 관절각을 표시하고 simple은 home 팔 자세를 표시합니다.

`full`은 팔 6축의 관절각·관절속도와 목표 관절각을 Go1 상태/명령 뒤에 추가합니다. 팔 가동범위·토크 제한은 upstream URDF 값이며, PD 게인 kp=55, kd=3은 초기 시뮬레이션 가정입니다. 좌우 손가락은 ±0.02 m에 고정합니다.

`simple`은 `full`의 home 관절각에서 브래킷·팔·그리퍼를 `dm_payload` 한 body로 합칩니다. 질량, 무게중심, 회전 관성은 각 링크의 변환과 평행축 정리를 사용해 합산합니다. 팔 6축·actuator를 없애므로 상태·제어 차원이 Go1과 같고 팔을 움직일 수 없습니다. 같은 primitive 충돌 형상을 고정 자세로 옮겨 충돌 외곽을 유지합니다. 화면에는 해당 고정 자세의 상세 메시를 표시합니다. 총질량과 home의 합성 공간 관성은 `full`과 동일하지만, 관절 유연성·팔 제어 토크·팔의 상대 운동은 표현하지 않습니다.

두 옵션 모두 근사 충돌 형상이므로 정밀 파지나 도구 접촉 예측에는 적합하지 않습니다. 팔 궤적 동시 최적화 및 실기 CAN 드라이버는 포함하지 않습니다.

## 장착 및 물성 가정

`full`은 `trunk → dm_mount → dm_base_link → dm_link1 … dm_link6 → dm_end_link → fingers` 구조이며 `simple`은 `trunk → dm_payload`입니다. 베이스 free joint는 Go1 하나입니다. `full`에서는 팔의 관절 반작용과 하중이 몸통과 다리로 전달되고, `simple`에서는 고정 하중과 관성으로 전달됩니다.

팔 베이스는 몸통 기준 `(0, 0, 0.10)` m에 회전 없이 장착합니다. 브래킷은 `0.18 × 0.13 × 0.05` m 박스, 질량 0.2 kg, 중심 높이 0.075 m입니다. 실측 장착 설계가 아닌 초기 가정입니다.

공식 URDF 질량 합은 **4.9291 kg**이며 README의 약 4.5 kg으로 스케일하지 않았습니다. 브래킷 포함 추가 질량은 **5.1291 kg**, 전체 약 **17.873 kg**입니다. 전원·배선·카메라·물체는 포함하지 않습니다.

손가락에 원본 inertial이 없으므로 한쪽 30 g과 대각 관성을 가정하고 end_link의 0.5 kg에서 총 60 g을 차감했습니다. 전체 질량은 보존하지만 실측 질량 분포는 아닙니다. 두 옵션 모두 이 질량을 보존하고 simple에서는 고정 자세의 합성 관성으로 반영합니다.

`dm_tcp`는 upstream end_link 원점의 기준 site로, 실제 도구 TCP 보정값은 아닙니다. CAD 연결 하우징 접촉 제외와 dm_link4 ↔ dm_link6 접촉 제외는 원래 결합 모델을 따릅니다. 전체 관절범위 자기충돌은 별도 검증 대상입니다.

## 원본 상세 모델과 출처

`scene.xml`과 아래 독립 preview는 원본 상세 모델 확인용이며 위 보행 실행의 기본 모델과 다릅니다. 이 독립 preview는 기존 2 ms 설정을 유지합니다.

```bash
python legged_mppi/scripts/preview_go1_dm.py
python legged_mppi/whole_body_mppi/models/go1_dm/build_model.py
```

[상세 모델 이미지](preview.png)

생성기는 기존 Go1 XML과 고정된 로컬 DM URDF로 scene.xml을 작성합니다. 생성된 XML 대신 생성기를 수정하세요. 실행 중 task 결합 XML을 디스크에 쓰지 않습니다.

DM 원본은 Seeed Studio [reBot-DevArm](https://github.com/Seeed-Projects/reBot-DevArm), commit `1e68165d3bd160a36fc25d9ad33946c1c547399b`의 Rebot_Arm_description/DM입니다. 원본은 ../rebot_dm/upstream/에 수정 없이 보관하고 출처는 ../rebot_dm/SOURCE.json에 기록했습니다.

Hardware Design © 2026 Seeed Studio Co., Ltd. 원본 모델·메시는 [CERN-OHL-W-2.0](../rebot_dm/upstream/LICENSE)입니다. 2026-10-21에 추가한 변환·결합·물성/접촉 변경사항은 위와 생성기에 기록했습니다. 기존 Go1 모델은 기존 라이선스를 따릅니다.

## 옵션 변경 후 검증

MuJoCo 3.1.6에서 `walk_straight --mpopi --headless`를 실행했습니다. `full --steps 1000`은 10초 후 기존 0.2 m 목표 허용오차 기준 Task succeeded., 최종 베이스 약 `(0.923, -0.010, 0.252)` m, 경고 0개였습니다. `simple --steps 1500`은 15초 후 Task succeeded., 최종 베이스 약 `(1.007, -0.005, 0.257)` m, 경고 0개였습니다. simple은 10초 실행에서는 아직 도달 허용오차에 들어오지 않았으므로 full과 동일한 궤적·완료 시간을 보장하지 않습니다.

전체 unittest 25개가 통과했습니다. 두 모델의 총질량, home 무게중심, 베이스 6축의 질량행렬 블록(합성 공간 관성), primitive 배치, TCP 방향 보존과 기존 optimizer·지형·단일 적분 경로를 검증합니다. 원본 상세 메시 모델의 preview는 이 옵션들과 별도로 유지합니다.

상세 메시 표시 변경 후 두 옵션의 오프스크린 렌더링을 확인했습니다. 물리 모델의 mesh asset은 5개를 유지하고 표시 모델만 53개를 사용합니다. 상세 표시는 그리기 비용만 추가하며 rollout의 물리 계산량은 늘리지 않습니다.

## EE 위치 목표를 포함한 whole-body task

EE 제어는 별도 task로 선택합니다. 기존 stand/walk task는 다리 제어 + 팔 자세 유지 동작을 보존하고, 아래 두 task는 다리 12축 + 팔 6축을 하나의 MPPI 계열 최적화에서 동시에 탐색합니다. go1_dm + full 모델에서만 지원합니다.

| task | 동작 | config | EE 목표 |
|---|---|---|---|
| `ee_reach` | 정지 상태의 EE 도달 | `control/configs/mppi_ee_reach.yml` | world `(0.5, 0.2, 0.2)` m |
| `ee_walk` | 보행 중 EE waypoint 추종 | `control/configs/mppi_ee_walk.yml` | world `(0.5, 0.2, 0.2)` m |

```bash
python legged_mppi/scripts/simulate_mppi.py --robot go1_dm --task ee_reach --mpopi
python legged_mppi/scripts/simulate_mppi.py --robot go1_dm --task ee_walk --mpopi
```

`control/tasks/definitions.py`의 `TASKS`에서 각 task의 `ee_waypoints`, `ee_frame`, `arm_joint_ref`, `arm_velocity_ref`을 수정합니다. config의 `whole_body` 블록에서는 weight, terminal_weight, arm_noise_sigma를 수정합니다. 각 파일은 기존 task와 동일하게 MPPI/gait 비용 설정도 포함합니다. EE 목표·가중치를 설정하는 CLI 옵션은 제공하지 않습니다.

```yaml
whole_body:
  weight: 2000.0
  terminal_weight: 5.0
  arm_noise_sigma: [0.12, 0.12, 0.12, 0.12, 0.12, 0.12]
```

기준점은 dm_tcp (upstream end_link 원점)이며 실제 그리퍼 도구 TCP 보정값은 아닙니다. world 목표는 공간에 고정되므로 보행 중 몸통이 멀어지면 도달 불가능해질 수 있습니다. base 목표는 몸통과 함께 이동합니다. 이번 구현은 위치 3축 추종이며 EE 방향·접촉력·파지 제어는 포함하지 않습니다.

기존 보행 비용 + EE 위치 제곱 오차의 누적/종단 비용 + 팔 토크 정규화 비용을 사용합니다. 관절각 제한과 모델 actuator 토크 제한을 적용하며 solver 알고리즘 자체는 변경하지 않습니다. EE 비용은 batch forward kinematics로 계산하고 매 rollout 시점의 추가 mj_forward를 피합니다.

EE task는 home standing으로 시작하고 기존 보행 task는 sit 초기화를 유지합니다. viewer의 초록색 구가 EE 목표이며, 최종 출력과 EE 오차 plot으로 추종 상태를 확인할 수 있습니다. Locomotion task succeeded.는 보행 목표 판정이고 EE 최종 오차는 별도로 출력합니다.

초기 실험 (MuJoCo 3.1.6, MPOPI): 정지 월드 목표에서 5초 후 오차 약 0.101→0.034 m, 보행 몸통 기준 목표에서 10초 후 보행 목표 도달 및 EE 오차 약 0.046 m였습니다. 두 경우 경고는 0개였습니다. 임의 목표·외란에 대한 안정성·정확도를 보장하는 결과는 아니며 추가 튜닝 여지가 있습니다.

Task/config 분리 후 전체 테스트 32개가 통과했습니다. `ee_walk --mpopi --headless --steps 1000` 실행에서 보행 목표 도달, EE 최종 오차 약 0.020 m (base frame), MuJoCo 경고 0개를 확인했습니다. 삭제한 EE CLI 옵션의 거부와 별도 config 로딩도 테스트합니다.

EE config의 `Q_diag`는 qpos+qvel 순서의 49개 상태 가중치이며, 팔 관절각 6개(기본 1.0)와 각속도 6개(기본 0.1)를 포함합니다. `R_diag`는 다리 12개 + 팔 6개 토크 가중치입니다. 팔 토크 가중치 기본값은 각 0.001이며, 기존 `whole_body.arm_control_weight`를 대체합니다. 팔 기준 자세는 home 자세, 기준 각속도는 0입니다.

### EE 속도 추종 task

```bash
python legged_mppi/scripts/simulate_mppi.py --robot go1_dm --task ee_reach_cmd_vel --mpopi
python legged_mppi/scripts/simulate_mppi.py --robot go1_dm --task ee_reach_cmd_vel --mpopi --headless --steps 500
```

`ee_reach_cmd_vel`과 `ee_walk_cmd_vel`은 순수 속도 명령 task이다. `definitions.py`의 `command_phases`에 `duration`(초), `body_cmd_vel`(body XY, m/s), `ee_cmd_vel`(world XYZ, m/s), `gait`를 지정한다. 기본값은 1초 정지 → 4초 동작 → 정지이며 마지막 zero 명령은 계속 유지한다. reach 동작 구간은 body `[0, 0]`, EE `[0.05, 0, 0]`; walk 동작 구간은 body `[0.2, 0]`, EE `[0.2, 0, 0]`이다. EE 속도는 몸체 운동을 포함한 world 절대 속도이다.

단계 전환은 sync/non-sync 모두 시뮬레이션 시간을 사용한다. 예측 horizon 안의 명령·gait 전환도 반영한다. Body/EE waypoint, 위치 오차 기반 속도 생성, EE 위치·종단 비용과 도착·holding 판정을 사용하지 않는다. Body XY 위치 가중치는 반드시 0이며 높이·자세·gait·관절 자세·토크·명령 변화 비용은 유지한다. 속도 0은 원래 위치로 복귀하는 위치 유지 명령이 아니다.

각 YAML의 `whole_body.velocity_weight`는 EE 속도 비용이고 `action_delta_diag`는 다리12+팔6 명령 변화 비용이다. Body XY 속도 비용은 Q_diag의 index 25, 26(0-based)에서 관리한다. `whole_body.weight`, `terminal_weight`는 cmd_vel YAML에서 제거했다. `ee_tracking.png`에는 실제 위치와 실제/명령 속도를 표시하고 위치 목표선을 그리지 않는다. CSV/NPZ에는 `command_phase_indices` 또는 `command_phase_index`, 속도 명령과 오차를 기록하고 waypoint·holding·위치 목표/오차를 기록하지 않는다.

### 팔 5·6번 관절 위치 유지

모든 EE task의 `arm_fixed_joints: [5, 6]`은 1부터 시작하는 팔 관절 번호입니다. 고정 위치 명령은 같은 task의 `arm_joint_ref`에서 가져오며 기본값은 둘 다 0 rad입니다. MPPI 후보·최종 명령의 해당 축을 같은 하한/상한으로 제한하므로 다리12+팔1~4만 변하는 명령을 탐색합니다. 기존 Q49/R18 및 모델 차원은 유지합니다. 물리적 관절 잠금이 아니라 PD 위치 유지이므로 실제 관절각은 하중에 따라 소폭 달라질 수 있습니다. `arm_fixed_joints: []`로 바꾸면 팔 6축을 다시 탐색합니다.

위치 task인 `ee_reach`, `ee_walk`은 `ee_waypoints`를 사용하며 기본값은 world 좌표 `[[0.5, 0.2, 0.2]]`입니다. 여러 좌표를 넣으면 순서대로 추종합니다. 공통 `ee_position_tolerance`, `ee_speed_tolerance`, `ee_settle_time`으로 도착을 판정하고 마지막 목표는 계속 유지합니다. EE waypoint와 기존 몸통 보행 단계는 독립적으로 진행합니다. 이 좌표는 사용자 지정 비교 목표이며 전 구간의 도달 가능성이나 지면 간섭 없는 동작을 보장하는 튜닝값은 아닙니다.

### 계산 후 재생 및 결과 저장

```bash
python legged_mppi/scripts/simulate_mppi.py --robot go1_dm --task ee_reach_cmd_vel --mpopi --playback --no-plot
```

`--playback`은 viewer 없이 계산을 완료하고 결과를 저장한 뒤, 기록된 상태를 시뮬레이션 시간 기준으로 한 번 재생합니다. 재생 중 제어기 최적화·물리 step은 호출하지 않으며, 렌더링이 늦으면 프레임을 건너뜁니다. 상세 DM 메시를 사용합니다. `--headless`와 동시 사용은 불가합니다.

모든 실행은 `whole_body_mppi/outputs/<제어기>/<task>/<UTC 실행시각>_<robot>/`에 저장됩니다. 제어기 플래그를 생략하면 폴더명은 `configured`입니다. `--headless` 또는 `--no-plot`에서도 저장은 수행하며 화면 표시만 생략합니다.

- `trajectory.npz`: 시뮬레이션 시간, qpos/qvel, 적용 명령, 목표, tracking error, cost, 계산시간 등 원시 배열. 상태 배열은 `[차원, 기록시점]`입니다.
- `metrics.csv`: 시간별 제어기 계산시간(ms), cost 평가시간(ms), horizon cost, 몸통 위치·속도 오차, EE 위치·속도 오차와 waypoint 상태.
- `summary.json`: 총 실행시간, 제어기 계산시간 평균/p95/최대·합계, deadline 초과 비율, tracking error RMS/최대/최종 등.
- `config.json`: task 정의, YAML 설정 및 실제 적용한 주요 solver 설정.
- `metrics.png`: 계산시간·cost·tracking error·몸통 위치·관절 명령 plot.
- `ee_tracking.png`: EE world xyz 목표/실제값 및 속도 명령/실제값(EE task).

`solve_seconds`는 `agent.update`에 걸린 시간이며 렌더링, plot/파일 저장, 기록을 위한 추가 cost 평가는 제외합니다. `cost_seconds`에 추가 평가 시간을 별도로 기록합니다. `run_wall_seconds`는 시뮬레이션 루프 전체 시간이며 실시간 viewer를 사용하면 렌더링 시간도 포함합니다. cost는 각 주기에 선택한 예측 궤적의 누적 horizon 비용이며 실제 stage cost의 누적값이 아닙니다. 제어 갱신이 없는 기록시점에는 직전 cost를 유지하고 계산시간은 NaN으로 둡니다.

### Rollout/비용 계산 최적화

기존 MuJoCo 3.1.6을 유지하면서 CPU 작업을 작은 균등 묶음으로 나눠 분배하고, 단일 궤적 진단은 worker 제출 없이 직접 실행합니다. 대각 Q/R은 원소별 제곱합으로 계산하며 명시적인 비대각 행렬에는 기존 quadratic 계산을 사용합니다. `ee_reach_cmd_vel`은 EE 속도 비용에 batch 운동학을 한 번 계산합니다. 물리 timestep, 접촉 설정, 목표 비용, 샘플 수와 optimizer는 변경하지 않습니다.

동일 입력 마이크로벤치마크:

```bash
python legged_mppi/scripts/benchmark_rollout.py --samples 128 --repeats 30 --output /tmp/rollout-benchmark
```

기존 실행 폴더와 비교하려면 `--reference /path/to/previous-benchmark`를 추가합니다. `benchmark.json`에 물리 rollout, 비용 평가, 전체 solve 시간 중앙값/p95를 기록하고, `reference.npz`에 동일성 비교용 상태·비용·명령을 저장합니다. 고정 관측에서 반복하는 벤치마크이며 실제 폐루프 추종 성능 실험과 구분해야 합니다.

### 계산 지연과 실제 시간 동기화 실험

```bash
# 기존: 계산 동안 물리 정지, 계산 직후 첫 명령 적용
python legged_mppi/scripts/simulate_mppi.py --robot go1_dm --task ee_reach_cmd_vel --mpopi --headless --steps 2000

# 지연 반영: 이전 계획을 실행하고, 새 계획은 관측 이후 지난 시간에 맞춰 적용
python legged_mppi/scripts/simulate_mppi.py --robot go1_dm --task ee_reach_cmd_vel --mpopi --sync --headless --steps 2000

# 창으로 보려면 --headless 대신 --no-plot 사용
```

`--sync` 하나로 계산 지연 반영과 실제 시간 동기화를 함께 활성화한다. 옵션을 생략하면 계산 중 물리가 정지하는 기존 방식으로 실행한다. `--sync`는 시작 시각부터의 물리 시간에 맞춰 남는 시간만 대기하며, 늦으면 물리 스텝을 생략하지 않고 따라잡는다. 실제 하드웨어 주기를 보장하지 않으며 지연은 `sync_lag_ms`에 기록한다.

지연 실험은 실제 병렬 스레드가 아니라 **측정 지연을 반영하는 직렬 에뮬레이션**이다. 관측으로부터 solve와 진단 cost 계산에 걸린 시간의 합을 물리 스텝 단위로 올림하여 새 계획 도착을 예약한다. 그동안 기존 계획의 시간별 명령을 실행한다. 최초 계획 전에는 현재 관절 위치를 유지하고, 계획 horizon을 넘기면 마지막 명령을 유지한다. 계산 중 새 관측으로 재계산하지 않으며 동시 pending 계획은 하나다. 다음 solve의 warm start와 gait는 지난 예측 스텝만큼 이동하고, EE waypoint 유지 시간과 cmd_vel 참조는 물리 주기마다 갱신한다. 전송 지연, 센서 지연, 병렬 CPU 경합은 포함하지 않는다.

기존 폴더 구조에 그대로 저장하며 `config.json`의 `execution`과 `summary.json`에 모드를 기록한다. `metrics.csv`/`trajectory.npz`에 `plan_age`, `plan_application_delay`, `plan_exhausted`, `sync_lag`를 추가하고 `metrics.png`에 계획 지연을 표시한다. `ee_final_waypoint_last_5s_error_m`은 마지막 waypoint가 활성화된 마지막 5초 오차이며, **도달 또는 정착을 보장하는 holding 전용 통계는 아니다**. 짧은 실행은 접근 구간도 포함하므로 유지 성능 평가는 충분히 긴 동일 설정 실행으로 비교한다.


### 보행 + EE 속도 추종 task

```bash
python legged_mppi/scripts/simulate_mppi.py --robot go1_dm --task ee_walk_cmd_vel --mpopi --sync
```

`ee_reach_cmd_vel`은 `mppi_ee_reach_cmd_vel.yml`, `ee_walk_cmd_vel`은 별도 `mppi_ee_walk_cmd_vel.yml`을 사용한다. 두 task는 위의 시간 기반 직접 속도 명령 구조를 공유한다. 팔 5·6 고정 명령을 유지하며 과거 실험 결과는 변경하지 않는다.
