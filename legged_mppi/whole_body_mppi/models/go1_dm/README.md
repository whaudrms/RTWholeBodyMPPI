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

## EE 선속도 cmd_vel whole-body task

몸통과 EE 모두 속도 명령을 추종합니다. 다리 12개와 팔 J1–J4의 총 16개 관절 목표 위치를 함께 최적화하며 `--robot go1_dm --rollout-model full`이 필요합니다.

| Task | 몸통 명령 `[vx,vy,yaw_rate]` | EE 선속도 기준 |
|---|---|---|
| `ee_reach_cmd_vel` | `[0,0,0]` | world 절대속도 |
| `ee_walk_cmd_vel` | `[0.2,0,0]` | 몸통 기준 상대속도 |

```bash
python legged_mppi/scripts/simulate_mppi.py --robot go1_dm --task ee_reach_cmd_vel --mpopi
python legged_mppi/scripts/simulate_mppi.py --robot go1_dm --task ee_walk_cmd_vel --mpopi
```

`control/tasks/definitions.py`에서 `ee_cmd_vel`(m/s), `ee_command_duration`(s), `ee_frame`을 설정합니다. 현재 EE 명령은 reach `[0,0,-0.03]`, walk `[0.03,0,0]`으로 2초간 이동 후 `[0,0,0]`입니다. 예측 horizon에도 같은 시간 스케줄을 적용합니다. EE 위치 목표 및 위치 hold 비용은 없습니다. 0 속도 명령은 그 순간의 위치에서 움직이지 않도록 유도하지만, 누적 위치 오차를 복구하는 위치 제어는 아닙니다. 유한 명령 시간 자체가 작업공간 안전을 보장하지는 않습니다.

world 속도는 몸통 이동·회전에 따른 TCP 속도를 포함합니다. base 속도는 몸통에 대한 TCP 위치의 시간 미분으로, 단순히 world 속도를 회전시킨 값이 아닙니다. 몸통에 대한 상대 이동이 없으면 몸통의 이동·회전과 무관하게 0입니다. 각속도·EE 방향·접촉력·파지 제어는 포함하지 않습니다. TCP는 `dm_tcp`입니다.

각 config의 `whole_body.weight`, `terminal_weight`는 EE 선속도 제곱 오차의 누적/종단 비용에 적용합니다. 기존 숫자는 초기값으로 유지했으며 속도 제어용 튜닝은 별도입니다. 팔 자세·속도 정규화, 토크 비용, actuator 한계는 유지합니다. batch 기구학으로 EE 순간 속도를 계산하며 MuJoCo Jacobian과 비교 검증합니다.

몸통은 heading-frame 속도 추종, 높이·기울기·gait 비용을 사용하며 x/y waypoint와 절대 yaw 목표는 없습니다. home 자세에서 시작하고 지정한 실행 시간 동안 지속합니다. 실행 후 `whole_body_mppi/outputs/`에 몸통 tracking PNG와 `_ee_velocity.png`가 저장됩니다. EE plot에는 각 축 명령/측정 속도, 오차, RMSE, 좌표계 및 실행 설정이 표시됩니다. 위치 목표를 뜻하던 초록색 구는 더 이상 표시하지 않습니다.

`ee_reach`, `ee_walk` 및 config 이름은 `_cmd_vel` 이름으로 변경했습니다. 기존 위치 추종 결과는 새 속도 task의 성능 검증 결과로 사용하지 않습니다. Shield/CLF-CBF-QP를 모든 optimizer와 조합할 수 있습니다. J5/J6는 샘플링부터 home PD 목표각으로 고정합니다(물리 관절은 유지). QP는 몸통·EE CLF와 slack을 각각 분리하고 기존 높이·기울기 CBF를 적용합니다. Go1과 같이 Shield 잔여 위반은 기록 후 계속하고, QP의 solver·입력·CBF 실패는 중단합니다. [설정과 실행 예](../../../docs/clf_cbf_qp.md)를 참고하세요.
