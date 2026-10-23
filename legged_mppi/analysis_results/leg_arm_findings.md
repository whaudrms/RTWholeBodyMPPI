# Go1-DM Leg–Arm 교차 상관 분석 결과

현재 결과에서는 Leg–Arm cross-covariance를 MPPI sampling에 바로 적용할 근거가 부족하다. 기존 sampling/optimization 파일은 수정하지 않았다.

실험은 `ee_walk`, seed 11/22/33/44/55, seed당 600회 업데이트(6초), 각 실행을 3개 연속 구간(2초씩)으로 나누어 수행했다. 일반 MPPI와 현재 YAML 설정을 각각 실행했다. 아래 비교는 마지막 iteration의 첫 예측 시점(h=0)에 한정한다. 전체 horizon/iteration 행렬은 NPZ에 보존했다.

| 설정 | iteration당 샘플 수 | 마지막 iteration ESS 중앙값 | 낮은 ESS 비율 | 판정 |
|---|---:|---:|---:|---|
| 일반 MPPI (`mppi`) | 120 | 30.19 | 25.5% (ESS<24) | 평균 상관 및 Uniform 대비 차이가 작고, 구간별 신뢰 표본 비율이 기준에 미달 |
| 현재 YAML (`configured`, 3 iterations) | 40 | 6.81 | 99.2% (ESS<20) | 안정적 상관 추정에 필요한 표본이 부족하여 반복성 판단 불가 |

일반 MPPI에서 ESS 필터를 통과한 데이터의 seed·구간별 평균을 다시 평균했을 때, 관절쌍별 최대 |Weighted Correlation|은 0.00888, 최대 |Weighted−Uniform|은 0.00560이었다. 현재 YAML에서는 유효 업데이트가 전체의 0.8%뿐이고, seed·구간별 유효 비율은 0~8%다. 이 소수 표본에서 얻은 수치나 부호 일치율을 반복 협응의 근거로 해석하면 안 된다.

두 설정 모두 탐색적 후보 조건을 통과한 관절쌍은 0개다. 이는 상관이 존재하지 않는다는 증명이 아니다. ESS 필터의 임계값은 경험적 기준이며, 현재 작업·6초 구간·h=0 분석은 다른 EE 목표나 보행 위상별 관계를 배제하지 않는다. 낮은 ESS는 상관 추정의 불안정성을 뜻하며 제어 실패를 의미하지 않는다. 10개 실행 모두 MuJoCo 경고 수는 0이었다.

권고: 현재 대각 sampling을 유지하고, 교차 공분산 적응을 검토하려면 먼저 충분한 유효 샘플을 확보한 진단 실험과 보행 위상별 재현성을 확인한다. 이후 후보 관계가 발견되어도 동일 rollout 예산에서 제어 비용·성공률을 비교하는 별도 실험이 필요하다. 이번 작업에서는 sampling에 공분산을 적용하지 않았다.

결과 파일:

- [현재 YAML: 업데이트별 히트맵](leg_arm_ee_walk_configured/updates.html)
- [현재 YAML: 상세 보고서](leg_arm_ee_walk_configured/report.md)
- [일반 MPPI: 업데이트별 히트맵](leg_arm_ee_walk/updates.html)
- [일반 MPPI: 상세 보고서](leg_arm_ee_walk/report.md)
- 각 결과 폴더의 `repeatability.csv`, `segments.csv`, `ess.csv`, `summary.png`, `ess.png` 및 `seed_*.npz`를 함께 확인한다.

검증: 합성 데이터의 알려진 상관/비용 조건부 상관, 고정 관절 NaN 처리, 보고서의 ESS 제외 처리, 실제 controller에서 관찰 전후 제어 명령·계획·RNG 상태의 정확한 동일성, 정상/예외 종료 시 메서드 복원을 확인했다. unittest 4개가 통과했다.
