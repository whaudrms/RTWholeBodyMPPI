# Go1-DM Leg–Arm sampling correlation analysis

기존 controller, sampling, optimization 파일을 수정하지 않는 별도 분석 실행기다. EE 작업의 18개 제어 명령을 관찰하며, 다른 solver(CMA/MPOPI/MBD)는 지원하지 않는다. 기존 사용자 시뮬레이션 명령에는 영향을 주지 않는다.

```bash
# repository root; numpy/scipy/mujoco/matplotlib 및 프로젝트 의존성이 설치된 Python
python legged_mppi/scripts/analyze_leg_arm.py \
  --task ee_walk --mode configured --seeds 11 22 33 44 55 \
  --steps 600 --segments 3 --output legged_mppi/analysis_results/my_leg_arm_run

# 같은 결과를 다른 예측 시점에서 다시 시각화 (원시 NPZ 보존, 보고서 덮어쓰기)
python legged_mppi/scripts/analyze_leg_arm.py \
  --report-only --horizon-index 10 --segments 3 \
  --output legged_mppi/analysis_results/my_leg_arm_run
```

기본 `--mode configured`는 YAML의 iteration/annealing 설정을 그대로 사용한다. `mppi`는 기존 `apply_mppi_mode`의 일반 MPPI 설정이다. 기본 분석 ESS 기준은 `ESS < max(20, 0.2*N)`이며 경험적 품질 필터이지 통계적 보장이 아니다. 각 iteration의 실제 batch 크기 N을 사용한다.

출력:

- `updates.html`: 브라우저에서 열면 매 업데이트의 weighted/uniform/difference 12×6 히트맵을 슬라이더로 탐색. 마지막 iteration의 지정 horizon index만 표시하며 낮은 ESS를 명시한다.
- `summary.png`, `ess.png`: seed·구간별 평균을 균등 가중한 요약과 ESS 추이.
- `seed_*.npz`: 모든 업데이트/iteration의 모든 horizon에 대한 `weighted`, `uniform`, `covariance` (record,H,12,6), `ess`, `count`, `low_ess`, `update`, `iteration`. covariance는 비용 가중 모집단 교차 공분산이다.
- `ess.csv`: 모든 iteration의 ESS 및 불안정 여부.
- `segments.csv`: seed·시간 구간·관절쌍별 평균과 유효 업데이트 수/비율.
- `repeatability.csv`: 평균 크기, weighted-uniform 차이, seed·구간 평균의 부호 일치율/표준편차, seed 단위 bootstrap 신뢰구간, 탐색적 후보 여부.
- `report.md`, `experiment.json`, `seed_*.json`: 판단, 분석 기준, 실행 설정, 시뮬레이터 warning 및 최종 base 상태.

상관은 샘플 축에서만 계산한다. 동일 rollout의 다리와 팔을 쌍으로 유지하고, clipping 이후 실제 평가된 actions와 solver의 실제 exp_weights를 사용한다. Uniform은 정확히 같은 actions에 균등 가중치를 사용한다. 고정 관절/분산 0은 NaN(회색 N/A)이며 0으로 취급하지 않는다. 평균 명령을 빼고 공분산을 계산하므로 nominal 이동을 상관으로 오인하지 않는다.

관찰기는 해당 인스턴스의 메서드를 실행 중에만 감싼다. 원본 sampler의 반환값을 복사하고 solver가 가중치를 설정한 다음 분석한다. RNG, actions, costs, solver 계산 순서를 바꾸지 않는다. 종료/예외 시 메서드를 복원한다. 관찰·저장 비용은 추가되므로 실시간 latency 벤치마크에는 사용하지 않는다. 실행기는 headless, sync=False로 기존 Simulator를 이용한다.

반복성 평가는 각 seed를 동일 길이의 연속 구간으로 나누고 낮은 ESS를 제외한 최종 iteration만 사용한다. seed 전체를 bootstrap 단위로 사용하며 adjacent updates나 겹치는 horizon을 독립 표본으로 취급하지 않는다. 최소 5 seeds·3 구간, 각 구간 80% 이상 유효하고 20개 이상 업데이트, |평균 weighted r|≥0.15, |weighted-uniform 차이|≥0.10, 구간 평균 부호 일치율≥80%를 탐색적 후보 기준으로 사용한다. 임계값은 성능 개선이나 유의성을 보장하지 않는다. 72개 관절쌍에 대한 다중검정 보정은 하지 않으므로 bootstrap 구간도 탐색적으로 해석한다.

낮은 ESS는 상관 추정의 불안정성을 뜻하며, 그 자체로 제어 성능이 나쁘다는 뜻은 아니다. 상관이 약하거나 부호가 불안정하면 full cross-covariance 도입 근거가 약하다. 후보가 존재해도 비용 기반 선택에서 생기는 의존성일 뿐 인과성/성능 향상을 입증하지 않는다. 보행 위상에 따라 상쇄되는 관계, 다른 EE 목표, 긴 실행 구간은 별도 분석이 필요하다. 이 실행기는 sampling에 공분산을 적용하지 않는다.
