# ESS 배터리 수명 예측

초기 100사이클의 충방전 기록으로 셀의 총 수명(Cycle Life)을 예측하고, ESS 점검·교체 계획에 활용할 수 있는지 검증한다.

**B1 내부에서는 MAPE 7.79%였지만, B2에서는 27.87%로 증가했다. B2 standard 30셀 모두 수명을 길게 예측해 배치 간 적용에 한계가 드러났다.**

[분석·모델 설계](docs/analysis.md) · [실행 검증 기록](docs/validation.md) · [결과 파일](results/README.md) · [Day 1 계획서](docs/day1-plan.pdf)

## 프로젝트 개요

- **데이터**: MIT–Stanford Battery Dataset · Severson et al., *Nature Energy* (2019)
- **학습**: B1(2017-05-12) 종료 근접 36셀 → 학습 28셀 / 정책 단위 홀드아웃 8셀
- **평가**: B2(2018-02-20) 39셀 · 선택 평가 B3(2018-04-12) 44셀
- **태스크**: 회귀 · 총 사이클 수 예측 · 원 단위 MAPE(%)

## EDA → 모델 전략

| 확인한 내용 | 구현에 반영한 판단 |
|---|---|
| B1에는 500사이클 미만 셀이 없고 B2에는 71.8% | 단수명 구간과 실험집단별 오류를 구분해 평가 |
| 초기 용량 변화가 작고 knee는 100사이클 이후 | 전체 수명 경로·knee를 예측 입력에서 제외 |
| ΔQ 분산과 수명의 순위 상관이 B1/B2/B3에서 −0.87/−0.71/−0.80 | `log10 var(Q100(V) − Q10(V))`를 핵심 후보로 사용 |
| 충전 정책·전류 패턴의 관계가 배치마다 다름 | 정책 단독·ΔQ 단독·결합 입력을 같은 CV로 비교 |
| ΔQ 통계가 중복되고 정책 변수도 강하게 연결됨 | 피처 수를 줄이고 Ridge·Elastic Net의 이득을 검증 |

[EDA 노트북](notebooks/01_EDA.ipynb)과 [피처 설계 노트북](notebooks/02_feature_engineering.ipynb)에 그림·통계량·해석을 담았다.

## 모델 선택

최종 모델은 **Ridge · ΔQ 단일 입력 · 원 수명 타깃**이다. 학습 28셀의 정책 그룹 4-fold CV는 **8.84 ± 2.33%**로, 상수 기준선(16.43%)보다 낮았다. ± 값은 폴드 표준편차이며 신뢰구간이 아니다.

입력 5구성 × 원·로그 수명을 비교했다. 하이퍼파라미터도 내부 정책 그룹 3-fold CV로 선택하며, 대치·표준화는 각 학습 폴드에서만 계산한다. Elastic Net과 얕은 트리는 사전 채택 기준인 **최선 Ridge 대비 MAPE 1%p 이상 감소 + 4개 중 3개 폴드 개선**을 충족하지 못했다.

[전체 후보 비교](results/cv_comparison.csv) · [모델링 노트북](notebooks/03_modeling.ipynb)

## 성능 결과

| 구분 | MAPE (%) | 해석 |
|---|---:|---|
| Train (Batch 1 CV) | 8.84 | 정책 그룹 4-fold CV · 후보 선택 점수 |
| Valid (Batch 1 Hold-out) | 7.79 | 새 정책 8셀 · 단일 분할 |
| Test (Batch 2) | 27.87 | B2 39셀 |
| Gap (Train-Valid) | -1.05 | 이번 분할에서 홀드아웃 오차 증가 없음 |
| Gap (Valid-Test) | +20.09 | B2에서 오차 증가 |
| Gap (Target-Test) | +18.77 | 과제 기준 9.1%와의 차이 |
| Test (Batch 3) | 12.11 | B3 44셀 · 추가 평가 |
| Gap (Batch2-Batch3) | -15.77 | B3 − B2 |
| Gap (Target-Test, Batch 3) | +3.01 | B3와 과제 기준의 차이 |

Gap의 단위는 **%p**, 계산 방향은 뒤 − 앞이다. CV는 모델 선택에도 사용했고 홀드아웃은 8셀뿐이므로, 음의 Gap으로 과적합을 배제할 수는 없다. 9.1%는 과제 지정 기준이며 논문과 학습·분할·피처·제외 기준이 다르다. [논문 비교 조건](docs/analysis.md#논문-성능과의-비교)

![실제 수명과 예측 수명](results/pred_vs_actual.png)

## 오류 분석

- **B2 standard**: MAPE 31.63%, 과대 예측 30/30셀. newstructure 9셀은 15.36%였다.
- **입력 범위 안에서도 오차 발생**: ΔQ 범위 안 21셀의 MAPE 36.34%가 범위 밖 18셀(18.00%)보다 컸다. 범위 경고만으로 오류를 걸러낼 수 없다.
- **B3 장수명 과소 예측**: B3_c38은 실제 1,935사이클을 1,074사이클로 예측했다. 학습 수명 범위는 599–1,074사이클이었다.

같은 정책에서도 배치별 수명이 달랐다. 로트·환경·초기 기록 공백은 원인 후보이며, 현재 데이터로 인과관계를 확정하지 않았다. [오류 근거와 개선 방향](docs/analysis.md#오류-분석)

## ESS 도메인 해석

통제된 실험 조건에서 셀 선별과 점검 우선순위를 정하는 보조 지표로 검토할 수 있다. B2의 과대 예측은 교체 지연으로 이어질 수 있으므로, 현재 성능만으로 실제 ESS 교체 시점을 정하기 어렵다.

실제 적용에는 여러 생산 로트의 독립 검증, 새 로트 보정, 일관된 진단 사이클, 예측 불확실성의 검증이 필요하다. [활용 범위와 한계](docs/analysis.md#ess-활용-범위와-한계)

## 실행

Python **3.11** 기준이다. 먼저 [데이터 준비 안내](data/README.md)에 따라 B1·B2 원본 파일을 준비한다. B3는 파일이 있을 때만 추가한다.

```bash
git clone https://github.com/minsol1/ess-battery-project.git
cd ess-battery-project
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python -m src.preprocess
python -m src.eda
python -m src.train
python -m src.verify
```

원본 → 캐시 → 피처 → 후보 비교 → 평가 → CSV·그림 생성까지 CLI로 재현한다. 노트북 실행과 테스트 명령은 [검증 기록](docs/validation.md)에 정리했다.

## 참고문헌·팀

Severson et al. (2019). *Data-driven prediction of battery cycle life before capacity degradation*. **Nature Energy, 4**, 383–391. [논문](https://doi.org/10.1038/s41560-019-0356-8) · [저자 데이터 처리 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)

울산캠퍼스 2반 · **안동선, 김민솔** 공동 참여. [과제 안내](https://actually-war-1ea.notion.site/DS-Mini-Project-32d7f4c8669380338a27f90c471c1fcb)
