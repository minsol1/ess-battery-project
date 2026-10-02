# 결과 파일 안내

[프로젝트 개요](../README.md) · [분석 근거](../docs/analysis.md) · [검증 기록](../docs/validation.md)

EDA 그림 6개는 [질문별 상세 페이지](../docs/eda.md)에 캡션·해석과 함께 정리했다. 원본 PNG는 [docs/assets](../docs/assets), 생성 명령은 `python scripts/export_eda_figures.py`다.

| 파일 | 확인할 내용 | 생성 명령 |
|---|---|---|
| [model_performance.csv](model_performance.csv) | 지정 형식의 CV·홀드아웃·B2·선택 B3 성능과 Gap | `python -m src.train` |
| [cv_comparison.csv](cv_comparison.csv) | 모델·입력·원/로그 타깃별 외부 CV와 폴드별 점수 | `python -m src.train` |
| [cv_diagnostics.csv](cv_diagnostics.csv) | EN 계수와 잔차 곡률의 탐색 진단 | `python -m src.train` |
| [split_manifest.csv](split_manifest.csv) | 셀별 학습·검증·제외·테스트와 외부 폴드 | `python -m src.train` |
| [run_metadata.json](run_metadata.json) | 최종 구성·파라미터·환경·소스 해시 | `python -m src.train` |
| [predictions.csv](predictions.csv) | 셀별 실제 수명·예측·APE | `python -m src.train` |
| [test_breakdown.csv](test_breakdown.csv) | 집단·입력 범위·수명 범위·정책별 MAPE·MAE·과대 예측률 | `python -m src.train` |
| [pred_vs_actual.png](pred_vs_actual.png) | 실제 수명 대비 예측과 과대·과소 예측 방향 | `python -m src.train` |
| [dq_shift_b1_b2.png](dq_shift_b1_b2.png) | B1 학습 관계와 B2 실험집단의 차이 | `python -m src.train` |
| [eda_metrics.json](eda_metrics.json) | 분포·왜도·knee·정책 내부 관계의 요약 | `python -m src.eda` |
| [feature_correlations.csv](feature_correlations.csv) | 초기 후보 피처와 총 수명 간 Spearman 상관 | `python -m src.eda` |
| [policy_summary.csv](policy_summary.csv) | 정책·배치·실험집단별 표본과 수명 | `python -m src.eda` |
| [fade_rates.csv](fade_rates.csv) | 초기·후반의 셀별 용량 감소 속도 | `python -m src.eda` |
| [knee_estimates.csv](knee_estimates.csv) | 전체 경로 기반 탐색 knee · 모델 입력 제외 | `python -m src.eda` |

CV의 SD는 4개 폴드 점수의 표준편차(ddof=0)다. `in_range`는 **최종 모델 입력** 각각이 학습 28셀의 최소·최대 안에 있는지 판정한다. `over_%`는 예측 총 수명이 실제보다 긴 셀의 비율이다.

`predictions.csv`의 레이블·마지막 QD·시간 공백 등은 평가·오류 분석용 메타정보이며, 최종 모델의 입력 목록은 `run_metadata.json`의 `features`에 명시돼 있다.
