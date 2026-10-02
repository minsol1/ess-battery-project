# Data

원본 데이터는 용량(약 8GB) 때문에 저장소에 포함하지 않는다.

## 다운로드
- 출처: Severson et al. (2019), *Data-driven prediction of battery cycle life before capacity degradation* — https://data.matr.io/1/projects/5c48dd2bc625d700019f3204
- Kaggle 미러: `kagglehub.dataset_download("itshpark/data-driven-prediction-of-battery-cycle")`

아래 파일을 이 폴더(`data/`)에 둔다.

| 배치 | 파일 | 용도 |
|---|---|---|
| B1 | `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | 학습 (+ 홀드아웃) |
| B2 | `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | 테스트 |
| B3 | `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | 추가 테스트 (선택) |

## 캐시 생성
```bash
python -m src.preprocess   # data/cache/ 에 cells.parquet · summary.parquet · qdlin.npz 생성
```
전체 raw 사이클은 읽지 않고 셀 메타정보, 사이클별 summary, `Qdlin`(사이클 인덱스 9·10·99·100)만 저장한다.
