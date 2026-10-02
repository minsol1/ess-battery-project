# 데이터 준비

[프로젝트로 돌아가기](../README.md) · [검증 설계](../docs/analysis.md)

원본은 약 8GB이며 저장소에 포함하지 않는다. 기본 실행에는 B1·B2가 필요하고, B3 파일이 있으면 추가 평가에 자동으로 포함한다.

## 원본 파일

[원본 데이터](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204) 또는 [Kaggle 미러](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)에서 아래 파일을 받아 `data/`에 둔다. `2018-04-03_varcharge` 배치는 사용하지 않는다.

| 배치 | 파일명 | 용도 |
|---|---|---|
| B1 | `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | 필수 · 학습 / 홀드아웃 |
| B2 | `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | 필수 · 외부 평가 |
| B3 | `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | 선택 · 추가 외부 평가 |

KaggleHub로 받으려면 프로젝트 가상환경에서 별도 다운로드 도구를 설치한다. 전체 미러에는 사용하지 않는 배치도 포함된다.

```bash
python -m pip install kagglehub
python - <<'PY'
from pathlib import Path
import shutil
import kagglehub
from src.preprocess import BATCH_FILES

download = Path(kagglehub.dataset_download("itshpark/data-driven-prediction-of-battery-cycle"))
for filename in BATCH_FILES.values():
    source = next(download.rglob(filename))
    destination = Path("data") / filename
    if not destination.exists():
        shutil.copy2(source, destination)
PY
```

## 캐시 생성

프로젝트 루트에서 실행한다. 캐시는 셀 메타정보, 사이클별 summary, Qdlin의 지정 사이클, 원본 전압축과 cycle 10 전류 패턴을 담는다. 전체 시계열은 복사하지 않는다.

```bash
python -m src.preprocess
```

- B1·B2가 없으면 필요한 파일명을 알려주고 종료한다.
- B3가 없으면 B1·B2만 처리하며, 평가표에 B3 행을 만들지 않는다.
- `summary.cycle`로 Q10·Q100과 배열 인덱스 9·99의 대응을 검사한다. 전압축은 원본 `Vdlin`의 3.5→2.0V, 1,000점이다.
- 0.885 Ah는 B1의 **종료 근접 판정선**이다. 논문의 EOL인 0.88 Ah와 구별하며, 저장 수명을 근사 레이블로 사용한다.
- 파일을 추가·교체하거나 포함 배치를 바꾸면 전처리를 다시 실행한다. 자동 로드는 현재 캐시의 배치 구성을 유지한다.

원본을 다른 폴더에 두거나 B3를 명시적으로 제외할 수도 있다.

```bash
python -m src.preprocess --data-dir /path/to/raw --batches B1 B2
```

`--cache-dir`와 학습의 `--results-dir` 옵션으로 실험 결과를 분리할 수 있다. 캐시·원본은 `.gitignore` 대상이다.
