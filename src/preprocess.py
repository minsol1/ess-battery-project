"""원본 .mat(v7.3/HDF5) → 셀 단위 캐시.

전체 raw 사이클(~8GB)은 읽지 않고, 모델링에 필요한 것만 캐시한다.
  - cells.parquet   : 셀 메타정보 (batch, policy, cycle_life, barcode, channel ...)
  - summary.parquet : 사이클별 요약 (QD, IR, 온도, 충전시간)
  - qdlin.npz       : 셀별 Qdlin (ΔQ(V) 계산용, 지정 사이클 인덱스만)

실행: python -m src.preprocess
"""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]   # 프로젝트 루트 (src/ 의 상위)
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

# 배치 이름 → 원본 파일 (varcharge 배치는 과제 범위 밖이라 제외)
BATCH_FILES = {
    "B1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "B2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "B3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}

# Qdlin 을 저장할 cycles 배열 인덱스 (0-based). ΔQ = Q100 - Q10 에 쓰는 인덱스는 features.py 에서 선택
QDLIN_IDX = (9, 10, 99, 100)


def _read_str(f, ref):
    """MATLAB 문자열(정수 코드 배열) → str. 빈 값이면 ''."""
    arr = f[ref][()]
    # 정수형이 아니거나 비어 있으면 문자열이 아님
    if arr.dtype.kind not in "ui" or arr.size == 0 or arr.ndim == 2 and 0 in arr.shape:
        return ""
    codes = arr.flatten()
    if codes.max() > 0x10FFFF:  # MATLAB 빈 문자열 placeholder
        return ""
    return "".join(chr(c) for c in codes)


def load_batch(batch, path):
    """배치 파일 하나 → (셀 메타 DataFrame, summary DataFrame, Qdlin dict)."""
    cells, summaries, qdlin = [], [], {}
    with h5py.File(path, "r") as f:
        # batch 그룹의 각 필드는 (셀 수, 1) 모양의 HDF5 참조 배열 → f[ref] 로 실제 값을 읽는다
        b = f["batch"]
        for i in range(b["summary"].shape[0]):
            cell_id = f"{batch}_c{i:02d}"   # 예: B1_c05

            # 1) 사이클별 요약 지표 (cycle, QDischarge, QCharge, IR, Tavg, Tmax, Tmin, chargetime)
            s = f[b["summary"][i, 0]]
            df = pd.DataFrame({k: s[k][()].flatten() for k in s.keys()})
            df = df.rename(columns={"QDischarge": "QD", "QCharge": "QC"})
            df.insert(0, "cell_id", cell_id)
            summaries.append(df)

            # 2) 사이클 내부 시계열 중 Qdlin(전압축 보간 방전용량)만, 필요한 사이클 인덱스만 읽음
            cyc = f[b["cycles"][i, 0]]
            n_cyc = cyc["Qdlin"].shape[0]
            for k in QDLIN_IDX:
                if k < n_cyc:
                    qdlin[f"{cell_id}|{k}"] = f[cyc["Qdlin"][k, 0]][()].flatten()

            # 3) 셀 메타정보
            policy = _read_str(f, b["policy_readable"][i, 0])
            qd = df["QD"].to_numpy()
            cells.append({
                "cell_id": cell_id,
                "batch": batch,
                "policy": policy,
                # 실험집단: 정책 문자열에 'newstructure' 표기가 있으면 newstructure
                "group": "newstructure" if "newstructure" in policy else "standard",
                "barcode": _read_str(f, b["barcode"][i, 0]),
                "channel_id": _read_str(f, b["channel_id"][i, 0]),
                "cycle_life": f[b["cycle_life"][i, 0]][()].item(),   # 레이블 없는 셀은 NaN
                "n_cycles": n_cyc,
                "n_summary": len(df),
                # 마지막 기록 QD — EOL(0.88 Ah)까지 실제로 도달했는지 판정에 사용
                "last_QD": qd[-1] if len(qd) else np.nan,
            })
    return pd.DataFrame(cells), pd.concat(summaries, ignore_index=True), qdlin


def build_cache(batches=BATCH_FILES):
    """모든 배치를 읽어 data/cache/ 에 저장."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cell_list, summ_list, qdlin = [], [], {}
    for batch, fname in batches.items():
        c, s, q = load_batch(batch, DATA_DIR / fname)
        cell_list.append(c); summ_list.append(s); qdlin.update(q)
        print(f"{batch}: {len(c)} cells")

    cells = pd.concat(cell_list, ignore_index=True)
    summary = pd.concat(summ_list, ignore_index=True)
    cells.to_parquet(CACHE_DIR / "cells.parquet", index=False)
    summary.to_parquet(CACHE_DIR / "summary.parquet", index=False)
    # 셀마다 배열 길이가 같지만 키로 찾기 쉽도록 "셀ID|인덱스" 키의 npz 로 저장
    np.savez_compressed(CACHE_DIR / "qdlin.npz", **qdlin)
    return cells, summary, qdlin


def load_cache():
    """캐시 로드. 없으면 생성."""
    if not (CACHE_DIR / "cells.parquet").exists():
        return build_cache()
    cells = pd.read_parquet(CACHE_DIR / "cells.parquet")
    summary = pd.read_parquet(CACHE_DIR / "summary.parquet")
    qdlin = dict(np.load(CACHE_DIR / "qdlin.npz"))
    return cells, summary, qdlin


if __name__ == "__main__":
    build_cache()
