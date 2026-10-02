"""원본 .mat(v7.3/HDF5) → 셀 단위 캐시.

전체 raw 사이클(~8GB)은 읽지 않고, 모델링에 필요한 것만 캐시한다.
  - cells.parquet   : 셀 메타정보 (batch, policy, cycle_life, barcode, channel ...)
  - summary.parquet : 사이클별 요약 (QD, IR, 온도, 충전시간)
  - qdlin.npz       : 셀별 Qdlin (ΔQ(V) 계산용, 지정 사이클 인덱스만)

실행: python -m src.preprocess
"""
import argparse
import json
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
CACHE_SCHEMA = 3
CACHE_FILES = ("cells.parquet", "summary.parquet", "qdlin.npz", "metadata.json")


def available_batches(data_dir=DATA_DIR, names=None):
    """B1·B2는 필수, B3는 파일이 있을 때만 포함한다."""
    data_dir = Path(data_dir)
    names = list(names) if names is not None else ["B1", "B2"] + (
        ["B3"] if (data_dir / BATCH_FILES["B3"]).is_file() else []
    )
    if not {"B1", "B2"}.issubset(names):
        raise ValueError("학습·평가에는 B1과 B2가 모두 필요합니다.")
    missing = [BATCH_FILES[b] for b in names if not (data_dir / BATCH_FILES[b]).is_file()]
    if missing:
        raise FileNotFoundError(f"{data_dir}에 원본 파일이 없습니다: {', '.join(missing)}. data/README.md를 확인하세요.")
    return {b: BATCH_FILES[b] for b in names}


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
        voltage = f[b["Vdlin"][0, 0]][()].flatten()
        qdlin[f"voltage|{batch}"] = voltage
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
            # summary의 실제 cycle 번호로 Q10·Q100의 0-based 위치를 확인한다.
            labels = df["cycle"].to_numpy()
            for index, cycle in ((9, 10), (99, 100)):
                if index < len(labels) and labels[index] != cycle:
                    raise ValueError(f"{cell_id}: cycles[{index}]와 cycle {cycle}의 대응이 다릅니다.")
            for k in QDLIN_IDX:
                if k < n_cyc:
                    qdlin[f"{cell_id}|{k}"] = f[cyc["Qdlin"][k, 0]][()].flatten()

            # 초기 구간의 측정 시간 간격을 점검한다. 모델 입력에는 사용하지 않는다.
            gaps = []
            for k in range(min(100, n_cyc)):
                t = f[cyc["t"][k, 0]][()].flatten()
                gaps.append(float(np.diff(t).max()) / 60 if len(t) > 1 else 0.0)
            largest_gap = int(np.argmax(gaps)) if gaps else 0
            high_current_fraction = np.nan
            if n_cyc > 9:
                t10 = f[cyc["t"][9, 0]][()].flatten()
                i10 = f[cyc["I"][9, 0]][()].flatten()
                qdlin[f"time|{cell_id}|9"] = t10
                qdlin[f"current|{cell_id}|9"] = i10
                dt = np.diff(t10)
                mid = (i10[1:] + i10[:-1]) / 2
                negative = np.flatnonzero(i10 < -0.1)
                stop = int(negative[0]) if len(negative) else len(i10)
                active = (mid[:stop - 1] > 0.1) & (dt[:stop - 1] > 0) & (dt[:stop - 1] < 1)
                minutes = dt[:stop - 1][active].sum()
                if minutes > 0:
                    high_current_fraction = dt[:stop - 1][active & (mid[:stop - 1] > 4.2)].sum() / minutes

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
                "max_time_gap_hours": gaps[largest_gap] if gaps else np.nan,
                "gap_cycle": largest_gap + 1,
                "high_current_fraction10": high_current_fraction,
            })
    return pd.DataFrame(cells), pd.concat(summaries, ignore_index=True), qdlin


def build_cache(batches=None, data_dir=DATA_DIR, cache_dir=CACHE_DIR):
    """필수 배치와 선택한 추가 배치를 읽어 캐시를 저장한다."""
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    batches = batches if batches is not None else available_batches(data_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cell_list, summ_list, qdlin = [], [], {}
    for batch, fname in batches.items():
        c, s, q = load_batch(batch, data_dir / fname)
        cell_list.append(c); summ_list.append(s); qdlin.update(q)
        print(f"{batch}: {len(c)} cells")

    cells = pd.concat(cell_list, ignore_index=True)
    summary = pd.concat(summ_list, ignore_index=True)
    cells.to_parquet(cache_dir / "cells.parquet", index=False)
    summary.to_parquet(cache_dir / "summary.parquet", index=False)
    # 셀마다 배열 길이가 같지만 키로 찾기 쉽도록 "셀ID|인덱스" 키의 npz 로 저장
    np.savez_compressed(cache_dir / "qdlin.npz", **qdlin)
    metadata = {"schema": CACHE_SCHEMA, "batches": list(batches), "qdlin_indices": list(QDLIN_IDX),
                "sources": {b: {"file": name, "bytes": (data_dir / name).stat().st_size}
                            for b, name in batches.items()}}
    (cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return cells, summary, qdlin


def load_cache(cache_dir=CACHE_DIR):
    """완전한 현재 버전 캐시를 로드한다. 일부 파일만 남은 캐시는 재생성한다."""
    cache_dir = Path(cache_dir)
    if not all((cache_dir / name).is_file() for name in CACHE_FILES):
        return build_cache(cache_dir=cache_dir)
    metadata = json.loads((cache_dir / "metadata.json").read_text())
    if metadata["schema"] != CACHE_SCHEMA:
        return build_cache(cache_dir=cache_dir)
    cells = pd.read_parquet(cache_dir / "cells.parquet")
    summary = pd.read_parquet(cache_dir / "summary.parquet")
    with np.load(cache_dir / "qdlin.npz", allow_pickle=False) as data:
        qdlin = dict(data)
    return cells, summary, qdlin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--batches", nargs="+", choices=list(BATCH_FILES))
    args = parser.parse_args()
    build_cache(available_batches(args.data_dir, args.batches), args.data_dir, args.cache_dir)
