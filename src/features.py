"""셀 단위 피처 테이블.

Day 1 설계(2-3, 2-4):
  - 셀의 초기 반응 : log10 var(ΔQ),  ΔQ(V) = Q100(V) - Q10(V)
  - 원정책         : C1(첫 C-rate), C2(둘째 C-rate), SOC1(전환 SOC, %)
  - 파생정책       : mean_C(0~80% 명목 평균 C-rate), max_C(최대 C-rate)
모델 입력이 아닌 컬럼(ΔQ 평균·최솟값, 초기 summary 통계)은 오류 분석용으로 함께 둔다.
"""
import re

import numpy as np
import pandas as pd

from src.preprocess import CACHE_DIR, load_cache

# Qdlin cycles 배열 인덱스 (0-based) → 10·100번째 사이클. Day 1 수치(B1 ρ=-0.87, 왜도 1.11→0.22)와 일치 확인
IDX_Q10, IDX_Q100 = 9, 99

# 종료 근접 판정선 (EOL 0.88 Ah) — Day 1 3-2
EOL_QD = 0.88
NEAR_EOL_QD = 0.885

# CV에서 비교할 입력 5구성 (Day 1 2-3): ΔQ 단독 / 정책 단독 2가지 / ΔQ + 정책 2가지
FEATURE_SETS = {
    "dQ": ["log_var_dQ"],
    "policy_raw": ["C1", "C2", "SOC1"],
    "policy_derived": ["mean_C", "max_C"],
    "dQ+policy_raw": ["log_var_dQ", "C1", "C2", "SOC1"],
    "dQ+policy_derived": ["log_var_dQ", "mean_C", "max_C"],
}

# 충전 정책 문자열 '첫C(전환SOC%)-둘째C' 파싱용. 예: '5.4C(40%)-3.6C' → 5.4, 40, 3.6
_POLICY_RE = re.compile(r"^([\d.]+)C\((\d+)%\)-([\d.]+)C")


def parse_policy(policy):
    """'5.4C(40%)-3.6C[-newstructure]' → (C1, SOC1, C2)."""
    m = _POLICY_RE.match(policy)
    if m is None:   # VarCharge·SLOWCYCLE 등 형식이 다른 정책 (레이블 보유 셀에는 없음)
        return np.nan, np.nan, np.nan
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def mean_c_rate(c1, soc1, c2):
    """0~80% 구간의 명목 평균 C-rate = 0.8 / [s/C1 + (0.8-s)/C2], s = 전환 SOC/100."""
    # 구간별 충전 시간(∝ 충전량/C-rate)을 더해 0→80% 전체 시간으로 환산한 뒤 평균 속도로 바꿈
    # 전환 SOC가 80% 이상이면(예: 4C(80%)-4C) 0~80% 전 구간이 C1 → mean_C = C1
    s = min(soc1 / 100, 0.8)
    return 0.8 / (s / c1 + (0.8 - s) / c2)


def delta_q(qdlin, cell_id):
    """ΔQ(V) = Q100(V) - Q10(V), 1,000개 전압점 (단위 Ah). 열화가 클수록 음의 값이 커진다."""
    keys = [f"{cell_id}|{i}" for i in (IDX_Q10, IDX_Q100)]
    if any(k not in qdlin for k in keys):
        raise ValueError(f"{cell_id}: cycle 10·100의 Qdlin이 필요합니다.")
    a, b = (np.asarray(qdlin[k], dtype=float) for k in keys)
    if a.shape != (1000,) or b.shape != a.shape or not np.isfinite([a, b]).all():
        raise ValueError(f"{cell_id}: Qdlin은 유한한 1,000개 전압점이어야 합니다.")
    return b - a


def early_summary(summary, start=2, end=100):
    """초기 사이클 summary 통계 (오류 분석용). IR 0값은 측정 누락으로 보고 제외."""
    # 1번 사이클은 초기화 성격이 있어 2~100 사이클만 사용
    s = summary[summary["cycle"].between(start, end)].copy()
    s["IR"] = s["IR"].where(s["IR"] > 0)   # IR = 0 → NaN (B2 일부 셀)
    g = s.groupby("cell_id")
    qd10 = summary[summary["cycle"] == 10].set_index("cell_id")["QD"]
    qd100 = summary[summary["cycle"] == 100].set_index("cell_id")["QD"]
    out = pd.DataFrame({
        "QD10": qd10,                          # 초기 용량
        "QD100_minus_QD10": qd100 - qd10,      # 초기 용량 변화 (B2 기록 공백에 민감 → 입력 보류)
        "Tavg_mean": g["Tavg"].mean(),
        "Tmax_mean": g["Tmax"].mean(),
        "chargetime_mean": g["chargetime"].mean(),
        "IR_mean": g["IR"].mean(),
    })
    return out.reset_index(names="cell_id")


def build_features(cache_dir=CACHE_DIR):
    """레이블 보유 셀(B1·B2·B3)의 피처 테이블."""
    cells, summary, qdlin = load_cache(cache_dir)
    # 레이블(cycle_life) 없는 셀(VarCharge·SLOWCYCLE 등)은 제외 → B1 46 / B2 39 / B3 44셀
    df = cells[cells["cycle_life"].notna()].copy()
    if df.empty or not np.isfinite(df["cycle_life"]).all() or (df["cycle_life"] <= 0).any():
        raise ValueError("양수인 cycle_life 레이블이 필요합니다.")

    # ── 충전 정책 피처 ──────────────────────────────
    pol = df["policy"].apply(parse_policy)
    df["C1"], df["SOC1"], df["C2"] = zip(*pol)
    df["mean_C"] = [mean_c_rate(a, s, b) for a, s, b in zip(df["C1"], df["SOC1"], df["C2"])]
    df["max_C"] = df[["C1", "C2"]].max(axis=1)
    # 그룹 단위 분할에 쓰는 정책 키 (newstructure 표기 제외)
    df["policy_key"] = df["policy"].str.replace("-newstructure", "", regex=False)

    # ── ΔQ(V) 피처 ───────────────────────────────────
    dq = {cid: delta_q(qdlin, cid) for cid in df["cell_id"]}
    df["var_dQ"] = [np.var(dq[c]) for c in df["cell_id"]]  # ddof=0, 1,000 전압점
    if (df["var_dQ"] <= 0).any():
        raise ValueError("ΔQ 분산이 0인 셀은 log10 변환할 수 없습니다.")
    df["log_var_dQ"] = np.log10(df["var_dQ"])              # 모델 입력 (왜도 +1.11 → +0.22)
    df["mean_dQ"] = [np.mean(dq[c]) for c in df["cell_id"]]   # 분산과 중복 → 분석용
    df["min_dQ"] = [np.min(dq[c]) for c in df["cell_id"]]     # 분산과 중복 → 분석용

    # ── 초기 summary 통계 (오류 분석용) ─────────────
    df = df.merge(early_summary(summary), on="cell_id", how="left")

    # 0.885 Ah는 종료 근접 판정선이다. 실제 EOL 0.88 Ah와 구별한다.
    df["near_eol"] = df["last_QD"] <= NEAR_EOL_QD
    df["log_cycle_life"] = np.log10(df["cycle_life"])
    return df.reset_index(drop=True)


if __name__ == "__main__":
    f = build_features()
    print(f.groupby("batch").agg(n=("cell_id", "size"), near_eol=("near_eol", "sum"),
                                 n_policy=("policy_key", "nunique")))
