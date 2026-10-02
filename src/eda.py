"""Day 1의 통계 근거를 현재 캐시에서 재계산한다. 전체 수명 경로는 EDA 전용이다."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.features import build_features
from src.preprocess import CACHE_DIR, load_cache
from src.train import RESULTS_DIR


def fade_rates(summary, cell_id, life):
    g = summary[(summary.cell_id == cell_id) & summary.QD.between(0.8, 1.2)].set_index("cycle")["QD"]
    q = lambda c: g.loc[c - 2:c + 2].median()
    return {"early": (q(10) - q(100)) * 1000 / 90,
            "late": (q(int(life * 0.8)) - g.iloc[-1]) * 1000 / (life * 0.2)}


def knee_one(summary, cell_id, life):
    """7점 중앙값 평활 후 두 직선 탐색. 물리적 knee의 정답으로 해석하지 않는다."""
    g = summary[(summary.cell_id == cell_id) & summary.cycle.between(100, life)]
    g = g[["cycle", "QD"]].replace([np.inf, -np.inf], np.nan).dropna().sort_values("cycle")
    if len(g) < 220:
        return {"knee": np.nan, "gain": np.nan, "slope_ratio": np.nan, "found": False}
    x = g.cycle.to_numpy()
    y = g.QD.rolling(7, center=True, min_periods=1).median().to_numpy()
    base_sse = np.sum((y - np.polyval(np.polyfit(x, y, 1), x)) ** 2)
    best = (np.inf, np.nan, np.nan, np.nan)
    for k in np.unique(np.linspace(70, len(x) - 70, 40, dtype=int)):
        left, right = np.polyfit(x[:k], y[:k], 1), np.polyfit(x[k:], y[k:], 1)
        sse = np.sum((y[:k] - np.polyval(left, x[:k])) ** 2) + np.sum((y[k:] - np.polyval(right, x[k:])) ** 2)
        if sse < best[0]:
            best = (sse, float(x[k]), left[0], right[0])
    gain = 1 - best[0] / base_sse if base_sse > 0 else np.nan
    ratio = abs(best[3] / best[2]) if best[2] != 0 else np.nan
    found = bool(np.isfinite(gain) and np.isfinite(ratio) and gain >= 0.2 and ratio >= 1.5 and best[3] < best[2] < 0)
    return {"knee": best[1], "gain": gain, "slope_ratio": ratio, "found": found}


def summarize(cache_dir=CACHE_DIR, results_dir=RESULTS_DIR):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    feat = build_features(cache_dir)
    _, summary, _ = load_cache(cache_dir)
    use = feat[feat.near_eol]
    fades = use[["cell_id", "batch", "cycle_life"]].join(pd.DataFrame(
        [fade_rates(summary, c, life) for c, life in zip(use.cell_id, use.cycle_life)], index=use.index))
    knees = use[["cell_id", "batch", "cycle_life"]].join(pd.DataFrame(
        [knee_one(summary, c, life) for c, life in zip(use.cell_id, use.cycle_life)], index=use.index))
    knees.to_csv(results_dir / "knee_estimates.csv", index=False)
    fades.to_csv(results_dir / "fade_rates.csv", index=False)
    cols = ["QD10", "QD100_minus_QD10", "IR_mean", "Tavg_mean", "Tmax_mean", "chargetime_mean",
            "log_var_dQ", "mean_dQ", "min_dQ", "high_current_fraction10"]
    correlations = pd.DataFrame({b: [spearmanr(g[c], g.cycle_life, nan_policy="omit")[0] for c in cols]
                                 for b, g in feat.groupby("batch")}, index=cols)
    correlations.to_csv(results_dir / "feature_correlations.csv", index_label="feature")
    policy = feat.groupby(["batch", "group", "policy_key"]).agg(
        n=("cell_id", "size"), life_mean=("cycle_life", "mean"), life_median=("cycle_life", "median"),
        log_var_dQ_mean=("log_var_dQ", "mean"))
    policy.to_csv(results_dir / "policy_summary.csv")
    batches = {}
    for batch, g in feat.groupby("batch"):
        f = fades[fades.batch == batch]
        lo, hi = f.cycle_life.quantile([1 / 3, 2 / 3])
        lower, upper = f[f.cycle_life <= lo], f[f.cycle_life >= hi]
        k = knees[(knees.batch == batch) & knees.found]
        batches[batch] = {
            "n": len(g), "near_eol": int(g.near_eol.sum()), "life_median": float(g.cycle_life.median()),
            "under_500_pct": float((g.cycle_life < 500).mean() * 100),
            "over_1000_pct": float((g.cycle_life > 1000).mean() * 100),
            "early_fade_lower": float(lower.early.median()), "early_fade_upper": float(upper.early.median()),
            "late_fade_ratio": float(lower.late.median() / upper.late.median()),
            "knee_found": len(k), "knee_median": float(k.knee.median()) if len(k) else None,
            "max_time_gap_hours_median": float(g.max_time_gap_hours.median()),
            "high_current_life_rho": float(spearmanr(g.high_current_fraction10, g.cycle_life)[0]),
            "high_current_late_fade_rho": float(spearmanr(
                use[use.batch == batch].high_current_fraction10, f.late)[0]),
        }
    b1 = feat[(feat.batch == "B1") & feat.near_eol]
    centered = b1[["log_var_dQ", "cycle_life"]] - b1.groupby("policy_key")[["log_var_dQ", "cycle_life"]].transform("mean")
    stats = {"batches": batches, "b1_near_eol": {
        "var_skew": float(b1.var_dQ.skew()), "log_var_skew": float(b1.log_var_dQ.skew()),
        "life_skew": float(b1.cycle_life.skew()), "c1_soc_rho": float(spearmanr(b1.C1, b1.SOC1)[0]),
        "mean_c_life_rho": float(spearmanr(b1.mean_C, b1.cycle_life)[0]),
        "centered_dq_life_rho": float(spearmanr(centered.log_var_dQ, centered.cycle_life)[0]),
        "dq_redundancy": b1[["log_var_dQ", "mean_dQ", "min_dQ"]].corr(method="spearman").to_dict(),
    }}
    (results_dir / "eda_metrics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return {"features": feat, "summary": summary, "fades": fades, "knees": knees, "stats": stats}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    summarize(args.cache_dir, args.results_dir)
