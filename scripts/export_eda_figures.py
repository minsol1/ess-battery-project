"""캐시에서 문서용 EDA 그림을 생성한다: python scripts/export_eda_figures.py."""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import EOL_QD, build_features, delta_q
from src.plotting import COLORS
from src.preprocess import CACHE_DIR, load_cache


def export_figures(cache_dir=CACHE_DIR, output_dir=ROOT / "docs" / "assets"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feat = build_features(cache_dir)
    _, summary, qdlin = load_cache(cache_dir)
    batches = list(feat.groupby("batch", sort=True))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})

    def panels(sharey=False):
        fig, axes = plt.subplots(1, len(batches), figsize=(4.4 * len(batches), 3.7),
                                 sharey=sharey, squeeze=False)
        return fig, axes[0]

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output_dir / name, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(name)

    fig, axes = panels(sharey=True)
    bins = np.linspace(150, 2300, 23)
    for ax, (batch, group) in zip(axes, batches):
        counts, _ = np.histogram(group.cycle_life, bins=bins)
        if counts.sum() != len(group):
            raise ValueError(f"{batch}: 수명 분포 그림 범위 밖의 셀이 있습니다.")
        ax.hist(group.cycle_life, bins=bins, color=COLORS[batch], edgecolor="white")
        for cutoff, color in [(500, "#b54b48"), (1000, "#37877c")]:
            ax.axvline(cutoff, ls="--", lw=1, color=color)
        ax.set(xlim=(150, 2300), xlabel="Recorded cycle life", ylabel="Cells",
               title=f"{batch} (n={len(group)}) · median {group.cycle_life.median():g}")
        ax.grid(axis="y", alpha=0.2)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    save(fig, "life_distribution.png")

    fig, axes = panels(sharey=True)
    for ax, (batch, group) in zip(axes, batches):
        group = group[group.near_eol]
        for cell_id in group.cell_id:
            trace = summary[(summary.cell_id == cell_id) & summary.QD.between(0.8, 1.2)]
            ax.plot(trace.cycle, trace.QD, color=COLORS[batch], alpha=0.4, lw=0.6)
        ax.axvspan(0, 100, color=COLORS["B1"], alpha=0.1)
        ax.axhline(EOL_QD, color="#b54b48", ls=":", lw=1)
        ax.set(xlim=(0, 2000), xlabel="Cycle", ylabel="Discharge capacity (Ah)",
               title=f"{batch} · near EOL (n={len(group)})")
    save(fig, "capacity_trajectories.png")

    fig, axes = panels(sharey=True)
    for ax, (batch, group) in zip(axes, batches):
        lo, hi = group.cycle_life.quantile([1/3, 2/3])
        for label, subset, color in [("Lower lifetime third", group[group.cycle_life <= lo], "#b54b48"),
                                      ("Upper lifetime third", group[group.cycle_life >= hi], "#37877c")]:
            curves = np.stack([delta_q(qdlin, c) * 1000 for c in subset.cell_id])
            ax.plot(qdlin[f"voltage|{batch}"], curves.mean(axis=0), color=color,
                    label=f"{label} (n={len(subset)})")
        ax.set(xlabel="Discharge voltage (V)", ylabel="Q100(V) - Q10(V) (mAh)", title=batch)
        ax.legend(frameon=False, fontsize=9)
    save(fig, "delta_q_curves.png")

    fig, axes = panels()
    for ax, (batch, group) in zip(axes, batches):
        group = group[group.near_eol]
        x = "max_C" if batch == "B3" else "mean_C"
        rho = spearmanr(group[x], group.cycle_life).statistic
        ax.scatter(group[x], group.cycle_life, color=COLORS[batch], s=28, alpha=0.8)
        ax.set(xlabel=x, ylabel="Recorded cycle life", title=f"{batch} · rho={rho:+.2f}")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.grid(alpha=0.15)
    save(fig, "charging_policy.png")

    fig, axes = panels()
    for ax, (batch, group) in zip(axes, batches):
        cell_id = group.cell_id.iloc[0]
        ax.plot(qdlin[f"time|{cell_id}|9"], qdlin[f"current|{cell_id}|9"], color=COLORS[batch])
        ax.set(xlabel="Time (min)", ylabel="Recorded current", title=f"{cell_id} · cycle 10")
        ax.grid(alpha=0.15)
    save(fig, "current_traces.png")

    cols = ["QD10", "QD100_minus_QD10", "IR_mean", "Tavg_mean", "Tmax_mean",
            "chargetime_mean", "log_var_dQ", "mean_dQ", "min_dQ"]
    correlations = pd.DataFrame({batch: [spearmanr(group[c], group.cycle_life, nan_policy="omit").statistic
                                       for c in cols] for batch, group in batches}, index=cols)
    fig, ax = plt.subplots(figsize=(6.1, 4.9))
    sns.heatmap(correlations, annot=True, fmt="+.2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Spearman rho"})
    ax.set_title("Early features and recorded cycle life")
    save(fig, "feature_correlations.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "assets")
    args = parser.parse_args()
    export_figures(args.cache_dir, args.output_dir)
