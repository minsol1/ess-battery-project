"""학습 결과와 같은 모델로 진단 그림을 생성한다."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"B1": "#2f6b9a", "B2": "#c77d2a", "B3": "#7f5aa6"}


def save_figures(train, pred, model, cols, results_dir):
    """환경에 별도 한글 폰트가 없어도 읽을 수 있는 그림 두 개를 저장한다."""
    results_dir = Path(results_dir)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(6, 6))
        for (split, batch), g in pred.groupby(["split", "batch"]):
            ax.scatter(g.cycle_life, g.pred, label=f"{batch} ({split}, n={len(g)})",
                       color=COLORS[batch], alpha=0.8, s=36,
                       marker="o" if split == "test" else "s")
        limit = max(pred.cycle_life.max(), pred.pred.max()) * 1.08
        ax.plot([0, limit], [0, limit], color="#687078", linestyle="--", lw=1, label="Prediction = actual")
        ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Actual cycle life", ylabel="Predicted cycle life")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        fig.savefig(results_dir / "pred_vs_actual.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(train.log_var_dQ, train.cycle_life, color=COLORS["B1"], label=f"B1 train (n={len(train)})", s=32)
        for group, marker in (("standard", "o"), ("newstructure", "^")):
            g = pred[(pred.batch == "B2") & (pred.group == group)]
            ax.scatter(g.log_var_dQ, g.cycle_life, color=COLORS["B2"], marker=marker,
                       label=f"B2 {group} (n={len(g)})", s=36)
        if "log_var_dQ" in cols:
            x = np.linspace(pred.log_var_dQ.min() - 0.1, pred.log_var_dQ.max() + 0.1, 120)
            frame = pd.DataFrame({c: np.full(len(x), train[c].median()) for c in cols})
            frame["log_var_dQ"] = x
            ax.plot(x, model.predict(frame), "--", color="#303840", lw=1.3,
                    label="Fitted model" if len(cols) == 1 else "Model (other inputs at train medians)")
        ax.set(xlabel="log10 variance of Q100(V) - Q10(V)", ylabel="Cycle life")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(results_dir / "dq_shift_b1_b2.png", dpi=160)
        plt.close(fig)
