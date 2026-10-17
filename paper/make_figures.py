"""Figures for the working paper (paper/main.tex). Reads committed result files only; writes paper/figures/*.pdf.

Run from the repository root:  python paper/make_figures.py
"""
import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "paper" / "figures"
GREY, BLUE, LIGHT = "#8a8a8a", "#1f5fbf", "#c8c8c8"

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})


def preds(name):
    return pd.read_csv(RES / "post_audit_t1b" / f"preds_T1_{name}.csv", parse_dates=["date"])


def fig_rates():
    """Actual monthly rate against the no-change, S, S+N and S+M+N forecasts (corrected cutoff)."""
    rw, s, sn, smn = (preds(n) for n in ["rw", "S", "S+N", "S+M+N"])
    fig, (a, b) = plt.subplots(2, 1, figsize=(6.3, 4.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    a.plot(s["date"], s["y_true"], color="black", lw=1.1, label="Actual (FEDFUNDS)")
    a.plot(s["date"], s["y_pred"], color=GREY, lw=0.8, label="Ridge, S")
    a.plot(sn["date"], sn["y_pred"], color=BLUE, lw=0.8, label="Ridge, S+N")
    a.set_ylabel("Percent")
    a.legend(frameon=False, loc="upper right", ncol=3)
    for p, col, lab in [(s, GREY, "S"), (sn, BLUE, "S+N"), (smn, "black", "S+M+N")]:
        b.plot(p["date"], p["y_pred"] - p["y_true"], color=col, lw=0.6, label=lab)
    b.axhline(0, color="black", lw=0.5)
    b.set_ylabel("Error (pp)")
    b.set_ylim(-1.2, 1.2)
    b.legend(frameon=False, loc="lower left", ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rates.pdf")
    plt.close(fig)


def fig_cumgain():
    """Running sum of squared-error gains of S+N over S, by month."""
    s, sn = preds("S"), preds("S+N")
    d = (s["y_pred"] - s["y_true"]) ** 2 - (sn["y_pred"] - sn["y_true"]) ** 2
    cum = d.cumsum()
    j = int(d.idxmax())
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    ax.plot(s["date"], cum, color=GREY, lw=1.1)
    ax.plot([s["date"][j]] * 2, [cum[j - 1], cum[j]], color=BLUE, lw=2.2)
    ax.annotate(f"{s['date'][j]:%B %Y}: +{d[j]:.2f}, {d[j] / d.sum():.0%} of the total",
                (s["date"][j], cum[j]), xytext=(-8, -4), textcoords="offset points", ha="right", va="top",
                color=BLUE, fontsize=8)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel(r"Cumulative gain (pp$^2$)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_cumgain.pdf")
    plt.close(fig)
    return float(d[j] / d.sum())


def fig_forest():
    """Network text gains: mean squared-error difference with the 90% block-bootstrap interval."""
    files = {"T2": RES / "study4" / "neural" / "comparison_neural_T2.csv",
             "T1": RES / "study4" / "neural_t1" / "comparison_neural_T1.csv"}
    subset = {"T2": "scheduled", "T1": "all test months"}
    titles = {"T2": "(a) FOMC meetings, 206 scheduled", "T1": "(b) Monthly rate, 308 months"}
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.4), sharey=True)
    for ax, task in zip(axes, ["T2", "T1"]):
        c = pd.read_csv(files[task])
        c = c[c["subset"] == subset[task]]
        labels = []
        for k, (arch, reader) in enumerate([(a, r) for a in ["lstm", "gru", "tft"] for r in ["H", "C"]]):
            y = -k
            labels.append(f"{arch.upper() if arch != 'tft' else 'TFT'} + {'Haiku' if reader == 'H' else 'ChronoBERT'}")
            for role, dy, filled in [("primary", 0.15, True), ("strict", -0.15, False)]:
                row = c[(c["arch"] == arch) & (c["role"] == role) & c["model"].str.endswith("+" + reader)].iloc[0]
                lo, hi = ast.literal_eval(row["sq_diff_boot_ci90"])
                sig = role == "primary" and row["nw_p_holm"] < 0.05
                col = BLUE if sig else GREY
                ax.plot([lo, hi], [y + dy] * 2, color=col, lw=1.4 if filled else 0.9, alpha=1 if filled else 0.7)
                ax.plot(row["sq_diff_mean"], y + dy, "o", ms=4.5, color=col,
                        mfc=col if filled else "white", mew=1.0)
                if role == "primary":
                    ax.text(1.02, y + dy, f"{row['nw_p_holm']:.3f}" if row["nw_p_holm"] < 0.2
                            else f"{row['nw_p_holm']:.2f}", transform=ax.get_yaxis_transform(), va="center",
                            fontsize=7.5, color=col)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(titles[task], fontsize=9, loc="left", pad=10)
        ax.set_xlabel(r"$\Delta$MSE (pp$^2$), base minus text")
        ax.set_yticks(range(0, -6, -1))
        ax.set_yticklabels(labels)
        ax.text(1.02, 0.62, "Holm p", transform=ax.get_yaxis_transform(), fontsize=7.5, color=GREY)
        ax.set_ylim(-5.6, 0.9)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    axes[0].plot([], [], "o", color=GREY, label="Mixed vintage (registered)")
    axes[0].plot([], [], "o", color=GREY, mfc="white", label="Strict real-time (robustness)")
    fig.legend(frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=3.5)
    fig.savefig(OUT / "fig_forest.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_rates()
    share = fig_cumgain()
    fig_forest()
    print(f"figures written to {OUT}; June 2020 share of the monthly gain {share:.1%}")
