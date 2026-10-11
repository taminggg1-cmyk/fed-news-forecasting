"""Paper-style figures from the saved forecasts (descriptive; no model is re-run).

Figures 1-4 follow the layout of Chan-Lau, Quach and Shi (2025, AMRO WP/25-10): Figure 1 the recursive
out-of-sample LSTM and GRU forecasts on the paper's window (Jan 2016 - Apr 2025), Figures 2-3 the frozen
("counterfactual") forecasts from models trained only on data before 2020 / 2016, Figure 4 the spread of
repeated runs (their 100 LLM replications; our 8 seeds). Figures 5-7 add the comparisons of this study:
with and without Fed text, RMSE of every model, and the per-meeting task. All inputs are CSVs in results/;
the outputs are PNGs in figures/.

Our "Actual" is FRED FEDFUNDS (the monthly average effective rate); the paper plots the target range
midpoint, so its flat stretches sit slightly higher (e.g. 0.13 vs 0.06-0.09 in 2020-21).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fed_news import config

R = config.RESULTS
OUT = config.REPO / "figures"
PAPER = tuple(pd.Timestamp(d) for d in config.PAPER_T1)
PAPER_PUBLISHED = {"LSTM": 0.2376, "GRU": 0.2705, "LLM (look-ahead)": 0.1762}   # their Tables 2 and 3
NOTE = ("Actual: FRED FEDFUNDS, monthly average. Each forecast for month t uses data available at the end of "
        "month t-1; neural lines are the mean of 8 seeds.")
CLASSES = ("cut", "hold", "hike")


def load(path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


def window(df: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    return df.loc[(df.index >= (start or df.index[0])) & (df.index <= (end or df.index[-1]))]


def rmse(df: pd.DataFrame) -> float:
    return float(np.sqrt(((df["y_pred"] - df["y_true"]) ** 2).mean()))


def t1(name: str) -> pd.DataFrame:
    """Walk-forward T1 forecasts by short name (Plan 1 linear rows, Plan 2 neural ensembles)."""
    plan1 = {"no change": "rw", "Taylor rule": "taylor", "linear-S": "S", "linear-S+N": "S+N",
             "linear-S+M": "S+M", "linear-S+M+N": "S+M+N"}
    if name in plan1:
        return load(R / f"preds_T1_{plan1[name]}.csv")
    if name == "3-month bill":
        return load(R / "text" / "preds_T1_bill3m.csv")
    return load(R / "h2a" / f"preds_T1_{name}.csv")


def paper_axes(ax, ylabel="Fed Funds Rate"):
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(alpha=0.25)


def plot_lines(ax, actual: pd.Series, lines, start=None, end=None):
    a = window(actual.to_frame("y"), start, end)["y"]
    ax.plot(a.index, a.values, "k--", lw=1.6, label="Actual")
    for label, df, color in lines:
        d = window(df, start, end)
        ax.plot(d.index, d["y_pred"], color=color, lw=1.5, label=f"{label} (RMSE {rmse(d):.3f})")


def save(fig, name: str, note: str = NOTE):
    fig.text(0.01, 0.005, note, fontsize=8, color="0.35", ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("wrote", OUT / name)


def fig1_paper_window():
    lstm_s, gru_s, lstm_n, gru_n = (t1(n) for n in ("lstm-S", "gru-S", "lstm-S+N", "gru-S+N"))
    actual = lstm_s["y_true"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 11), sharex=True, sharey=True)
    for ax, (l, g), title in ((axes[0], (lstm_s, gru_s), "(a) Economic statistics only: replication of the paper's Figure 1"),
                              (axes[1], (lstm_n, gru_n), "(b) Statistics + Fed text (ChronoBERT stance and news features)")):
        plot_lines(ax, actual, [("Out-of-sample LSTM predictions", l, "blue"),
                                ("Out-of-sample GRU predictions", g, "red")], *PAPER)
        ax.set_title(title, loc="left")
        paper_axes(ax)
    fig.suptitle("Figure 1. Recursive out-of-sample LSTM and GRU predictions vs actual Fed Funds rate, "
                 "Jan 2016 - Apr 2025", fontsize=13)
    save(fig, "fig1_lstm_gru_paper_window.png",
         NOTE + f" Paper's published RMSE on this window: LSTM {PAPER_PUBLISHED['LSTM']}, GRU "
         f"{PAPER_PUBLISHED['GRU']} (monthly refits, revised data); ours refit every 3 months on real-time data.")


def fig_frozen(start: str, number: int):
    f = lambda n: load(R / "frozen" / f"preds_T1_{n}_{start}.csv")
    actual = f("rw")["y_true"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 11), sharex=True)
    plot_lines(axes[0], actual, [("Out-of-sample LSTM predictions", f("lstm-S"), "blue"),
                                 ("Out-of-sample GRU predictions", f("gru-S"), "red")])
    axes[0].set_title("(a) The paper's LSTM and GRU, statistics only (paper's Figure "
                      f"{2 if start == '2020' else 3})", loc="left")
    plot_lines(axes[1], actual, [("No change", f("rw"), "0.55"), ("Linear, statistics", f("linear-S"), "tab:green"),
                                 ("Linear, statistics + text", f("linear-S+N"), "tab:orange"),
                                 ("TFT, statistics + text", f("tft-S+N"), "tab:purple")])
    axes[1].set_title("(b) Simpler models frozen the same way", loc="left")
    for ax in axes:
        paper_axes(ax)
    fig.suptitle(f"Figure {number}. Counterfactual (frozen) forecasts: models trained only on data before Jan "
                 f"{start}, never updated", fontsize=13)
    save(fig, f"fig{number}_frozen_{start}.png",
         NOTE + f" Frozen: fitted once on data before {start}-01 and applied to every later month without re-training.")


def fig4_seed_spread():
    fig, axes = plt.subplots(2, 1, figsize=(15, 11), sharex=True)
    for ax, (name, label, color) in zip(axes, (("lstm-S", "LSTM, statistics only", "blue"),
                                               ("tft-S+N", "TFT, statistics + text", "purple"))):
        seeds = [window(load(R / "h2a" / f"preds_T1_{name}_seed{s}.csv"), *PAPER) for s in range(8)]
        mean = window(t1(name), *PAPER)
        for i, d in enumerate(seeds):
            ax.plot(d.index, d["y_pred"], color=color, alpha=0.25, lw=0.8,
                    label="Individual seeds (8)" if i == 0 else None)
        ax.plot(mean.index, mean["y_true"], "k--", lw=1.6, label="Actual")
        ax.plot(mean.index, mean["y_pred"], color=color, lw=2, label=f"Average of 8 seeds (RMSE {rmse(mean):.3f})")
        sd = np.std([rmse(d) for d in seeds], ddof=1)
        ax.set_title(f"{label}: seed RMSE {np.mean([rmse(d) for d in seeds]):.3f} +/- {sd:.3f}", loc="left")
        paper_axes(ax)
    fig.suptitle("Figure 4. Average prediction vs actual Fed Funds rate, with the spread of individual runs "
                 "(paper: 100 LLM replications; here: 8 random seeds)", fontsize=13)
    save(fig, "fig4_seed_spread.png")


def fig5_text_effect():
    pairs = (("Linear", "linear-S", "linear-S+N"), ("Linear + market rates", "linear-S+M", "linear-S+M+N"),
             ("Hybrid (linear + TFT)", "hybrid-S", "hybrid-S+N"), ("TFT", "tft-S", "tft-S+N"),
             ("LSTM (paper settings)", "lstm-S", "lstm-S+N"), ("GRU (paper settings)", "gru-S", "gru-S+N"))
    fig, axes = plt.subplots(3, 2, figsize=(16, 14), sharex=True, sharey=True)
    for ax, (title, a, b) in zip(axes.ravel(), pairs):
        da, db = t1(a), t1(b)
        plot_lines(ax, da["y_true"], [("Without text", da, "tab:blue"), ("With Fed text", db, "tab:red")])
        change = 100 * (rmse(db) / rmse(da) - 1)
        ax.set_title(f"{title}: text changes RMSE by {change:+.1f}%", loc="left")
        paper_axes(ax)
        ax.set_ylim(-2.5, 7.5)
    fig.suptitle("Figure 5. Does Fed text help? Every model with and without the text features, "
                 "monthly test Jan 2001 - Sep 2026 (308 months)", fontsize=13)
    save(fig, "fig5_with_vs_without_text.png",
         NOTE + " Pre-registered tests of the text gain: linear (H1) one-sided DM p = 0.077; neural (H2a) none "
         "significant after Holm (smallest raw p = 0.008, hybrid).")


def _bars(ax, rows, title):
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [r[2] for r in rows]
    y = np.arange(len(rows))[::-1]
    bars = ax.barh(y, vals, color=colors, edgecolor="0.2")
    for yi, r, bar in zip(y, rows, bars):
        if len(r) > 3 and r[3]:
            bar.set_hatch("//")
        ax.text(r[1] + 0.005, yi, f"{r[1]:.3f}", va="center", fontsize=9)
    ax.set_yticks(y, labels)
    ax.set_xlabel("RMSE (percentage points, lower is better)")
    ax.set_title(title, loc="left")
    ax.grid(axis="x", alpha=0.25)


def fig6_rmse():
    color = {"S": "tab:blue", "S+N": "tab:red"}
    full, paper = [], []
    for label, name, c in (("No change", "no change", "0.55"), ("Taylor rule", "Taylor rule", "0.55"),
                           ("3-month bill + spread", "3-month bill", "tab:green")):
        d = t1(name)
        full.append((label, rmse(d), c))
        paper.append((label, rmse(window(d, *PAPER)), c))
    for label, base in (("Linear", "linear"), ("Linear + market", "linear"), ("Hybrid", "hybrid"), ("TFT", "tft"),
                        ("LSTM", "lstm"), ("GRU", "gru")):
        for info in ("S", "S+N"):
            name = f"{base}-S+M" + ("+N" if info == "S+N" else "") if label == "Linear + market" else f"{base}-{info}"
            d = t1(name)
            tag = "with text" if info == "S+N" else "no text"
            full.append((f"{label}, {tag}", rmse(d), color[info]))
            paper.append((f"{label}, {tag}", rmse(window(d, *PAPER)), color[info]))
    paper += [(f"Paper's {k} (published)", v, "gold", True) for k, v in PAPER_PUBLISHED.items()]
    fig, axes = plt.subplots(1, 2, figsize=(17, 9))
    _bars(axes[0], full, "(a) Full test, Jan 2001 - Sep 2026 (308 months)")
    _bars(axes[1], paper, "(b) Paper's window, Jan 2016 - Apr 2025 (112 months)")
    fig.suptitle("Figure 6. Forecast error of every model (blue: statistics only; red: with Fed text; "
                 "gold: published numbers of Chan-Lau et al. 2025)", fontsize=13)
    save(fig, "fig6_rmse_all_models.png",
         NOTE + " The paper's LLM had look-ahead (acknowledged by its authors), so it is not a fair comparison.")


def fig7_meetings():
    files = {"Always hold": R / "preds_T2_rw.csv", "Linear, statistics": R / "preds_T2_S.csv",
             "Linear, statistics + text": R / "preds_T2_S+N.csv", "Linear + market": R / "preds_T2_S+M.csv",
             "Linear + market + text": R / "preds_T2_S+M+N.csv", "TFT, statistics": R / "h2a" / "preds_T2_tft-S.csv",
             "TFT, statistics + text": R / "h2a" / "preds_T2_tft-S+N.csv",
             "Meeting LSTM, statistics": R / "h2a" / "preds_T2_mlstm-S.csv"}
    preds = {k: load(v) for k, v in files.items()}
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(2, 1, 1)
    base = preds["Always hold"]
    ax.step(base.index, base["y_true"], "k--", where="post", lw=1.6, label="Actual target (range midpoint)")
    for k, c in (("Linear, statistics", "tab:blue"), ("Linear, statistics + text", "tab:red"),
                 ("Linear + market + text", "tab:green")):
        d = preds[k]
        ax.plot(d.index, d["y_pred"], "o-", ms=2.5, lw=1, color=c, label=f"{k} (RMSE {rmse(d):.3f})")
    ax.set_title("(a) Target rate forecast at each FOMC meeting, 212 meetings Jan 2001 - Sep 2026", loc="left")
    paper_axes(ax, "Target rate (%)")
    ax2 = fig.add_subplot(2, 1, 2)
    names = list(preds)
    colors = ("0.55", "tab:blue", "tab:red", "#9fd89f", "tab:green", "#c9a9e0", "tab:purple", "tab:brown")
    acc = np.array([[(d.loc[d["class_true"] == i, "class_pred"] == i).mean() for i in range(3)] +
                    [(d["class_pred"] == d["class_true"]).mean()] for d in preds.values()])
    n = [(base["class_true"] == i).sum() for i in range(3)] + [len(base)]
    x = np.arange(4)
    w = 0.8 / len(names)
    for j, k in enumerate(names):
        xs = x + (j - len(names) / 2 + 0.5) * w
        ax2.bar(xs, 100 * acc[j], w, label=k, color=colors[j], edgecolor="0.2")
        ax2.text(xs[-1], 100 * acc[j][-1] + 1, f"{100 * acc[j][-1]:.0f}", ha="center", fontsize=8)
    ax2.set_xticks(x, [f"{c} ({m} meetings)" for c, m in zip(CLASSES, n[:3])] + [f"all ({n[3]} meetings)"])
    ax2.set_ylabel("Correctly called (%)")
    ax2.set_ylim(0, 128)
    ax2.set_yticks(range(0, 101, 20))
    ax2.legend(ncol=4, loc="upper center", fontsize=9)
    ax2.set_title("(b) Share of meetings where the decision (cut / hold / hike) was called correctly", loc="left")
    ax2.grid(axis="y", alpha=0.25)
    fig.suptitle("Figure 7. Per-meeting task: FOMC decisions", fontsize=13)
    save(fig, "fig7_fomc_meetings.png",
         "Each forecast uses only data dated before the meeting; models are trained on earlier meetings only. "
         "Decision classes use a +/-0.125 point band.")


def main() -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    fig1_paper_window()
    fig_frozen("2020", 2)
    fig_frozen("2016", 3)
    fig4_seed_spread()
    fig5_text_effect()
    fig6_rmse()
    fig7_meetings()


if __name__ == "__main__":
    main()
