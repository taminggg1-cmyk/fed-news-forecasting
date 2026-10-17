"""Plan 5 look-ahead comparison (exploratory): a standard language model trained on all labels at once.

One ModernBERT classifier (pre-trained on text up to 2024) is fine-tuned on every Trillion Dollar Words and
World Central Banks label, with no date restriction, by the unmodified Plan 1 training code (stance.
train_and_score with the model name swapped), and scores every document. This pipeline deliberately allows
look-ahead: it measures how much a naive design would inflate the apparent value of text.
"""
import argparse

import numpy as np
import pandas as pd

from fed_news import config, stance
from fed_news.clark_west import compare
from fed_news.corpus import build_corpus
from fed_news.evaluate import metrics
from fed_news.event_study import run as run_event_study, walk_forward
from fed_news.events import build_events, load_events
from fed_news.labels import load_labels
from fed_news.panel import N_COLS, S_COLS
from fed_news.stats_tests import dm_hln
from fed_news.text_features import fed_stance
from fed_news.walkforward import Spec, common_rows, run_spec

NAIVE_MODEL = "answerdotai/ModernBERT-base"
OUT_PATH = config.DATA / "doc_stance_naive.pkl"
OUT = config.RESULTS / "text"


def score_naive(name: str = NAIVE_MODEL) -> pd.DataFrame:
    labels, corpus = load_labels(), build_corpus()
    docs = corpus[corpus["type"].isin(stance.STANCE_TYPES)].copy()
    sent = stance._sentence_table(docs)
    original = stance.model_name
    stance.model_name = lambda vintage: name
    try:
        probs = np.mean([stance.train_and_score(labels, list(sent["sentence"]), 0, seed)
                         for seed in config.STANCE_SEEDS], axis=0)
    finally:
        stance.model_name = original
    sent = sent.assign(stance=probs[:, 1] - probs[:, 0])
    agg = (sent.groupby("doc_id")["stance"].agg(["mean", "size"])
           .rename(columns={"mean": "stance", "size": "n_sent"}).reset_index())
    out = docs[["doc_id", "type", "release"]].merge(agg, on="doc_id", how="left")
    out["n_sent"] = out["n_sent"].fillna(0).astype(int)
    out = out.sort_values(["release", "doc_id"]).reset_index(drop=True)
    out.to_pickle(OUT_PATH)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "naive_model.txt").write_text(name + "\n", encoding="utf-8")
    return out


def naive_panel(name: str, doc_stance: pd.DataFrame) -> pd.DataFrame:
    rows = pd.read_pickle(config.DATA / f"{name}.pkl")
    full = pd.read_pickle(config.DATA / f"{name}_full.pkl").copy()
    st = pd.Series([fed_stance(doc_stance, s, c) for s, c in zip(rows["win_start"], rows["cutoff"])],
                   index=rows.index).ffill()
    full["n_stance"] = st.reindex(full.index)
    full["n_stance_chg"] = st.diff().reindex(full.index)
    return full


def event_compare(chrono_events: pd.DataFrame, naive_events: pd.DataFrame, **window) -> pd.DataFrame:
    """Exploratory, added after results: both walk-forward event studies on their common test days."""
    wc, wn = walk_forward(chrono_events, "dy2", **window), walk_forward(naive_events, "dy2", **window)
    days = wc.index.intersection(wn.index)
    wc, wn = wc.loc[days], wn.loc[days]
    mse = lambda w, col: float(np.mean((w["y"] - w[col]) ** 2))
    row = {"n_common": len(days)}
    for tag, w in (("chrono", wc), ("naive", wn)):
        row[f"mse_small_{tag}"] = mse(w, "pred_small")
        row[f"mse_large_{tag}"] = mse(w, "pred_large")
        row[f"mse_ratio_{tag}"] = row[f"mse_large_{tag}"] / row[f"mse_small_{tag}"]
    row["dm_p_naive_vs_chrono"] = dm_hln((wc["y"] - wc["pred_large"]) ** 2, (wn["y"] - wn["pred_large"]) ** 2)[1]
    return pd.DataFrame([row])


def compare_pipelines() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    naive = pd.read_pickle(OUT_PATH)
    load = lambda task, n: pd.read_csv(config.RESULTS / f"preds_{task}_{n}.csv", parse_dates=["date"])
    rows = []
    for task, name in (("T1", "t1"), ("T2", "t2")):
        p = run_spec(common_rows(naive_panel(name, naive)), Spec("S+N naive", task, "primary", S_COLS + N_COLS),
                     config.TEST_START)
        p.to_csv(OUT / f"preds_{task}_S+N_naive.csv", index=False)
        base, chrono = load(task, "S"), load(task, "S+N")
        dates = set(base["date"]) & set(chrono["date"]) & set(p["date"])
        on = lambda f: f[f["date"].isin(dates)].sort_values("date").reset_index(drop=True)
        rmse_s = metrics(on(base))["rmse"]
        for label, preds in (("chronobert (time-consistent)", chrono), ("naive (ModernBERT, all labels)", p)):
            m = metrics(on(preds))
            rows.append({"task": task, "pipeline": label, "n": m["n"], "rmse": m["rmse"], "rmse_S": rmse_s,
                         "change_vs_S_pct": 100 * (m["rmse"] / rmse_s - 1), **compare(on(base), on(preds)),
                         "dm_p_naive_vs_chrono": np.nan if preds is chrono
                         else compare(on(chrono), on(p))["dm_hln_p"]})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "naive_comparison.csv", index=False)
    y2 = pd.read_pickle(config.DATA / "market_daily.pkl")["y2"]
    naive_events = build_events(naive, y2)
    summary, _ = run_event_study(naive_events, label="naive")
    summary.pop("verdict")  # the H5 rule applies only to the pre-registered ChronoBERT pipeline
    pd.DataFrame([summary]).to_csv(OUT / "naive_event_study.csv", index=False)
    event_compare(load_events(), naive_events).to_csv(OUT / "lookahead_event_compare.csv", index=False)
    print(table.round(4).to_string(index=False))
    return table


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args(argv)
    if args.score:
        score_naive()
    if args.compare:
        compare_pipelines()


if __name__ == "__main__":
    main()
