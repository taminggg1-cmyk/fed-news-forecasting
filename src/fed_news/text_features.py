"""The six news columns (spec section 6).

n_stance:     mean of group means (statement+press conference, minutes, speeches) over documents
              released in [win_start, cutoff); carried forward when the window is empty.
n_stance_chg: n_stance minus the previous row's n_stance.
n_beige:      Loughran-McDonald polarity of the latest Beige Book released before the cutoff.
n_news, n_gpr, n_epu: 30-day means ending 3 days before the cutoff (log levels for GPR, EPU).
"""
import numpy as np
import pandas as pd
import pysentiment2 as ps

from fed_news import config
from fed_news.corpus import build_corpus
from fed_news.indices import load_indices, window_mean
from fed_news.panel import N_COLS

GROUP = {"statement": "policy", "presconf": "policy", "minutes": "minutes", "speech": "speech"}
_LM = ps.LM()


def fed_stance(stance: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    win = stance[(stance["release"] >= start) & (stance["release"] < end)]
    if win.empty:
        return np.nan
    return float(win.groupby(win["type"].map(GROUP))["stance"].mean().mean())


def lm_tone(text: str) -> float:
    return float(_LM.get_score(_LM.tokenize(text))["Polarity"])


def beige_tones(corpus: pd.DataFrame) -> pd.Series:
    beige = corpus[corpus["type"] == "beige"]
    return pd.Series([lm_tone(t) for t in beige["text"]], index=beige["release"]).sort_index()


def _latest_before(series: pd.Series, cutoff: pd.Timestamp) -> float:
    sub = series[series.index < cutoff]
    return float(sub.iloc[-1]) if len(sub) else np.nan


def build_text_features(rows: pd.DataFrame, stance: pd.DataFrame, beige: pd.Series,
                        idx: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=rows.index)
    out["n_stance"] = pd.Series([fed_stance(stance, s, c) for s, c in zip(rows["win_start"], rows["cutoff"])],
                                index=rows.index).ffill()
    out["n_stance_chg"] = out["n_stance"].diff()
    out["n_beige"] = [_latest_before(beige, c) for c in rows["cutoff"]]
    out["n_news"] = [window_mean(idx["news"], c) for c in rows["cutoff"]]
    out["n_gpr"] = [window_mean(np.log(idx["gpr"]), c) for c in rows["cutoff"]]
    out["n_epu"] = [window_mean(np.log(idx["epu"]), c) for c in rows["cutoff"]]
    return out[N_COLS]


def attach_all() -> None:
    stance = pd.read_pickle(config.DATA / "doc_stance.pkl")
    beige = beige_tones(build_corpus())
    idx = load_indices()
    for name in ("t1", "t2"):
        rows = pd.read_pickle(config.DATA / f"{name}.pkl")
        full = rows.join(build_text_features(rows, stance, beige, idx))
        full.to_pickle(config.DATA / f"{name}_full.pkl")
        print(f"{name}: N non-missing\n{full[N_COLS].notna().sum().to_string()}", flush=True)


if __name__ == "__main__":
    attach_all()
