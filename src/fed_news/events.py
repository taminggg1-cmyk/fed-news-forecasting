"""Event days for the Plan 5 event study (spec 2026-10-10-text-methodology-design.md, section 2).

Each document is mapped to its outcome trading day: the first date with a 2-year yield on or after its
release day. All documents with the same outcome day form one event (so a weekend release joins the next
trading day's releases and no two events share an outcome). Event stance = mean document stance. Surprise =
event stance minus the document-mean stance of earlier events' documents released within the 30 calendar days
before the outcome day; if there are none, minus the mean stance of the latest earlier event's documents.
dy2 = yield change on the outcome day (basis points); dy2_before = the change on the previous trading day.
"""
import numpy as np
import pandas as pd

from fed_news import config

EVENT_TYPES = ("statement", "minutes", "presconf", "speech")
SURPRISE_DAYS = 30
TEST_START, TEST_END = "2001-01-01", "2026-09-30"


def _docs(doc_stance: pd.DataFrame) -> pd.DataFrame:
    d = doc_stance[doc_stance["type"].isin(EVENT_TYPES)].dropna(subset=["stance"]).copy()
    d["day"] = pd.to_datetime(d["release"]).dt.normalize()
    return d.sort_values(["day", "doc_id"]).reset_index(drop=True)


def day_category(types) -> str:
    types = set(types)
    if types & {"statement", "presconf"}:
        return "statement"
    return "minutes" if "minutes" in types else "speech"


def assign_keys(docs: pd.DataFrame, trading: pd.DatetimeIndex) -> pd.DataFrame:
    """Add `key` (outcome trading day) and `k` (its position); drop documents with no outcome day or no
    previous trading day."""
    pos = trading.searchsorted(docs["day"].to_numpy(), side="left")
    keep = (pos < len(trading)) & (pos >= 1)
    out = docs[keep].copy()
    out["k"] = pos[keep]
    out["key"] = trading[pos[keep]]
    return out


def surprise_for(docs: pd.DataFrame, key: pd.Timestamp) -> float:
    today = docs.loc[docs["key"] == key, "stance"]
    prior = docs[docs["key"] < key]
    if today.empty or prior.empty:
        return np.nan
    window = prior[prior["day"] >= key - pd.Timedelta(days=SURPRISE_DAYS)]
    base = window["stance"].mean() if len(window) else prior.loc[prior["key"] == prior["key"].max(), "stance"].mean()
    return float(today.mean() - base)


def build_events(doc_stance: pd.DataFrame, y2: pd.Series, end: str = TEST_END) -> pd.DataFrame:
    y = y2.dropna().sort_index()
    trading = y.index
    docs = _docs(doc_stance)
    docs = assign_keys(docs[docs["day"] <= pd.Timestamp(end)], trading)
    vals = y.to_numpy(float)
    rows = []
    for key, g in docs.groupby("key"):
        k = int(g["k"].iloc[0])
        rows.append({"day": key, "stance": float(g["stance"].mean()), "surprise": surprise_for(docs, key),
                     "dy2": float(100 * (vals[k] - vals[k - 1])),
                     "dy2_before": float(100 * (vals[k - 1] - vals[k - 2])) if k >= 2 else np.nan,
                     "n_docs": len(g), "category": day_category(g["type"]),
                     "first_release": g["day"].min()})
    out = pd.DataFrame(rows).set_index("day")
    return out.dropna(subset=["surprise", "dy2", "dy2_before"])


def load_events(doc_stance: pd.DataFrame | None = None) -> pd.DataFrame:
    stance = pd.read_pickle(config.DATA / "doc_stance.pkl") if doc_stance is None else doc_stance
    y2 = pd.read_pickle(config.DATA / "market_daily.pkl")["y2"]
    return build_events(stance, y2)
