"""Event days for the Plan 5 event study (spec 2026-10-10-text-methodology-design.md, section 2).

An event day is a calendar day with at least one statement, set of minutes, press-conference transcript or
speech. Day stance = mean document stance that day. Surprise = day stance minus the mean stance of documents
released in the 30 calendar days before the day (the day itself excluded), or minus the latest earlier
document's stance when that window is empty. Yield changes are basis points of the 2-year yield.
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


def surprise_at(docs: pd.DataFrame, day: pd.Timestamp) -> float:
    today = docs.loc[docs["day"] == day, "stance"]
    prior = docs[docs["day"] < day]
    if today.empty or prior.empty:
        return np.nan
    window = prior[prior["day"] >= day - pd.Timedelta(days=SURPRISE_DAYS)]
    base = window["stance"].mean() if len(window) else prior["stance"].iloc[-1]
    return float(today.mean() - base)


def yield_change(y: pd.Series, day: pd.Timestamp) -> float:
    """Basis points: first yield on or after `day` minus the last yield strictly before `day`."""
    y = y.dropna()
    after, before = y[y.index >= day], y[y.index < day]
    if after.empty or before.empty:
        return np.nan
    return float(100 * (after.iloc[0] - before.iloc[-1]))


def yield_change_before(y: pd.Series, day: pd.Timestamp) -> float:
    """Basis points: the change on the last business day before `day` (the void-rule outcome)."""
    before = y.dropna()
    before = before[before.index < day]
    if len(before) < 2:
        return np.nan
    return float(100 * (before.iloc[-1] - before.iloc[-2]))


def build_events(doc_stance: pd.DataFrame, y2: pd.Series, end: str = TEST_END) -> pd.DataFrame:
    docs = _docs(doc_stance)
    docs = docs[docs["day"] <= pd.Timestamp(end)]
    rows = [{"day": day, "stance": float(g["stance"].mean()), "surprise": surprise_at(docs, day),
             "dy2": yield_change(y2, day), "dy2_before": yield_change_before(y2, day),
             "n_docs": len(g), "category": day_category(g["type"])}
            for day, g in docs.groupby("day")]
    out = pd.DataFrame(rows).set_index("day")
    return out.dropna(subset=["surprise", "dy2", "dy2_before"])


def load_events(doc_stance: pd.DataFrame | None = None) -> pd.DataFrame:
    stance = pd.read_pickle(config.DATA / "doc_stance.pkl") if doc_stance is None else doc_stance
    y2 = pd.read_pickle(config.DATA / "market_daily.pkl")["y2"]
    return build_events(stance, y2)
