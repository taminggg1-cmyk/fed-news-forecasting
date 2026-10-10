"""Study 2 regional-speech corpus (spec sections 2-3): FRASER titles of 35 regional presidents plus BIS
speeches, restricted to each president's tenure years, de-duplicated on (surname, day) with FRASER preferred,
and without speeches that duplicate a Plan 1 corpus speech. Output: D:/fed_news/regional_corpus.pkl."""
import glob
import hashlib
import re
from pathlib import Path

import pandas as pd

from fed_news import config, fraser

TENURE = {
    "Minehan": [("Boston", 1994, 2007)], "Rosengren": [("Boston", 2007, 2021)], "Collins": [("Boston", 2022, 2026)],
    "McDonough": [("New York", 1993, 2003)], "Geithner": [("New York", 2003, 2009)],
    "Dudley": [("New York", 2009, 2018)], "Williams": [("San Francisco", 2011, 2018), ("New York", 2018, 2026)],
    "Santomero": [("Philadelphia", 2000, 2006)], "Plosser": [("Philadelphia", 2006, 2015)],
    "Harker": [("Philadelphia", 2015, 2025)], "Paulson": [("Philadelphia", 2025, 2026)],
    "Jordan": [("Cleveland", 1992, 2003)], "Pianalto": [("Cleveland", 2003, 2014)],
    "Mester": [("Cleveland", 2014, 2024)], "Hammack": [("Cleveland", 2024, 2026)],
    "Broaddus": [("Richmond", 1993, 2004)], "Lacker": [("Richmond", 2004, 2017)], "Barkin": [("Richmond", 2018, 2026)],
    "Guynn": [("Atlanta", 1996, 2006)], "Lockhart": [("Atlanta", 2007, 2017)], "Bostic": [("Atlanta", 2017, 2025)],
    "Moskow": [("Chicago", 1994, 2007)], "Evans": [("Chicago", 2007, 2023)], "Goolsbee": [("Chicago", 2023, 2026)],
    "Poole": [("St. Louis", 1998, 2008)], "Bullard": [("St. Louis", 2008, 2023)], "Musalem": [("St. Louis", 2024, 2026)],
    "Stern": [("Minneapolis", 1985, 2009)], "Kocherlakota": [("Minneapolis", 2009, 2015)],
    "Kashkari": [("Minneapolis", 2016, 2026)],
    "Hoenig": [("Kansas City", 1991, 2011)], "George": [("Kansas City", 2011, 2023)], "Schmid": [("Kansas City", 2023, 2026)],
    "McTeer": [("Dallas", 1991, 2004)], "Fisher": [("Dallas", 2005, 2015)], "Kaplan": [("Dallas", 2015, 2021)],
    "Logan": [("Dallas", 2022, 2026)],
    "Parry": [("San Francisco", 1986, 2004)], "Yellen": [("San Francisco", 2004, 2010)], "Daly": [("San Francisco", 2018, 2026)],
}
FRASER_TITLES = {"9017": "Minehan", "9015": "Rosengren", "9016": "Collins", "6748": "McDonough", "6750": "Geithner",
                 "6749": "Dudley", "6109": "Santomero", "6101": "Plosser", "6102": "Harker", "9813": "Paulson",
                 "3769": "Jordan", "3770": "Pianalto", "9033": "Mester", "9815": "Hammack", "9267": "Broaddus",
                 "6827": "Lacker", "9266": "Barkin", "5170": "Guynn", "5579": "Lockhart", "8996": "Bostic",
                 "5967": "Moskow", "8969": "Evans", "8963": "Goolsbee", "485": "Poole", "7161": "Bullard",
                 "7081": "Musalem", "1002": "Stern", "9360": "Kocherlakota", "9278": "George", "6144": "McTeer",
                 "6147": "Fisher", "6146": "Kaplan", "7058": "Logan", "1270": "Parry", "930": "Yellen"}
BANKS = ("Boston", "New York", "Philadelphia", "Cleveland", "Richmond", "Atlanta", "Chicago", "St. Louis",
         "Minneapolis", "Kansas City", "Dallas", "San Francisco")
START, END = "1994-01-01", "2026-09-30"
COLUMNS = ["doc_id", "type", "speaker", "release", "text", "source"]
BIS_DIR = config.DATA / "bis" / "raw"
OUT_PATH = config.DATA / "regional_corpus.pkl"


def surname(name: str) -> str:
    parts = [p for p in re.split(r"[\s,]+", str(name or "").strip()) if p and p.rstrip(".").lower() not in ("jr", "sr", "iii")]
    return parts[-1] if parts else ""


def in_tenure(name: str, year: int, bank: str | None = None) -> bool:
    return any(y0 <= year <= y1 and (bank is None or b == bank) for b, y0, y1 in TENURE.get(name, []))


def _in_window(dates: pd.Series) -> pd.Series:
    return dates.notna() & (dates >= pd.Timestamp(START)) & (dates <= pd.Timestamp(END))


def fraser_selection(items: pd.DataFrame) -> pd.DataFrame:
    sp = items["title_id"].map(FRASER_TITLES)
    keep = sp.notna() & _in_window(items["date"])
    keep &= pd.Series([isinstance(s, str) and pd.notna(d) and in_tenure(s, d.year) for s, d in zip(sp, items["date"])],
                      index=items.index)
    return items[keep].assign(speaker=sp[keep]).reset_index(drop=True)


def fraser_docs(selection: pd.DataFrame, text_dir=None) -> pd.DataFrame:
    rows = []
    for r in selection.itertuples():
        path = Path(text_dir or fraser.TEXT) / f"{r.item_id}.txt"
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if text.strip():
            rows.append({"doc_id": f"fraser:{r.date:%Y%m%d}:{r.item_id}", "type": "speech",
                         "speaker": FRASER_TITLES[r.title_id], "release": r.date, "text": text, "source": "fraser"})
    return pd.DataFrame(rows, columns=COLUMNS)


def _bank(desc: str) -> str | None:
    low = desc.lower()
    for b in BANKS:
        if f"federal reserve bank of {b.lower()}" in low:
            return b
    return None


def bis_docs(bis_dir=None) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(Path(bis_dir or BIS_DIR) / "speeches_*.csv"))):
        df = pd.read_csv(f, usecols=["date", "description", "author", "text"])
        for r in df.itertuples():
            desc, day = str(r.description or ""), pd.to_datetime(r.date, errors="coerce")
            bank, sp = _bank(desc), surname(r.author)
            if bank is None or pd.isna(day) or not re.search(r"(?<!vice )president", desc.lower()):
                continue
            if not (str(day.date()) >= START and str(day.date()) <= END) or not in_tenure(sp, day.year, bank):
                continue
            text = str(r.text or "")
            rows.append({"doc_id": f"bis:{day:%Y%m%d}:{sp}:{hashlib.sha1(text.encode()).hexdigest()[:8]}",
                         "type": "speech", "speaker": sp, "release": day.normalize(), "text": text, "source": "bis"})
    return pd.DataFrame(rows, columns=COLUMNS)


def build_regional(fraser_df: pd.DataFrame, bis_df: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    both = pd.concat([fraser_df, bis_df], ignore_index=True)
    both["day"] = pd.to_datetime(both["release"]).dt.normalize()
    both["rank"] = both["source"].map({"fraser": 0, "bis": 1})
    both = both.sort_values(["rank", "doc_id"]).drop_duplicates(["speaker", "day"], keep="first")
    speeches = corpus[corpus["type"] == "speech"]
    taken = {(surname(s), pd.Timestamp(d).normalize()) for s, d in zip(speeches["speaker"], speeches["release"])}
    keep = [(s, d) not in taken for s, d in zip(both["speaker"], both["day"])]
    out = both[keep].sort_values(["release", "doc_id"]).reset_index(drop=True)
    return out[COLUMNS]


def main() -> pd.DataFrame:
    from fed_news.corpus import build_corpus
    sel = fraser_selection(fraser.load_items())
    print(f"FRASER items selected: {len(sel)}", flush=True)
    fraser.download_texts(sel)
    out = build_regional(fraser_docs(sel), bis_docs(), build_corpus())
    out.to_pickle(OUT_PATH)
    print(out.groupby("source").size().to_string(), flush=True)
    print(out.groupby("speaker").size().sort_values(ascending=False).to_string(), flush=True)
    return out


if __name__ == "__main__":
    main()
