# Study 2: Regional Fed Speeches Event Study (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the regional-speech corpus (FRASER + BIS), re-score all documents with per-year ChronoBERT classifiers trained on all eligible labels, and run the pre-registered H6 event study plus secondary analyses.

**Architecture:** New modules only. `fraser.py` downloads FRASER texts; `regional_corpus.py` applies the tenure table, parses BIS, de-duplicates and writes `regional_corpus.pkl`; `stance2.py` re-trains and scores (`doc_stance2.pkl`); `study2.py` runs H6 and secondaries through the unchanged Plan 5 event code; `evaluate_study2.py` writes `RESULTS_STUDY2.md`.

**Tech Stack:** Python 3.13 (`/d/gold_lm/venv/Scripts/python.exe`), pandas, numpy, requests, torch CUDA, transformers, pytest.

**Spec:** `docs/superpowers/specs/2026-10-10-study2-regional-speeches-design.md` (binding).

## Global Constraints

- Never edit files of Plans 1, 2, 4/5: `src/fed_news/{config,http,market,alfred,fomc_calendar,meetings,corpus,indices,panel,labels,stance,text_features,models,stats_tests,walkforward,run_models,evaluate,tft,neural,neural_walk,run_neural,evaluate_h2a,frozen,diag_tft,clark_west,events,event_study,market_bench,ablation,naive_stance,evaluate_text}.py` and their tests. Import only.
- Tenure table, FRASER title map, de-duplication rule, classifier settings and the H6 rules exactly as in the spec sections 2-5 and the code below.
- No real-data downloads, GPU runs or event-study runs inside a task. The controller runs them after tag `prereg-study2`.
- The FRASER key is read from `config.SECRETS / "fraser_api_key.txt"`; it never appears in errors, logs, files under the repo, or commits.
- Outputs: data to `D:/fed_news/` (`fraser/text/`, `regional_corpus.pkl`, `stance2/`, `doc_stance2.pkl`); results to `results/study2/`.
- Every commit ends with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Stage only your own files.

## Review Focus

1. "Vice President" or staff speeches must not pass the BIS president filter.
2. Surname collisions (Williams, Fisher, Evans, George, Jordan): BIS rows must match the president's bank; FRASER rows are mapped by title id.
3. A speech outside its speaker's tenure years is excluded (e.g., Yellen 2012).
4. De-duplication keeps FRASER over BIS and drops regional rows that duplicate a corpus speech (same surname and day).
5. `stance2` trains each year's classifier on ALL eligible labels (no 20% hold-out) and resumes from parts.

---

### Task 1: FRASER download and regional corpus

**Files:** Create `src/fed_news/fraser.py`, `src/fed_news/regional_corpus.py`, `tests/test_fraser.py`, `tests/test_regional_corpus.py`.

**Interfaces:**
- Produces: `fraser.load_items(meta_dir=None) -> DataFrame` (`item_id, title_id, date, title, text_urls`); `fraser.Throttle(interval)`; `fraser.fetch_text(url, session, throttle, retries=5) -> str`; `fraser.download_texts(items, session=None, throttle=None, text_dir=None) -> int`; `regional_corpus.TENURE`, `FRASER_TITLES`, `in_tenure(surname, year, bank=None)`, `fraser_selection(items)`, `fraser_docs(items, text_dir=None)`, `bis_docs(bis_dir=None)`, `surname(name)`, `build_regional(fraser, bis, corpus)`, `main()` writing `D:/fed_news/regional_corpus.pkl` (columns `doc_id, type, speaker, release, text, source`).

- [ ] **Step 1: Write `tests/test_fraser.py`**

```python
import json

import pandas as pd

from fed_news import fraser


def item(item_id, date, urls, title="Speech"):
    return {"titleInfo": [{"title": title}], "originInfo": {"sortDate": date},
            "recordInfo": {"recordIdentifier": [str(item_id)]}, "location": {"textUrl": urls}}


def test_load_items(tmp_path):
    (tmp_path / "items_7161.json").write_text(json.dumps([item(1, "2015-01-16", ["https://x/a.txt", "https://x/b.txt"]),
                                                         item(2, "bad", [])]), encoding="utf-8")
    out = fraser.load_items(tmp_path)
    assert list(out["item_id"]) == ["1", "2"] and list(out["title_id"]) == ["7161", "7161"]
    assert out["date"].iloc[0] == pd.Timestamp("2015-01-16") and pd.isna(out["date"].iloc[1])
    assert out["text_urls"].iloc[0] == ["https://x/a.txt", "https://x/b.txt"]


class FakeResp:
    def __init__(self, code, body=b""):
        self.status_code, self.content = code, body


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, headers=None, timeout=None):
        self.calls.append((url, headers))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_download_joins_urls_and_skips_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(fraser, "api_key", lambda: "SECRET")
    items = pd.DataFrame({"item_id": ["1", "2"], "text_urls": [["u1", "u2"], ["u3"]]})
    (tmp_path / "2.txt").write_text("cached", encoding="utf-8")
    s = FakeSession([FakeResp(200, b"part one"), FakeResp(200, b"part two")])
    n = fraser.download_texts(items, session=s, throttle=fraser.Throttle(0), text_dir=tmp_path)
    assert n == 1 and (tmp_path / "1.txt").read_text(encoding="utf-8") == "part one\npart two"
    assert [c[0] for c in s.calls] == ["u1", "u2"] and s.calls[0][1] == {"X-API-Key": "SECRET"}


def test_fetch_text_404_is_empty_and_failure_hides_key(monkeypatch):
    monkeypatch.setattr(fraser, "api_key", lambda: "SECRET")
    assert fraser.fetch_text("u", FakeSession([FakeResp(404)]), fraser.Throttle(0)) == ""
    import requests
    s = FakeSession([requests.ConnectionError("boom SECRET")] * 2)
    monkeypatch.setattr(fraser.time, "sleep", lambda *_: None)
    try:
        fraser.fetch_text("u", s, fraser.Throttle(0), retries=2)
    except RuntimeError as err:
        assert "SECRET" not in str(err)
    else:
        raise AssertionError("must raise")
```

- [ ] **Step 2: Write `tests/test_regional_corpus.py`**

```python
import pandas as pd

from fed_news import regional_corpus as rc


def test_in_tenure_and_surname():
    assert rc.in_tenure("Yellen", 2006) and not rc.in_tenure("Yellen", 2012)
    assert rc.in_tenure("Williams", 2015, "San Francisco") and rc.in_tenure("Williams", 2020, "New York")
    assert not rc.in_tenure("Williams", 2015, "New York")
    assert rc.surname("Governor Lael Brainard") == "Brainard" and rc.surname("Robert D. McTeer Jr.") == "McTeer"
    assert rc.surname("") == ""


def test_fraser_selection_and_docs(tmp_path):
    items = pd.DataFrame({"item_id": ["1", "2", "3", "4"], "title_id": ["930", "930", "7161", "9999"],
                          "date": pd.to_datetime(["2006-05-01", "2012-05-01", "2015-01-16", "2015-01-16"]),
                          "title": ["a", "b", "c", "d"], "text_urls": [["u"]] * 4})
    sel = rc.fraser_selection(items)
    assert list(sel["item_id"]) == ["1", "3"]                       # Yellen 2012 and unknown title dropped
    (tmp_path / "1.txt").write_text("Yellen text", encoding="utf-8")
    (tmp_path / "3.txt").write_text("", encoding="utf-8")            # empty text dropped
    docs = rc.fraser_docs(sel, text_dir=tmp_path)
    assert list(docs["speaker"]) == ["Yellen"] and docs["source"].iloc[0] == "fraser" and docs["type"].iloc[0] == "speech"


def test_bis_docs_president_bank_and_tenure(tmp_path):
    rows = pd.DataFrame({
        "date": ["2019-01-18", "2019-02-22", "2019-03-01", "2012-03-01", "2019-04-01"],
        "description": ["Remarks by Mr John C Williams, President and Chief Executive Officer of the Federal Reserve Bank of New York, at ...",
                        "Remarks by Mr Simon M Potter, Executive Vice President of the Federal Reserve Bank of New York, at ...",
                        "Remarks by Mr Richard Fisher, President of the Federal Reserve Bank of Boston, at ...",
                        "Remarks by Ms Mary C Daly, President of the Federal Reserve Bank of San Francisco, at ...",
                        "Remarks by Ms Lael Brainard, Member of the Board of Governors of the Federal Reserve System, at ..."],
        "author": ["John C Williams", "Simon M Potter", "Richard Fisher", "Mary C Daly", "Lael Brainard"],
        "text": ["t1", "t2", "t3", "t4", "t5"]})
    rows.to_csv(tmp_path / "speeches_2019.csv", index=False)
    out = rc.bis_docs(tmp_path)
    assert list(out["speaker"]) == ["Williams"]                      # VP, wrong bank, out of tenure, Board: all excluded
    assert out["source"].iloc[0] == "bis" and out["release"].iloc[0] == pd.Timestamp("2019-01-18")


def test_build_regional_dedup_and_corpus_overlap():
    def doc(i, sp, day, src):
        return {"doc_id": i, "type": "speech", "speaker": sp, "release": pd.Timestamp(day), "text": i, "source": src}
    fr = pd.DataFrame([doc("f1", "Bullard", "2015-01-16", "fraser"), doc("f2", "Yellen", "2008-03-03", "fraser")])
    bis = pd.DataFrame([doc("b1", "Bullard", "2015-01-16", "bis"), doc("b2", "Daly", "2019-05-05", "bis")])
    corpus = pd.DataFrame({"doc_id": ["c1"], "type": ["speech"], "speaker": ["Vice Chair Janet L. Yellen"],
                           "release": [pd.Timestamp("2008-03-03")], "text": ["x"]})
    out = rc.build_regional(fr, bis, corpus)
    assert list(out["doc_id"]) == ["f1", "b2"]
```

- [ ] **Step 3: Run both test files; expect ImportError.**

- [ ] **Step 4: Write `src/fed_news/fraser.py`**

```python
"""FRASER (St. Louis Fed) speech texts for Study 2 (spec 2026-10-10-study2-regional-speeches-design.md).

Item metadata were saved by the count step under D:/fed_news/fraser/meta/items_<titleId>.json. This module
reads them and downloads every textUrl of an item, at most one request every 2.1 seconds (FRASER allows 30 a
minute), caching texts under D:/fed_news/fraser/text/<item_id>.txt. The API key is read from the secrets folder
and never printed or included in errors.
"""
import glob
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

from fed_news import config

META = config.DATA / "fraser" / "meta"
TEXT = config.DATA / "fraser" / "text"
MIN_INTERVAL = 2.1


def api_key() -> str:
    return (config.SECRETS / "fraser_api_key.txt").read_text().strip()


def load_items(meta_dir=None) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(Path(meta_dir or META) / "items_*.json"))):
        tid = re.search(r"items_(\d+)\.json$", f).group(1)
        for r in json.loads(Path(f).read_text(encoding="utf-8")):
            rid = (r.get("recordInfo") or {}).get("recordIdentifier")
            urls = (r.get("location") or {}).get("textUrl") or []
            urls = urls if isinstance(urls, list) else [urls]
            rows.append({"item_id": str(rid[0] if isinstance(rid, list) else rid), "title_id": tid,
                         "date": (r.get("originInfo") or {}).get("sortDate"),
                         "title": ((r.get("titleInfo") or [{}])[0]).get("title", ""),
                         "text_urls": [u for u in urls if isinstance(u, str) and u.startswith("http")]})
    out = pd.DataFrame(rows, columns=["item_id", "title_id", "date", "title", "text_urls"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


class Throttle:
    def __init__(self, interval: float = MIN_INTERVAL):
        self.interval, self.last = interval, 0.0

    def wait(self) -> None:
        gap = self.interval - (time.time() - self.last)
        if gap > 0:
            time.sleep(gap)
        self.last = time.time()


def fetch_text(url: str, session, throttle: Throttle, retries: int = 5) -> str:
    for attempt in range(retries):
        throttle.wait()
        try:
            r = session.get(url, headers={"X-API-Key": api_key()}, timeout=90)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r.content.decode("utf-8", errors="replace")
        if r.status_code == 404:
            return ""
        time.sleep(30 if r.status_code == 429 else 5 * (attempt + 1))
    raise RuntimeError(f"FRASER text download failed after {retries} attempts")


def download_texts(items: pd.DataFrame, session=None, throttle=None, text_dir=None) -> int:
    out_dir = Path(text_dir or TEXT)
    out_dir.mkdir(parents=True, exist_ok=True)
    session, throttle = session or requests.Session(), throttle or Throttle()
    n = 0
    for row in items.itertuples():
        path = out_dir / f"{row.item_id}.txt"
        if path.exists():
            continue
        text = "\n".join(fetch_text(u, session, throttle) for u in row.text_urls)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
        n += 1
        if n % 100 == 0:
            print(f"{n} texts downloaded", flush=True)
    return n
```

- [ ] **Step 5: Write `src/fed_news/regional_corpus.py`**

```python
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
```

- [ ] **Step 6: Run the two test files (expect 7 passed), the full offline suite, and commit the four files** (`Add Study 2 FRASER download and regional speech corpus`). Do NOT download anything.

---

### Task 2: Re-trained stance classifiers

**Files:** Create `src/fed_news/stance2.py`, `tests/test_stance2.py`.

**Interfaces:**
- Consumes: `stance.train_and_score, eligible, classifier_year, vintage_year, _sentence_table, STANCE_TYPES`; `labels.load_labels`; `corpus.build_corpus`; `regional_corpus.OUT_PATH`.
- Produces: `stance2.all_docs() -> DataFrame` (`doc_id, type, release, text, source`); `stance2.run(docs=None, labels=None, parts_dir=None) -> DataFrame` writing `D:/fed_news/doc_stance2.pkl` (`doc_id, type, release, stance, n_sent, source`).

- [ ] **Step 1: Write `tests/test_stance2.py`**

```python
import numpy as np
import pandas as pd

from fed_news import stance, stance2


def docs():
    t = "Rates may rise again later this year if inflation stays high. Inflation remains well above the objective."
    return pd.DataFrame({"doc_id": ["a", "b", "c"], "type": ["speech", "minutes", "speech"],
                         "release": pd.to_datetime(["2005-03-01", "2005-06-01", "2006-02-01"]),
                         "text": [t, t, t], "source": ["corpus", "corpus", "fraser"]})


def labels():
    return pd.DataFrame({"sentence": [f"s{i}" for i in range(200)], "label": [i % 3 for i in range(200)],
                         "year": [1999 + (i % 10) for i in range(200)]})


def test_run_uses_all_eligible_labels_and_resumes(tmp_path, monkeypatch):
    seen = []

    def fake(train, texts, vintage, seed):
        seen.append((vintage, len(train)))
        out = np.zeros((len(texts), 3), dtype=np.float32)
        out[:, 1], out[:, 0] = 0.6, 0.1
        return out

    monkeypatch.setattr(stance, "train_and_score", fake)
    monkeypatch.setattr(stance2, "OUT_PATH", tmp_path / "ds2.pkl")
    lab = labels()
    out = stance2.run(docs(), lab, parts_dir=tmp_path / "parts")
    n2005 = int((lab["year"] < 2005).sum())
    n2006 = int((lab["year"] < 2006).sum())
    assert (2004, n2005) in seen and (2005, n2006) in seen                 # vintage y-1, ALL labels before y
    assert np.allclose(out["stance"], 0.5) and set(out["source"]) == {"corpus", "fraser"}
    assert list(out.columns) == ["doc_id", "type", "release", "stance", "n_sent", "source"]
    calls = len(seen)
    stance2.run(docs(), lab, parts_dir=tmp_path / "parts")
    assert len(seen) == calls                                              # resumed from parts, no retraining
```

- [ ] **Step 2: Run; expect ImportError.**

- [ ] **Step 3: Write `src/fed_news/stance2.py`**

```python
"""Study 2 stance (spec section 4): per-year ChronoBERT classifiers trained on ALL eligible labels (Plan 1 held
out 20% for accuracy reporting), scoring every Plan 1 corpus document and every regional speech with the Plan 1
training code. Output D:/fed_news/doc_stance2.pkl; resumable per classifier year (parts in D:/fed_news/stance2/).
Held-out accuracy is the Plan 1 80/20 figure (results/stance_accuracy.csv)."""
import hashlib

import numpy as np
import pandas as pd

from fed_news import config, stance
from fed_news.corpus import build_corpus
from fed_news.labels import load_labels
from fed_news.regional_corpus import OUT_PATH as REGIONAL_PATH

PARTS = config.DATA / "stance2"
OUT_PATH = config.DATA / "doc_stance2.pkl"
COLUMNS = ["doc_id", "type", "release", "stance", "n_sent", "source"]


def all_docs() -> pd.DataFrame:
    corpus = build_corpus()
    corpus = corpus[corpus["type"].isin(stance.STANCE_TYPES)][["doc_id", "type", "release", "text"]].assign(source="corpus")
    regional = pd.read_pickle(REGIONAL_PATH)[["doc_id", "type", "release", "text", "source"]]
    return pd.concat([corpus, regional], ignore_index=True)


def _fingerprint(train: pd.DataFrame, doc_ids) -> str:
    h = hashlib.sha1()
    h.update(str(len(train)).encode())
    h.update("|".join(sorted(map(str, doc_ids))).encode())
    return h.hexdigest()


def run(docs: pd.DataFrame | None = None, labels: pd.DataFrame | None = None, parts_dir=None) -> pd.DataFrame:
    docs = all_docs() if docs is None else docs.copy()
    labels = load_labels() if labels is None else labels
    docs["cyear"] = pd.to_datetime(docs["release"]).dt.year.map(stance.classifier_year)
    parts = parts_dir or PARTS
    parts.mkdir(parents=True, exist_ok=True)
    sentences = []
    for cyear in sorted(docs["cyear"].unique()):
        ydocs = docs[docs["cyear"] == cyear]
        train = stance.eligible(labels, cyear)
        fp = _fingerprint(train, ydocs["doc_id"])
        part = parts / f"{cyear}.pkl"
        if part.exists() and pd.read_pickle(part)[1] == fp:
            sent = pd.read_pickle(part)[0]
        else:
            sent = stance._sentence_table(ydocs)
            probs = np.mean([stance.train_and_score(train, list(sent["sentence"]), stance.vintage_year(cyear), seed)
                             for seed in config.STANCE_SEEDS], axis=0)
            sent = sent.assign(stance=probs[:, 1] - probs[:, 0])
            tmp = part.with_suffix(".tmp")
            pd.to_pickle((sent, fp), tmp)
            tmp.replace(part)
            print(f"{cyear}: {len(ydocs)} documents, {len(sent)} sentences, {len(train)} labels", flush=True)
        sentences.append(sent)
    sent = pd.concat(sentences, ignore_index=True)
    agg = (sent.groupby("doc_id")["stance"].agg(["mean", "size"])
           .rename(columns={"mean": "stance", "size": "n_sent"}).reset_index())
    out = docs[["doc_id", "type", "release", "source"]].merge(agg, on="doc_id", how="left")
    out["n_sent"] = out["n_sent"].fillna(0).astype(int)
    out = out.sort_values(["release", "doc_id"]).reset_index(drop=True)[COLUMNS]
    out.to_pickle(OUT_PATH)
    return out


if __name__ == "__main__":
    run()
```

- [ ] **Step 4: Run the test (expect 1 passed), the full suite, commit both files** (`Add Study 2 stance re-training on all eligible labels`). No GPU run.

---

### Task 3: H6 analysis and report

**Files:** Create `src/fed_news/study2.py`, `src/fed_news/evaluate_study2.py`, `tests/test_study2.py`.

**Interfaces:**
- Consumes: `events.build_events`, `event_study.run`, `naive_stance.naive_panel`, `walkforward.Spec/common_rows/run_spec`, `clark_west.compare`, `evaluate.metrics`, `panel.S_COLS/N_COLS`.
- Produces: `study2.event_rows(ds, y2) -> (DataFrame, DataFrame)` (summary rows for H6 and the two secondary event sets; H6 walk-forward forecasts); `study2.forecasting(ds) -> DataFrame`; `study2.main()` writing `results/study2/{h6.csv, h6_forecasts.csv, secondary_events.csv, forecasting.csv, preds_T*_S+N_study2.csv}`; `evaluate_study2.main()` writing `RESULTS_STUDY2.md`.

- [ ] **Step 1: Write `tests/test_study2.py`**

```python
import numpy as np
import pandas as pd

from fed_news import config, evaluate_study2 as ev2, study2


def test_event_rows_split_by_source(monkeypatch):
    seen = []

    def fake_build(ds, y2):
        seen.append(sorted(set(ds["source"])))
        return pd.DataFrame({"x": [1]})

    def fake_run(events, label="", **kw):
        return {"pipeline": label, "cw_p": 0.5, "void_cw_p": 0.5, "verdict": "not supported"}, pd.DataFrame({"y": [0]})

    monkeypatch.setattr(study2, "build_events", fake_build)
    monkeypatch.setattr(study2, "run_event_study", fake_run)
    ds = pd.DataFrame({"source": ["corpus", "fraser", "bis"]})
    rows, wf = study2.event_rows(ds, pd.Series(dtype=float))
    assert seen == [["bis", "corpus", "fraser"], ["bis", "fraser"], ["corpus"]]
    assert list(rows["pipeline"]) == ["H6: all documents", "secondary: regional speeches only",
                                      "secondary: Plan 5 documents, new classifier"]
    assert list(rows["role"]) == ["confirmatory", "secondary", "secondary"]


def test_report(tmp_path, monkeypatch):
    out = tmp_path / "study2"
    out.mkdir()
    pd.DataFrame([{"pipeline": "H6: all documents", "role": "confirmatory", "n_test": 3300, "cw_p": 0.03,
                   "void_cw_p": 0.4, "verdict": "supported"}]).to_csv(out / "h6.csv", index=False)
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    monkeypatch.setattr(config, "REPO", tmp_path)
    monkeypatch.setattr(ev2, "_git", lambda *a: "abc")
    monkeypatch.setattr(ev2, "_sha", lambda p: "deadbeef")
    ev2.main()
    text = (tmp_path / "RESULTS_STUDY2.md").read_text(encoding="utf-8")
    assert "**H6 verdict: supported.**" in text and "not run" in text and "conservative" in text
```

- [ ] **Step 2: Run; expect ImportError.**

- [ ] **Step 3: Write `src/fed_news/study2.py`**

```python
"""Study 2 (H6, PREREG_STUDY2.md): the Plan 5 event study on the expanded document table, plus secondary
analyses (regional speeches only; Plan 5 documents with the new classifier; forecasting with rebuilt
Fed-stance features). Only the H6 row decides; secondary rows are reported, not decisive."""
import pandas as pd

from fed_news import config
from fed_news.clark_west import compare
from fed_news.evaluate import metrics
from fed_news.event_study import run as run_event_study
from fed_news.events import build_events
from fed_news.naive_stance import naive_panel
from fed_news.panel import N_COLS, S_COLS
from fed_news.walkforward import Spec, common_rows, run_spec

OUT = config.RESULTS / "study2"
DS_PATH = config.DATA / "doc_stance2.pkl"


def event_rows(ds: pd.DataFrame, y2: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    sets = [("H6: all documents", "confirmatory", ds),
            ("secondary: regional speeches only", "secondary", ds[ds["source"] != "corpus"]),
            ("secondary: Plan 5 documents, new classifier", "secondary", ds[ds["source"] == "corpus"])]
    rows, h6_wf = [], None
    for label, role, sub in sets:
        summary, wf = run_event_study(build_events(sub, y2), label=label)
        if role != "confirmatory":
            summary.pop("verdict", None)
        else:
            h6_wf = wf
        rows.append({"pipeline": label, "role": role, **{k: v for k, v in summary.items() if k != "pipeline"}})
    return pd.DataFrame(rows), h6_wf


def forecasting(ds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, name in (("T1", "t1"), ("T2", "t2")):
        p = run_spec(common_rows(naive_panel(name, ds)), Spec("S+N study2", task, "primary", S_COLS + N_COLS),
                     config.TEST_START)
        p.to_csv(OUT / f"preds_{task}_S+N_study2.csv", index=False)
        base = pd.read_csv(config.RESULTS / f"preds_{task}_S.csv", parse_dates=["date"])
        dates = set(base["date"]) & set(p["date"])
        on = lambda f: f[f["date"].isin(dates)].sort_values("date").reset_index(drop=True)
        rmse_s, m = metrics(on(base))["rmse"], metrics(on(p))
        rows.append({"task": task, "n": m["n"], "rmse": m["rmse"], "rmse_S": rmse_s,
                     "change_vs_S_pct": 100 * (m["rmse"] / rmse_s - 1), **compare(on(base), on(p))})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = pd.read_pickle(DS_PATH)
    y2 = pd.read_pickle(config.DATA / "market_daily.pkl")["y2"]
    rows, wf = event_rows(ds, y2)
    rows[rows["role"] == "confirmatory"].to_csv(OUT / "h6.csv", index=False)
    rows[rows["role"] != "confirmatory"].to_csv(OUT / "secondary_events.csv", index=False)
    wf.to_csv(OUT / "h6_forecasts.csv")
    forecasting(ds).to_csv(OUT / "forecasting.csv", index=False)
    print(rows.T.to_string())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write `src/fed_news/evaluate_study2.py`**

```python
"""Study 2 report: RESULTS_STUDY2.md (H6 verdict, secondary event sets, forecasting, provenance)."""
import hashlib
import subprocess

import pandas as pd

from fed_news import config

TAG = "prereg-study2"


def _git(*args) -> str:
    out = subprocess.run(["git", "-C", str(config.REPO), *args], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else ""


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else "missing"


def _table(name: str) -> str:
    path = config.RESULTS / "study2" / name
    return pd.read_csv(path).round(4).to_markdown(index=False) if path.exists() else "not run"


def main() -> None:
    h6 = config.RESULTS / "study2" / "h6.csv"
    verdict = pd.read_csv(h6)["verdict"].iloc[0] if h6.exists() else "not run"
    lines = [
        "# Results: Study 2 (regional Fed speeches, H6)", "",
        f"**H6 verdict: {verdict}.**", "",
        "H6 (PREREG_STUDY2.md, tag prereg-study2): on the event set expanded with regional Federal Reserve Bank "
        "presidents' speeches, the stance surprise predicts the same-day 2-year Treasury yield change out of sample "
        "(one-sided Clark-West, p < 0.05); void if the same test on the previous trading day's change gives p < 0.05.", "",
        "## H6 (confirmatory)", "", _table("h6.csv"), "",
        "The daily test is conservative for releases after the 2-year yield's afternoon quote: those are matched to a "
        "yield fixed before they happened, which biases the test toward no effect.", "",
        "## Secondary event sets (reported, not decisive)", "", _table("secondary_events.csv"), "",
        "## Forecasting with rebuilt Fed-stance features (secondary)", "", _table("forecasting.csv"), "",
        "## Provenance", "",
        f"- {TAG} tag commit: {_git('rev-list', '-n', '1', TAG) or 'no tag'}",
        f"- HEAD: {_git('rev-parse', '--short', 'HEAD')}",
        f"- sha256 (first 16 hex) regional_corpus.pkl: {_sha(config.DATA / 'regional_corpus.pkl')}",
        f"- sha256 (first 16 hex) doc_stance2.pkl: {_sha(config.DATA / 'doc_stance2.pkl')}", "",
    ]
    (config.REPO / "RESULTS_STUDY2.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:4]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the test (expect 2 passed), the full suite, commit the three files** (`Add Study 2 H6 analysis and report writer`). No real-data run.

---

### Task 4: Pre-register, collect, score, test (controller)

- [ ] **Step 1: `PREREG_STUDY2.md`** (fill the commit hash):

```markdown
# Pre-registration: Study 2 (H6, regional Fed speeches)

Date: 2026-10-10. Code commit: <git rev-parse --short HEAD before this commit>.
Binding: docs/superpowers/specs/2026-10-10-study2-regional-speeches-design.md; code src/fed_news/{fraser,
regional_corpus,stance2,study2,evaluate_study2}.py at this commit, with the unchanged Plan 5 event code
(events.py, event_study.py at tag prereg-text). No regional speech text had been scored and no new event-day
yield change had been computed before this tag; only FRASER item metadata (dates, titles) and BIS speech
counts had been examined.

## H6
On the event set expanded with regional Federal Reserve Bank presidents' speeches (FRASER titles of 35
presidents plus BIS speeches, tenure years and de-duplication as in the spec), with documents re-scored by
per-year ChronoBERT classifiers trained on all eligible labels, the stance surprise predicts the same-day
2-year Treasury yield change out of sample: one-sided Clark-West p < 0.05 (Plan 5 walk-forward OLS, events
keyed by outcome trading day, 30-day document-mean baseline, test outcome days 2001-01-01 to 2026-09-30).
Void if the same test on the previous trading day's change gives p < 0.05.

## Secondary (not decisive)
Regional speeches only; Plan 5 documents with the new classifier; full-sample OLS with Newey-West errors;
monthly and per-meeting forecasting (Plan 1 primary model, S + rebuilt N against S).
```

- [ ] **Step 2:** commit and `git tag prereg-study2`.
- [ ] **Step 3:** background: `PYTHONPATH=src $PY -u -m fed_news.regional_corpus` (FRASER texts, about 4,000 requests at 2.1 s, about 2-3 hours).
- [ ] **Step 4:** background GPU: `PYTHONPATH=src $PY -u -m fed_news.stance2`.
- [ ] **Step 5:** `-m fed_news.study2`, then `-m fed_news.evaluate_study2`; full suite; commit `PREREG`-referenced results (`RESULTS_STUDY2.md results/study2/`) with a DEVIATIONS note for anything that changed.
