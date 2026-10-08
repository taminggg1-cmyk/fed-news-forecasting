# Fed News Plan 1: Data, Features and the Primary Test (H1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the point-in-time data pipeline (real-time macro, Fed documents, newspaper indices), the leakage-free stance classifier, the six news features, and the walk-forward ladder that decides H1: does statistics + news beat statistics alone on the paper's two tasks.

**Architecture:** A `src/fed_news` package of small modules, one per data source or model piece, each writing a cached pickle under `D:/fed_news`. Panels for T1 (monthly) and T2 (per meeting) are built from those caches; a walk-forward engine refits each model row by row and writes predictions; `evaluate.py` applies the pre-registered tests and writes `RESULTS.md`. Any forecast that uses a news column on a test window is refused until the git tag `prereg` exists.

**Tech Stack:** Python 3.13 in `D:/gold_lm/venv` (torch 2.11 + CUDA, transformers 5.17, scikit-learn 1.8, statsmodels 0.14, scipy 1.17, pandas 3.0, numpy 2.3, bs4, lxml, pypdf; added here: pytest, pyarrow, openpyxl, xlrd, pysentiment2). Git Bash on Windows.

**Spec:** `docs/superpowers/specs/2026-10-08-fed-news-design.md` (read it before starting; this plan argues from it).

**Scope:** Plan 1 of 3. Plan 2 adds the neural rows (paper LSTM/GRU, TFT, hybrid); Plan 3 adds the language-model analyses (ChronoGPT scorer, paper-LLM re-run, Guardian LLM reader, recall audit, instruction arm). `PREREG.md` (Task 12) freezes all three.

## Global Constraints

- Python: `PY=/d/gold_lm/venv/Scripts/python.exe`; run every command from `/c/Users/User/fed_news`. `HF_HOME=D:/hf_cache`. Data root `D:/fed_news`.
- Sample starts January 1994. Primary test windows: T1 January 2001 to the latest complete month; T2 meetings from January 2001 to the latest meeting. Paper windows (January 2016 to April 2025) are reported, not tested.
- Cutoff: every input must be dated strictly before the row's cutoff. T1 cutoff = first day of month t. T2 cutoff = the day before the decision date.
- Vintage rule: a row dated in year y >= 2000 uses `manelalab/chrono-bert-v1-{y-1}1231`; rows in 2025-2026 use 2024; rows 1994-1999 use 1999. The classifier for year y trains only on labels with `year < y`.
- Stance classifier: learning rate 2e-5, 3 epochs, batch 16, maximum length 256, seeds 0, 1, 2 averaged, 20% of eligible labels held out.
- News block N is exactly six columns: `n_stance`, `n_stance_chg`, `n_beige`, `n_news`, `n_gpr`, `n_epu`.
- Primary models: penalised ordered logit for T2 with L2 grid {0.01, 0.03, 0.1, 0.3, 1, 3}; ridge on the monthly change for T1 with 7 log-spaced penalties from 0.1 to 1000; five-fold expanding-window inner cross-validation; standardise on training rows only.
- T1 trains on rows up to t-2 (the paper's protocol); T2 trains on all meetings before the current one.
- Classes: cut / hold / hike from the target change with a 0.02 tolerance; continuous forecasts become classes with a ±0.125 band.
- Columns starting `y_` never enter a feature matrix.
- No forecast that uses any N column may be made on a test window before git tag `prereg` exists.
- Win rule: H1 holds only if T1 DM-HLN, T2 DM-HLN and T2 McNemar are all one-sided p < 0.05; void if the placebo passes any of the three; if H1 fails it is reported as a null with no changes afterwards.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A Fed document whose release date is missing or unparseable (for example minutes with no "Released" text nearby) must be dropped and counted, never dated by its meeting date. Test: Task 5 `test_minutes_without_release_are_dropped`.
2. A decision on a weekend (the unscheduled cut of Sunday 2020-03-15, effective the 16th) must still produce a meeting row classed as a cut. Test: Task 4 `test_sunday_unscheduled_cut`.
3. A classifier year with fewer than 50 eligible labels must fail loudly, not train a near-empty model. Test: Task 8 `test_eligible_raises_when_too_few`.
4. A window with no Fed documents must carry the previous Fed stance forward, never become 0. Test: Task 9 `test_empty_window_carries_forward`.
5. A row with any missing S, M, N or Taylor input must be dropped from every ladder row, so S and S+N are always compared on identical rows. Test: Task 11 `test_common_rows_drops_any_missing`.

## File Structure

```
fed_news/
  pyproject.toml              pytest config (pythonpath, markers)
  .gitignore
  PREREG.md                   Task 12
  RESULTS.md                  Task 13 output
  results/                    predictions, tables, coverage, stance accuracy
  src/fed_news/
    __init__.py
    config.py                 paths and every fixed setting
    http.py                   cached GET with retries
    market.py                 daily FRED series, target splice, as-of lookup
    alfred.py                 ALFRED vintages, real-time as-of series
    fomc_calendar.py          parse the Fed's FOMC calendar pages
    meetings.py               meeting table with decisions
    corpus.py                 Fed documents (statements, minutes, press conferences, Beige Books, speeches)
    indices.py                SF Fed sentiment, GPR, EPU, window means
    panel.py                  T1 and T2 rows: targets, S, M, Taylor inputs
    labels.py                 Trillion Dollar Words + WCB stance labels
    stance.py                 per-year ChronoBERT classifiers, document stance
    text_features.py          the six N columns
    models.py                 ordered logit, ridge, CV, Taylor rule, class mapping
    stats_tests.py            DM-HLN, McNemar, Holm
    walkforward.py            ladder specs, walk-forward engine, prereg guard
    run_models.py             runs the ladder, writes predictions
    evaluate.py               metrics, H1 decision, RESULTS.md
  tests/                      one test file per module
```

---

### Task 1: Scaffold, configuration and cached HTTP

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/fed_news/__init__.py`, `src/fed_news/config.py`, `src/fed_news/http.py`
- Test: `tests/test_http.py`

**Interfaces:**
- Produces: `config` constants listed below; `config.ensure_dirs()`; `http.get(url: str, binary: bool = False, retries: int = 3, pause: float = 0.3, cache: bool = True) -> str | bytes` (raises `FileNotFoundError` on HTTP 404).

- [ ] **Step 1: Install test and file-format dependencies**

```bash
PY=/d/gold_lm/venv/Scripts/python.exe
$PY -m pip install -q pytest pyarrow openpyxl xlrd pysentiment2 tabulate
$PY -c "import pytest, pyarrow, openpyxl, xlrd, pysentiment2, tabulate; print('ok')"
```
Expected: `ok`

- [ ] **Step 2: Write `pyproject.toml` and `.gitignore`**

```toml
[project]
name = "fed_news"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
markers = [
    "network: downloads data from the internet",
    "gpu: needs CUDA and cached Hugging Face models",
]
addopts = "-m 'not network and not gpu'"
```

```gitignore
__pycache__/
*.pyc
.pytest_cache/
results/*.pkl
```

- [ ] **Step 3: Write `src/fed_news/__init__.py` (empty) and `src/fed_news/config.py`**

```python
"""Paths and every fixed setting of the design (spec sections 4, 6, 7, 8)."""
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", "D:/hf_cache")

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("FED_NEWS_DATA", "D:/fed_news"))
CACHE = DATA / "cache"
SECRETS = DATA / "secrets"
RESULTS = REPO / "results"

SAMPLE_START = "1994-01-01"
TEST_START = "2001-01-01"
PAPER_T1 = ("2016-01-01", "2025-04-01")
PAPER_T2 = ("2016-01-01", "2025-04-30")

FIRST_VINTAGE, LAST_VINTAGE = 1999, 2024
FIRST_CLASSIFIER_YEAR = 2000
STANCE_SEEDS = (0, 1, 2)
STANCE_LR, STANCE_EPOCHS, STANCE_BATCH, STANCE_MAXLEN = 2e-5, 3, 16, 256
STANCE_HOLDOUT = 0.2
MIN_LABELS = 50

TEXT_WINDOW_DAYS = 30
INDEX_LAG_DAYS = 3
MIN_DOC_CHARS = 500

ORDLOGIT_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
RIDGE_GRID = tuple(float(a) for a in np.logspace(-1, 3, 7))
CV_FOLDS = 5
CLASS_BAND = 0.125
CLASS_EPS = 0.02
ALPHA = 0.05
PREREG_TAG = "prereg"


def ensure_dirs() -> None:
    for path in (DATA, CACHE, SECRETS, RESULTS):
        path.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Write the failing test `tests/test_http.py`**

```python
import pytest

from fed_news import config, http


class FakeResponse:
    def __init__(self, status=200, content=b"hello"):
        self.status_code = status
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(str(self.status_code))


def test_get_caches_to_disk(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(config, "CACHE", tmp_path)
    monkeypatch.setattr(http.requests, "get", lambda *a, **k: calls.append(1) or FakeResponse())
    assert http.get("https://example.org/a", pause=0) == "hello"
    assert http.get("https://example.org/a", pause=0) == "hello"
    assert len(calls) == 1


def test_get_without_cache_refetches(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(config, "CACHE", tmp_path)
    monkeypatch.setattr(http.requests, "get", lambda *a, **k: calls.append(1) or FakeResponse())
    http.get("https://example.org/b", pause=0, cache=False)
    http.get("https://example.org/b", pause=0, cache=False)
    assert len(calls) == 2


def test_get_404_raises_file_not_found_without_retry(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(config, "CACHE", tmp_path)
    monkeypatch.setattr(http.requests, "get", lambda *a, **k: calls.append(1) or FakeResponse(404))
    with pytest.raises(FileNotFoundError):
        http.get("https://example.org/missing", pause=0)
    assert len(calls) == 1
```

- [ ] **Step 5: Run it to verify it fails**

Run: `$PY -m pytest tests/test_http.py -v`
Expected: FAIL with `ImportError: cannot import name 'http'`

- [ ] **Step 6: Write `src/fed_news/http.py`**

```python
"""HTTP GET with retries and an on-disk cache, so every scrape can be re-run offline."""
import hashlib
import time

import requests

from fed_news import config

HEADERS = {"User-Agent": "Mozilla/5.0 (academic research; fed funds forecasting)"}


def get(url: str, binary: bool = False, retries: int = 3, pause: float = 0.3,
        cache: bool = True):
    path = config.CACHE / "http" / hashlib.sha1(url.encode()).hexdigest()
    if cache and path.exists():
        data = path.read_bytes()
    else:
        for attempt in range(retries):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=60)
                if resp.status_code == 404:
                    raise FileNotFoundError(url)
                resp.raise_for_status()
                break
            except requests.RequestException:
                if attempt == retries - 1:
                    raise
                time.sleep(3 * (attempt + 1))
        data = resp.content
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        time.sleep(pause)
    return data if binary else data.decode("utf-8-sig", errors="replace")
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_http.py -v`
Expected: 3 passed

- [ ] **Step 8: Create the data folders and commit**

```bash
PYTHONPATH=src $PY -c "from fed_news import config; config.ensure_dirs(); print(config.DATA)"
git add pyproject.toml .gitignore src tests
git commit -m "Scaffold fed_news package: config and cached HTTP

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
Expected output of the first line: `D:\fed_news`.

---

### Task 2: Daily market data and the target rate

**Files:**
- Create: `src/fed_news/market.py`
- Test: `tests/test_market.py`

**Interfaces:**
- Consumes: `http.get`, `config.DATA`
- Produces: `market.DAILY: dict[str, str]`; `market.fred_csv(series_id: str) -> pd.Series`; `market.splice_target(daily: pd.DataFrame) -> pd.Series`; `market.load_daily(refresh: bool = False) -> pd.DataFrame` with columns `target_point, target_lower, target_upper, eff_rate, tbill_3m, y6m, y1, y2, y10, vix, wti, target`; `market.load_fedfunds(refresh: bool = False) -> pd.Series` (monthly, month-start index); `market.asof(series: pd.Series, cutoff: pd.Timestamp) -> float`.

- [ ] **Step 1: Write the failing test `tests/test_market.py`**

```python
import numpy as np
import pandas as pd
import pytest

from fed_news import market


def test_splice_target_uses_point_then_range_midpoint():
    idx = pd.to_datetime(["2008-12-15", "2008-12-16", "2008-12-17"])
    daily = pd.DataFrame({"target_point": [1.0, np.nan, np.nan],
                          "target_lower": [np.nan, 0.0, 0.0],
                          "target_upper": [np.nan, 0.25, 0.25]}, index=idx)
    assert market.splice_target(daily).tolist() == [1.0, 0.125, 0.125]


def test_asof_is_strictly_before_cutoff():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
    assert market.asof(s, pd.Timestamp("2020-01-03")) == 2.0
    assert np.isnan(market.asof(s, pd.Timestamp("2020-01-01")))


def test_asof_skips_missing_values():
    s = pd.Series([1.0, np.nan], index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    assert market.asof(s, pd.Timestamp("2020-01-05")) == 1.0


@pytest.mark.network
def test_target_around_february_1994_hike():
    daily = market.load_daily(refresh=True)
    assert market.asof(daily["target"], pd.Timestamp("1994-02-04")) == pytest.approx(3.0)
    assert daily["target"].loc["1994-02-07"] == pytest.approx(3.25)
    assert daily["target"].loc["2008-12-17"] == pytest.approx(0.125)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/test_market.py -v`
Expected: FAIL with `ImportError: cannot import name 'market'`

- [ ] **Step 3: Write `src/fed_news/market.py`**

```python
"""Daily FRED series that are never revised: target rate, effective rate, yields, VIX, oil.

The decision variable is the FOMC target, a step function: DFEDTAR (point target) to
2008-12-15, then the midpoint of DFEDTARL/DFEDTARU. FEDFUNDS (monthly average effective
rate) is the paper's T1 target.
"""
import io

import numpy as np
import pandas as pd

from fed_news import config
from fed_news.http import get

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
DAILY = {
    "DFEDTAR": "target_point", "DFEDTARL": "target_lower", "DFEDTARU": "target_upper",
    "DFF": "eff_rate", "DTB3": "tbill_3m", "DGS6MO": "y6m", "DGS1": "y1", "DGS2": "y2",
    "DGS10": "y10", "VIXCLS": "vix", "DCOILWTICO": "wti",
}


def fred_csv(series_id: str) -> pd.Series:
    frame = pd.read_csv(io.StringIO(get(FRED_CSV.format(series_id), cache=False)))
    frame.columns = ["date", "value"]
    values = pd.to_numeric(frame["value"].replace(".", np.nan), errors="coerce")
    return pd.Series(values.to_numpy(), index=pd.to_datetime(frame["date"]), name=series_id)


def splice_target(daily: pd.DataFrame) -> pd.Series:
    midpoint = (daily["target_lower"] + daily["target_upper"]) / 2.0
    return daily["target_point"].where(daily["target_point"].notna(), midpoint).ffill()


def load_daily(refresh: bool = False) -> pd.DataFrame:
    path = config.DATA / "market_daily.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    daily = pd.DataFrame({name: fred_csv(sid) for sid, name in DAILY.items()}).sort_index()
    daily = daily.loc["1990-01-01":]
    daily["target"] = splice_target(daily)
    daily.to_pickle(path)
    return daily


def load_fedfunds(refresh: bool = False) -> pd.Series:
    path = config.DATA / "fedfunds.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    series = fred_csv("FEDFUNDS").loc["1990-01-01":]
    series.to_pickle(path)
    return series


def asof(series: pd.Series, cutoff: pd.Timestamp) -> float:
    """Last non-missing value dated strictly before `cutoff`."""
    sub = series.loc[series.index < cutoff].dropna()
    return float(sub.iloc[-1]) if len(sub) else np.nan
```

- [ ] **Step 4: Run the offline tests, then the network test**

Run: `$PY -m pytest tests/test_market.py -v` then `$PY -m pytest tests/test_market.py -v -m network`
Expected: 3 passed; then 1 passed (this also writes `D:/fed_news/market_daily.pkl`).

- [ ] **Step 5: Fetch FEDFUNDS once and commit**

```bash
PYTHONPATH=src $PY -c "from fed_news import market; s = market.load_fedfunds(refresh=True); print(s.index.min().date(), s.index.max().date(), len(s))"
git add src/fed_news/market.py tests/test_market.py
git commit -m "Add daily market data and target splice

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
Expected print: `1990-01-01 2026-09-01 441` (last month may differ by one).

---

### Task 3: Real-time macro data from ALFRED

**Files:**
- Create: `src/fed_news/alfred.py`
- Test: `tests/test_alfred.py`

**Interfaces:**
- Consumes: `http.get`, `config.DATA`, `config.SECRETS`
- Produces: `alfred.SERIES: dict[str, tuple[str, str, int]]` (name -> FRED id, frequency `"M"`/`"Q"`, release lag days); `alfred.parse_observations(payload: dict) -> pd.DataFrame` with columns `date, realtime_start, realtime_end, value`; `alfred.fetch_vintages(sid: str, refresh: bool = False) -> pd.DataFrame`; `alfred.asof_series(obs: pd.DataFrame, cutoff: pd.Timestamp, freq: str, lag_days: int) -> tuple[pd.Series, bool]` (series as published strictly before cutoff, and whether the fallback was used); `alfred.load_all(refresh: bool = False) -> dict[str, pd.DataFrame]` keyed by the names in `SERIES`.

- [ ] **Step 1: USER ACTION, FRED API key**

Ask the user to create a free key at https://fredaccount.stlouisfed.org/apikeys and save it as one line in `D:/fed_news/secrets/fred_api_key.txt`. Do not continue past Step 5 until the file exists. Never print the key or commit it.

- [ ] **Step 2: Write the failing test `tests/test_alfred.py`**

```python
import pandas as pd
import pytest

from fed_news import alfred

T = pd.Timestamp


def vintages():
    return pd.DataFrame({
        "date": [T("2000-01-01"), T("2000-01-01"), T("2000-02-01")],
        "realtime_start": [T("2000-02-10"), T("2000-03-10"), T("2000-03-10")],
        "realtime_end": [T("2000-03-09"), T("2262-04-01"), T("2262-04-01")],
        "value": [1.0, 1.5, 2.0],
    })


def test_value_released_on_cutoff_day_is_invisible():
    s, fallback = alfred.asof_series(vintages(), T("2000-03-10"), "M", 9)
    assert s.to_dict() == {T("2000-01-01"): 1.0}
    assert fallback is False


def test_revision_visible_the_day_after_release():
    s, _ = alfred.asof_series(vintages(), T("2000-03-11"), "M", 9)
    assert s.to_dict() == {T("2000-01-01"): 1.5, T("2000-02-01"): 2.0}


def test_fallback_before_first_vintage_uses_release_lag():
    s, fallback = alfred.asof_series(vintages(), T("2000-02-10"), "M", 9)
    assert fallback is True
    assert s.to_dict() == {T("2000-01-01"): 1.0}          # Jan 31 + 9 days = Feb 9 < Feb 10
    s, _ = alfred.asof_series(vintages(), T("2000-02-09"), "M", 9)
    assert s.empty                                          # Feb 9 is not strictly before Feb 9


def test_quarterly_fallback_uses_quarter_end():
    obs = pd.DataFrame({"date": [T("2000-01-01")], "realtime_start": [T("2001-01-01")],
                        "realtime_end": [T("2262-04-01")], "value": [100.0]})
    s, _ = alfred.asof_series(obs, T("2000-04-30"), "Q", 31)
    assert s.empty                                          # Mar 31 + 31 days = May 1
    s, _ = alfred.asof_series(obs, T("2000-05-02"), "Q", 31)
    assert len(s) == 1


def test_parse_observations_handles_missing_and_open_end():
    payload = {"observations": [
        {"realtime_start": "2000-02-10", "realtime_end": "9999-12-31", "date": "2000-01-01", "value": "."},
        {"realtime_start": "2000-02-10", "realtime_end": "9999-12-31", "date": "2000-02-01", "value": "2.5"},
    ]}
    frame = alfred.parse_observations(payload)
    assert frame["value"].isna().tolist() == [True, False]
    assert frame["realtime_end"].iloc[0] == T("2262-04-01")


@pytest.mark.network
def test_payems_vintages_download():
    obs = alfred.fetch_vintages("PAYEMS", refresh=True)
    assert obs["realtime_start"].nunique() > 100
    assert obs["date"].min() <= T("1994-01-01")
```

- [ ] **Step 3: Run it to verify it fails**

Run: `$PY -m pytest tests/test_alfred.py -v`
Expected: FAIL with `ImportError: cannot import name 'alfred'`

- [ ] **Step 4: Write `src/fed_news/alfred.py`**

```python
"""Real-time macro data: every published vintage of each series, from ALFRED (spec section 5).

A value is visible at a cutoff only if its vintage was released strictly before the cutoff.
Before a series' first archived vintage, the earliest vintage is used and each period is gated
by a conservative release lag after the period ends; callers count those fallbacks.
"""
import json

import numpy as np
import pandas as pd

from fed_news import config
from fed_news.http import get

API = ("https://api.stlouisfed.org/fred/series/observations?series_id={sid}&api_key={key}"
       "&file_type=json&realtime_start=1776-07-04&realtime_end=9999-12-31"
       "&limit=100000&offset={offset}")

# name -> (FRED id, frequency, release lag in days after the period ends)
SERIES = {
    "m2": ("M2SL", "M", 27),
    "pce": ("PCEPI", "M", 31),
    "pce_core": ("PCEPILFE", "M", 31),
    "payems": ("PAYEMS", "M", 9),
    "earn": ("CES0500000030", "M", 9),
    "gdp": ("GDP", "Q", 31),
    "u6": ("U6RATE", "M", 9),
    "u3": ("UNRATE", "M", 9),
    "ppi": ("PPIACO", "M", 15),
    "cpi": ("CPIAUCSL", "M", 14),
    "cpi_core": ("CPILFESL", "M", 14),
    "hpi": ("HPIPONM226S", "M", 62),
}


def api_key() -> str:
    return (config.SECRETS / "fred_api_key.txt").read_text().strip()


def parse_observations(payload: dict) -> pd.DataFrame:
    frame = pd.DataFrame(payload["observations"])
    frame["value"] = pd.to_numeric(frame["value"].replace(".", np.nan), errors="coerce")
    frame["realtime_end"] = frame["realtime_end"].replace("9999-12-31", "2262-04-01")
    for col in ("date", "realtime_start", "realtime_end"):
        frame[col] = pd.to_datetime(frame[col])
    return frame[["date", "realtime_start", "realtime_end", "value"]]


def fetch_vintages(sid: str, refresh: bool = False) -> pd.DataFrame:
    path = config.DATA / "alfred" / f"{sid}.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    key, parts, offset = api_key(), [], 0
    while True:
        payload = json.loads(get(API.format(sid=sid, key=key, offset=offset), cache=False))
        parts.append(parse_observations(payload))
        offset += len(payload["observations"])
        if not payload["observations"] or offset >= int(payload["count"]):
            break
    frame = pd.concat(parts, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(path)
    return frame


def period_end(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    return index + (pd.offsets.QuarterEnd(0) if freq == "Q" else pd.offsets.MonthEnd(0))


def asof_series(obs: pd.DataFrame, cutoff: pd.Timestamp, freq: str, lag_days: int):
    """The series as published strictly before `cutoff`, and whether the fallback was used."""
    day = cutoff - pd.Timedelta(days=1)
    first = obs["realtime_start"].min()
    if day >= first:
        live = obs[(obs["realtime_start"] <= day) & (obs["realtime_end"] >= day)]
        return live.set_index("date")["value"].sort_index(), False
    earliest = obs[obs["realtime_start"] == first].set_index("date")["value"].sort_index()
    released = period_end(earliest.index, freq) + pd.Timedelta(days=lag_days)
    return earliest[released < cutoff], True


def load_all(refresh: bool = False) -> dict[str, pd.DataFrame]:
    return {name: fetch_vintages(sid, refresh) for name, (sid, _, _) in SERIES.items()}
```

- [ ] **Step 5: Run the offline tests**

Run: `$PY -m pytest tests/test_alfred.py -v`
Expected: 5 passed

- [ ] **Step 6: Download every series and record coverage (needs the key from Step 1)**

```bash
$PY -m pytest tests/test_alfred.py -v -m network
PYTHONPATH=src $PY -c "
from fed_news import alfred
for name, obs in alfred.load_all(refresh=True).items():
    print(f'{name:9s} periods {obs.date.min().date()}..{obs.date.max().date()}  first vintage {obs.realtime_start.min().date()}  rows {len(obs)}')
"
```
Expected: 12 lines. If `CES0500000030` returns HTTP 400 (no ALFRED history), change its entry to `("AHETPI", "M", 9)`, rerun, and add a line to `RESULTS.md` notes later ("weekly earnings unavailable in ALFRED; hourly earnings AHETPI used"). Record each first-vintage date; they go into `PREREG.md`.

- [ ] **Step 7: Commit**

```bash
git add src/fed_news/alfred.py tests/test_alfred.py
git commit -m "Add ALFRED real-time macro data with strict release gating

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: FOMC calendar and the meeting table

**Files:**
- Create: `src/fed_news/fomc_calendar.py`, `src/fed_news/meetings.py`
- Test: `tests/test_calendar.py`, `tests/test_meetings.py`

**Interfaces:**
- Consumes: `http.get`, `market.load_daily`, `market.asof`
- Produces: `fomc_calendar.parse_calendar_html(html: str) -> pd.DataFrame` with columns `kind` (`statement`/`minutes`/`presconf`/`beige`), `date`, `release` (NaT when unknown), `url`; `fomc_calendar.load_calendar(refresh: bool = False) -> pd.DataFrame`; `meetings.build_meetings(cal: pd.DataFrame, target: pd.Series) -> pd.DataFrame` indexed by decision `date` with columns `cutoff, target_before, target_after, y_change, y_class, unscheduled`; `meetings.load_meetings(refresh: bool = False) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing test `tests/test_calendar.py`**

```python
import pandas as pd
import pytest

from fed_news import fomc_calendar

SNIPPET = """
<a href="/fomc/minutes/20050202.htm">Minutes</a> (Released Feb 23, 2005)
<a href="http://www.federalreserve.gov/fomc/beigebook/2005/20050119/default.htm">Beige</a>
<a href="/boarddocs/press/monetary/2005/20050202/">Statement</a>
<a href="/monetarypolicy/fomcminutes20220126.htm">Minutes</a> | Released February 16, 2022
<a href="/monetarypolicy/fomcpresconf20220126.htm">Press Conference</a>
<a href="/newsevents/pressreleases/monetary20220126a.htm">Statement</a>
<a href="/monetarypolicy/beigebook202201.htm">Beige Book</a>
<a href="/fomc/MINUTES/1994/19940204min.htm">Minutes</a> Released March 25, 1994
"""


def test_parse_finds_every_kind_with_dates():
    cal = fomc_calendar.parse_calendar_html(SNIPPET)
    got = {(r.kind, r.date.strftime("%Y%m%d")) for r in cal.itertuples()}
    assert ("minutes", "20050202") in got and ("minutes", "20220126") in got
    assert ("minutes", "19940204") in got and ("statement", "20050202") in got
    assert ("statement", "20220126") in got and ("presconf", "20220126") in got
    assert ("beige", "20050119") in got and ("beige", "20220101") in got


def test_minutes_release_dates_are_parsed():
    cal = fomc_calendar.parse_calendar_html(SNIPPET).set_index(["kind", "date"])
    assert cal.loc[("minutes", pd.Timestamp("2005-02-02")), "release"] == pd.Timestamp("2005-02-23")
    assert cal.loc[("minutes", pd.Timestamp("2022-01-26")), "release"] == pd.Timestamp("2022-02-16")
    assert cal.loc[("minutes", pd.Timestamp("1994-02-04")), "release"] == pd.Timestamp("1994-03-25")


def test_six_digit_beige_has_unknown_release_and_statement_release_is_its_date():
    cal = fomc_calendar.parse_calendar_html(SNIPPET).set_index(["kind", "date"])
    assert pd.isna(cal.loc[("beige", pd.Timestamp("2022-01-01")), "release"])
    assert cal.loc[("statement", pd.Timestamp("2022-01-26")), "release"] == pd.Timestamp("2022-01-26")


def test_implausible_release_is_dropped_to_nat():
    html = '<a href="/fomc/minutes/20050202.htm">M</a> Released Feb 23, 2011'
    cal = fomc_calendar.parse_calendar_html(html)
    assert pd.isna(cal["release"].iloc[0])


@pytest.mark.network
def test_live_calendar_2005_minutes_release():
    cal = fomc_calendar.load_calendar(refresh=True).set_index(["kind", "date"])
    assert cal.loc[("minutes", pd.Timestamp("2005-03-22")), "release"] == pd.Timestamp("2005-04-12")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/test_calendar.py -v`
Expected: FAIL with `ImportError: cannot import name 'fomc_calendar'`

- [ ] **Step 3: Write `src/fed_news/fomc_calendar.py`**

```python
"""Parse the Fed's FOMC calendar pages into dated links (statements, minutes, press
conferences, Beige Books). Historical pages cover 1994-2020; the current page covers 2021 on.

Minutes are released weeks after the meeting (after the next meeting before 2005), so their
release date is read from the "Released ..." text that follows each minutes link. A release
date that cannot be read, or is not within 0-120 days after the meeting, is left as NaT and
the document is later dropped rather than dated by its meeting.
"""
import re

import pandas as pd

from fed_news import config
from fed_news.http import get

BASE = "https://www.federalreserve.gov"
HISTORICAL = BASE + "/monetarypolicy/fomchistorical{}.htm"
CURRENT = BASE + "/monetarypolicy/fomccalendars.htm"

PATTERNS = {
    "statement": [r"/newsevents/pressreleases/monetary(\d{8})a\.htm",
                  r"/newsevents/press/monetary/(\d{8})a\.htm",
                  r"/boarddocs/press/(?:monetary|general)/\d{4}/(\d{8})/"],
    "minutes": [r"/fomc/MINUTES/\d{4}/(\d{8})min\.htm", r"/fomc/minutes/(\d{8})\.htm",
                r"/monetarypolicy/fomcminutes(\d{8})\.htm"],
    "presconf": [r"/monetarypolicy/fomcpresconf(\d{8})\.htm"],
    "beige": [r"/monetarypolicy/files/fomc\d{8}beige(\d{8})\.pdf",
              r"/fomc/beigebook/\d{4}/(\d{8})/default\.htm",
              r"/monetarypolicy/beigebook(\d{6})\.htm"],
}
RELEASED = re.compile(r"Released\s+([A-Z][a-z]+\.?\s+\d{1,2},\s+\d{4})")


def _release_after(html: str, end: int, date: pd.Timestamp) -> pd.Timestamp:
    found = RELEASED.search(html, end, end + 800)
    if not found:
        return pd.NaT
    release = pd.Timestamp(pd.to_datetime(found.group(1).replace(".", "")))
    if not (date <= release <= date + pd.Timedelta(days=120)):
        return pd.NaT
    return release


def parse_calendar_html(html: str) -> pd.DataFrame:
    rows = []
    for kind, patterns in PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, html, re.I):
                path, raw = match.group(0), match.group(1)
                url = BASE + path + ("default.htm" if path.endswith("/") else "")
                date = pd.to_datetime(raw, format="%Y%m%d" if len(raw) == 8 else "%Y%m")
                if kind == "minutes":
                    release = _release_after(html, match.end(), date)
                elif kind == "beige" and len(raw) == 6:
                    release = pd.NaT
                else:
                    release = date
                rows.append({"kind": kind, "date": date, "release": release, "url": url})
    frame = pd.DataFrame(rows, columns=["kind", "date", "release", "url"])
    return frame.drop_duplicates(subset=["kind", "date"]).reset_index(drop=True)


def load_calendar(refresh: bool = False) -> pd.DataFrame:
    path = config.DATA / "calendar.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    frames = [parse_calendar_html(get(HISTORICAL.format(year))) for year in range(1994, 2021)]
    frames.append(parse_calendar_html(get(CURRENT, cache=False)))
    cal = (pd.concat(frames, ignore_index=True)
           .drop_duplicates(subset=["kind", "date"])
           .sort_values(["kind", "date"]).reset_index(drop=True))
    cal.to_pickle(path)
    return cal
```

- [ ] **Step 4: Run calendar tests**

Run: `$PY -m pytest tests/test_calendar.py -v` then `$PY -m pytest tests/test_calendar.py -v -m network`
Expected: 4 passed; then 1 passed.

- [ ] **Step 5: Write the failing test `tests/test_meetings.py`**

```python
import pandas as pd
import pytest

from fed_news import meetings

T = pd.Timestamp


def cal(rows):
    return pd.DataFrame([{"kind": k, "date": T(d), "release": T(d), "url": ""} for k, d in rows])


def target(changes, start="2005-01-01", end="2005-04-30", level=2.0):
    idx = pd.date_range(start, end, freq="D")
    s = pd.Series(level, index=idx)
    for day, new in changes:
        s.loc[T(day):] = new
    return s


def test_two_day_meeting_is_merged_and_classified():
    c = cal([("minutes", "2005-02-01"), ("statement", "2005-02-02"), ("minutes", "2005-03-22")])
    m = meetings.build_meetings(c, target([("2005-02-02", 2.25)]))
    assert list(m.index) == [T("2005-02-02"), T("2005-03-22")]
    assert m.loc["2005-02-02", "y_class"] == 2 and m.loc["2005-03-22", "y_class"] == 1
    assert m.loc["2005-02-02", "cutoff"] == T("2005-02-01")
    assert m.loc["2005-02-02", "y_change"] == pytest.approx(0.25)


def test_target_change_without_calendar_entry_is_unscheduled():
    c = cal([("minutes", "2005-02-02"), ("minutes", "2005-03-22")])
    m = meetings.build_meetings(c, target([("2005-03-01", 1.75)]))
    assert T("2005-03-01") in m.index
    assert m.loc["2005-03-01", "unscheduled"] == 1 and m.loc["2005-03-01", "y_class"] == 0


def test_sunday_unscheduled_cut():
    c = cal([("minutes", "2020-03-03"), ("statement", "2020-03-15"), ("minutes", "2020-04-29")])
    s = target([("2020-03-04", 1.125), ("2020-03-16", 0.125)], "2020-01-01", "2020-05-31", 1.625)
    m = meetings.build_meetings(c, s)
    assert m.loc["2020-03-15", "y_class"] == 0
    assert m.loc["2020-03-15", "target_before"] == pytest.approx(1.125)
    assert m.loc["2020-03-15", "target_after"] == pytest.approx(0.125)
    assert m.loc["2020-03-15", "unscheduled"] == 1
```

- [ ] **Step 6: Run it to verify it fails**

Run: `$PY -m pytest tests/test_meetings.py -v`
Expected: FAIL with `ImportError: cannot import name 'meetings'`

- [ ] **Step 7: Write `src/fed_news/meetings.py`**

```python
"""One row per FOMC decision from 1994 (spec section 4, T2).

Decision dates are the union of minutes-link and statement dates (dates within two days are
merged to the later one), plus any target change more than three days after every such date.
The decision is the target in force up to four days after the meeting (or the day before the
next meeting) minus the target in force before it. Cutoff = the day before the decision date.
"""
import numpy as np
import pandas as pd

from fed_news import config, market
from fed_news.fomc_calendar import load_calendar


def _decision_dates(cal: pd.DataFrame, target: pd.Series) -> list[pd.Timestamp]:
    dates = sorted(set(cal.loc[cal["kind"].isin(["minutes", "statement"]), "date"]))
    merged: list[pd.Timestamp] = []
    for day in dates:
        if merged and (day - merged[-1]).days <= 2:
            merged[-1] = day
        else:
            merged.append(day)
    steps = target.diff()
    for day in steps.index[steps.abs() > 1e-9]:
        if not any(0 <= (day - m).days <= 3 for m in merged):
            merged.append(day)
    return sorted(merged)


def build_meetings(cal: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    minutes_dates = set(cal.loc[cal["kind"] == "minutes", "date"])
    dates = [d for d in _decision_dates(cal, target)
             if d >= pd.Timestamp(config.SAMPLE_START) and d + pd.Timedelta(days=4) <= target.index.max()]
    rows = []
    for i, day in enumerate(dates):
        nxt = dates[i + 1] if i + 1 < len(dates) else day + pd.Timedelta(days=60)
        horizon = min(day + pd.Timedelta(days=5), nxt)
        before = market.asof(target, day)
        after = market.asof(target, horizon)
        change = after - before
        rows.append({
            "date": day, "cutoff": day - pd.Timedelta(days=1),
            "target_before": before, "target_after": after, "y_change": change,
            "y_class": 0 if change < -config.CLASS_EPS else (2 if change > config.CLASS_EPS else 1),
            "unscheduled": int(not any(abs((day - m).days) <= 2 for m in minutes_dates)),
        })
    return pd.DataFrame(rows).set_index("date")


def load_meetings(refresh: bool = False) -> pd.DataFrame:
    path = config.DATA / "meetings.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    frame = build_meetings(load_calendar(), market.load_daily()["target"])
    frame.to_pickle(path)
    return frame


if __name__ == "__main__":
    m = load_meetings(refresh=True)
    print(f"meetings {len(m)}  {m.index.min().date()} .. {m.index.max().date()}")
    print(m["y_class"].value_counts().sort_index().to_string())
    print("unscheduled:", list(m.index[m["unscheduled"] == 1].date))
```

- [ ] **Step 8: Run tests, build the table, compare with `fedfunds_gpt2` from 2000**

```bash
$PY -m pytest tests/test_meetings.py -v
PYTHONPATH=src $PY -m fed_news.meetings
PYTHONPATH=src $PY -c "
import pandas as pd
from fed_news import meetings
ours = set(meetings.load_meetings().loc['2000-01-01':'2026-09-30'].index)
theirs = set(pd.read_pickle('C:/Users/User/fedfunds_gpt2/data/meetings.pkl').loc['2000-01-01':].index.normalize())
print('only ours', sorted(ours - theirs)); print('only theirs', sorted(theirs - ours))
"
```
Expected: 3 passed; about 270 meetings from 1994; the unscheduled list includes 1994-04-18, 1998-10-15, 2001-01-03, 2001-04-18, 2001-09-17, 2008-01-22, 2008-10-08, 2020-03-03 and 2020-03-15. Both "only" lists should be empty. If they are not, inspect each date in the calendar pickle and fix the parser (not the comparison) before committing.

- [ ] **Step 9: Commit**

```bash
git add src/fed_news/fomc_calendar.py src/fed_news/meetings.py tests/test_calendar.py tests/test_meetings.py
git commit -m "Add FOMC calendar parser and meeting table from 1994

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Fed document corpus

**Files:**
- Create: `src/fed_news/corpus.py`
- Test: `tests/test_corpus.py`

**Interfaces:**
- Consumes: `http.get`, `fomc_calendar.load_calendar`, `fomc_calendar.BASE`
- Produces: `corpus.DOC_COLS = ["doc_id", "type", "speaker", "release", "text"]`; `corpus.html_to_text(html: str) -> str`; `corpus.pdf_to_text(data: bytes) -> str`; `corpus.split_sentences(text: str) -> list[str]`; `corpus.beige_release(text: str, month: pd.Timestamp) -> pd.Timestamp`; `corpus.docs_from_calendar(cal: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, int]` (documents, number dropped); `corpus.parse_speech_json(raw: str) -> pd.DataFrame` and `corpus.parse_old_speech_list(page: str) -> pd.DataFrame` (columns `url, release, speaker`); `corpus.build_corpus(refresh: bool = False) -> pd.DataFrame`; `corpus.coverage(docs: pd.DataFrame) -> pd.DataFrame` (year x type counts); `corpus.check_coverage(cov: pd.DataFrame, years, required) -> None` raising `corpus.CoverageError`. Document `type` values: `statement`, `minutes`, `presconf`, `beige`, `speech`.

- [ ] **Step 1: Write the failing test `tests/test_corpus.py`**

```python
import pandas as pd
import pytest

from fed_news import corpus

T = pd.Timestamp


def test_html_to_text_prefers_article_and_drops_scripts():
    html = ("<html><body><nav>menu</nav><div id='article'><p>The Committee decided.</p>"
            "<script>x=1</script></div><footer>foot</footer></body></html>")
    assert corpus.html_to_text(html) == "The Committee decided."


def test_split_sentences_keeps_reasonable_lengths():
    text = ("Inflation has moved up over the past year and remains elevated. Ok. "
            "The Committee decided to raise the target range for the federal funds rate.")
    assert corpus.split_sentences(text) == [
        "Inflation has moved up over the past year and remains elevated.",
        "The Committee decided to raise the target range for the federal funds rate."]


def test_beige_release_matches_url_month():
    text = "Last Update: December 3, 2018 ... Beige Book January 16, 2019 Summary"
    assert corpus.beige_release(text, T("2019-01-01")) == T("2019-01-16")
    assert pd.isna(corpus.beige_release("no dates here", T("2019-01-01")))


def test_minutes_without_release_are_dropped(monkeypatch):
    cal = pd.DataFrame([
        {"kind": "minutes", "date": T("2005-02-02"), "release": pd.NaT, "url": "u1"},
        {"kind": "minutes", "date": T("2005-03-22"), "release": T("2005-04-12"), "url": "u2"},
    ])
    monkeypatch.setattr(corpus, "get", lambda url, **k: "<div id='article'>" + "word " * 200 + "</div>")
    docs, dropped = corpus.docs_from_calendar(cal, "minutes")
    assert dropped == 1
    assert docs["release"].tolist() == [T("2005-04-12")]
    assert list(docs.columns) == corpus.DOC_COLS


def test_parse_speech_json():
    raw = ('\ufeff[{"d":"10/8/2026 4:30:00 AM","t":"Title","s":"Governor A &amp; B",'
           '"l":"/newsevents/speech/waller20261008a.htm"}]')
    frame = corpus.parse_speech_json(raw)
    assert frame.loc[0, "release"] == T("2026-10-08")
    assert frame.loc[0, "speaker"] == "Governor A & B"
    assert frame.loc[0, "url"].endswith("/newsevents/speech/waller20261008a.htm")


def test_parse_old_speech_list():
    page = ('<a href="/boarddocs/speeches/2002/20021121/default.htm">x</a>'
            '<a href="http://www.federalreserve.gov/boarddocs/speeches/2002/200211212/default.htm">y</a>')
    frame = corpus.parse_old_speech_list(page)
    assert frame["release"].tolist() == [T("2002-11-21"), T("2002-11-21")]
    assert all(u.startswith("https://www.federalreserve.gov/boarddocs/") for u in frame["url"])


def test_check_coverage_raises_on_missing_year():
    docs = pd.DataFrame({"type": ["statement", "minutes", "statement"],
                         "release": [T("2001-03-20"), T("2001-04-12"), T("2002-01-30")]})
    cov = corpus.coverage(docs)
    corpus.check_coverage(cov, [2001], ("statement", "minutes"))
    with pytest.raises(corpus.CoverageError, match="2002"):
        corpus.check_coverage(cov, [2001, 2002], ("statement", "minutes"))


@pytest.mark.network
def test_live_speech_json_is_parseable():
    frame = corpus.parse_speech_json(corpus.get(corpus.SPEECH_JSON, cache=False))
    assert len(frame) > 500 and frame["release"].min() < T("2008-01-01")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/test_corpus.py -v`
Expected: FAIL with `ImportError: cannot import name 'corpus'`

- [ ] **Step 3: Write `src/fed_news/corpus.py`**

```python
"""The Fed's own documents, each with the date it became public (spec section 5, N).

Statements and press conferences are public on the decision day; minutes on their stated
release date; Beige Books on their publication date; speeches on the day delivered. A document
without a usable release date, or with under MIN_DOC_CHARS characters of text, is dropped and
counted. Every text is cached by URL, so a rebuild is offline.
"""
import hashlib
import html as htmllib
import io
import json
import re

import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader

from fed_news import config
from fed_news.fomc_calendar import BASE, load_calendar
from fed_news.http import get

DOC_COLS = ["doc_id", "type", "speaker", "release", "text"]
TYPE_OF_KIND = {"statement": "statement", "minutes": "minutes", "presconf": "presconf", "beige": "beige"}
PRESCONF_PDF = BASE + "/mediacenter/files/FOMCpresconf{:%Y%m%d}.pdf"
SPEECH_JSON = BASE + "/json/ne-speeches.json"
OLD_SPEECH_LIST = BASE + "/newsevents/speech/{}speech.htm"
OLD_SPEECH_LINK = re.compile(
    r'href="((?:https?://www\.federalreserve\.gov)?/boarddocs/speeches/\d{4}/(\d{8})\d*/default\.htm)"', re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
MONTH_DATE = re.compile(r"(January|February|March|April|May|June|July|August|September|October|"
                        r"November|December)\s+(\d{1,2}),\s+(\d{4})")


class CoverageError(RuntimeError):
    pass


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    node = (soup.find(id="article") or soup.find("div", class_="col-xs-12 col-sm-8 col-md-8")
            or soup.body or soup)
    return re.sub(r"\s+", " ", node.get_text(" ")).strip()


def pdf_to_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    return re.sub(r"\s+", " ", " ".join(page.extract_text() or "" for page in reader.pages)).strip()


def split_sentences(text: str) -> list[str]:
    parts = (s.strip() for s in SENT_SPLIT.split(text))
    return [s for s in parts if 30 <= len(s) <= 1000]


def beige_release(text: str, month: pd.Timestamp) -> pd.Timestamp:
    for m in MONTH_DATE.finditer(text):
        day = pd.Timestamp(f"{m.group(1)} {m.group(2)} {m.group(3)}")
        if day.year == month.year and day.month == month.month:
            return day
    return pd.NaT


def _doc_id(doc_type: str, release: pd.Timestamp, url: str) -> str:
    return f"{doc_type}:{release:%Y%m%d}:{hashlib.sha1(url.encode()).hexdigest()[:8]}"


def _text(url: str) -> str:
    return pdf_to_text(get(url, binary=True)) if url.lower().endswith(".pdf") else html_to_text(get(url))


def docs_from_calendar(cal: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, int]:
    rows, dropped = [], 0
    for r in cal[cal["kind"] == kind].itertuples():
        try:
            if kind == "presconf":
                try:
                    text = pdf_to_text(get(PRESCONF_PDF.format(r.date), binary=True))
                except FileNotFoundError:
                    text = _text(r.url)
            else:
                text = _text(r.url)
        except FileNotFoundError:
            dropped += 1
            continue
        release = r.release
        if kind == "beige" and pd.isna(release):
            release = beige_release(text, r.date)
        if pd.isna(release) or len(text) < config.MIN_DOC_CHARS:
            dropped += 1
            continue
        doc_type = TYPE_OF_KIND[kind]
        rows.append({"doc_id": _doc_id(doc_type, release, r.url), "type": doc_type,
                     "speaker": "", "release": pd.Timestamp(release), "text": text})
    return pd.DataFrame(rows, columns=DOC_COLS), dropped


def parse_speech_json(raw: str) -> pd.DataFrame:
    rows = []
    for item in json.loads(raw.lstrip("\ufeff")):
        link = item.get("l", "")
        if not link:
            continue
        rows.append({"url": BASE + link if link.startswith("/") else link,
                     "release": pd.to_datetime(item["d"], format="%m/%d/%Y %I:%M:%S %p").normalize(),
                     "speaker": htmllib.unescape(item.get("s", ""))})
    return pd.DataFrame(rows, columns=["url", "release", "speaker"])


def parse_old_speech_list(page: str) -> pd.DataFrame:
    rows = []
    for link, day in OLD_SPEECH_LINK.findall(page):
        url = link.replace("http://", "https://")
        rows.append({"url": url if url.startswith("https") else BASE + url,
                     "release": pd.to_datetime(day, format="%Y%m%d"), "speaker": ""})
    return pd.DataFrame(rows, columns=["url", "release", "speaker"]).drop_duplicates("url")


def fetch_speeches() -> tuple[pd.DataFrame, int]:
    listing = [parse_speech_json(get(SPEECH_JSON, cache=False))]
    for year in range(1996, 2006):
        try:
            listing.append(parse_old_speech_list(get(OLD_SPEECH_LIST.format(year))))
        except FileNotFoundError:
            continue
    listing = pd.concat(listing, ignore_index=True).drop_duplicates("url")
    rows, dropped = [], 0
    for r in listing.itertuples():
        try:
            text = _text(r.url)
        except FileNotFoundError:
            dropped += 1
            continue
        if len(text) < config.MIN_DOC_CHARS:
            dropped += 1
            continue
        rows.append({"doc_id": _doc_id("speech", r.release, r.url), "type": "speech",
                     "speaker": r.speaker, "release": r.release, "text": text})
    return pd.DataFrame(rows, columns=DOC_COLS), dropped


def coverage(docs: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(docs["release"].dt.year, docs["type"])


def check_coverage(cov: pd.DataFrame, years, required) -> None:
    missing = [f"{year}:{t}" for year in years for t in required
               if t not in cov.columns or year not in cov.index or cov.loc[year, t] == 0]
    if missing:
        raise CoverageError("no documents for " + ", ".join(missing))


def build_corpus(refresh: bool = False) -> pd.DataFrame:
    path = config.DATA / "corpus.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    cal = load_calendar()
    parts, dropped = [], {}
    for kind in ("statement", "minutes", "presconf", "beige"):
        docs, n = docs_from_calendar(cal, kind)
        parts.append(docs)
        dropped[kind] = n
        print(f"{kind:9s} kept {len(docs):5d} dropped {n}", flush=True)
    speeches, n = fetch_speeches()
    parts.append(speeches)
    dropped["speech"] = n
    print(f"speech    kept {len(speeches):5d} dropped {n}", flush=True)
    docs = pd.concat(parts, ignore_index=True).drop_duplicates("doc_id")
    docs = docs[docs["release"] >= pd.Timestamp("1993-06-01")].sort_values("release").reset_index(drop=True)
    docs.to_pickle(path)
    config.RESULTS.mkdir(exist_ok=True)
    coverage(docs).to_csv(config.RESULTS / "coverage.csv")
    (config.RESULTS / "dropped_docs.json").write_text(json.dumps(dropped, indent=1))
    return docs


if __name__ == "__main__":
    corpus = build_corpus(refresh=True)
    cov = coverage(corpus)
    print(cov.to_string())
    check_coverage(cov, range(2001, corpus["release"].dt.year.max() + 1), ("statement", "minutes"))
    print("coverage check passed")
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_corpus.py -v` then `$PY -m pytest tests/test_corpus.py -v -m network`
Expected: 7 passed; then 1 passed.

- [ ] **Step 5: Build the corpus in the background (about one hour, about 3,500 pages)**

```bash
mkdir -p logs && PYTHONPATH=src nohup $PY -u -m fed_news.corpus > logs/corpus.log 2>&1 &
```
Wait for it to finish (check `tail logs/corpus.log`). Expected ending: a year-by-type table and `coverage check passed`. If the check raises `CoverageError`, open the listed years' calendar pages and fix the regex in `fomc_calendar.PATTERNS` or `corpus`; then rebuild. Do not relax `check_coverage`.

- [ ] **Step 6: Commit**

```bash
git add src/fed_news/corpus.py tests/test_corpus.py results/coverage.csv results/dropped_docs.json
git commit -m "Add Fed document corpus with release dates and coverage check

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Newspaper indices

**Files:**
- Create: `src/fed_news/indices.py`
- Test: `tests/test_indices.py`

**Interfaces:**
- Consumes: `http.get`
- Produces: `indices.load_indices(refresh: bool = False) -> pd.DataFrame` (daily index; columns `news`, `gpr`, `epu`); `indices.window_mean(series: pd.Series, cutoff: pd.Timestamp, days: int = 30, lag: int = 3) -> float` (mean over dates d with `cutoff - lag - days <= d < cutoff - lag`).

- [ ] **Step 1: Write the failing test `tests/test_indices.py`**

```python
import numpy as np
import pandas as pd
import pytest

from fed_news import indices


def test_window_mean_excludes_last_lag_days_and_cutoff():
    s = pd.Series(np.arange(10.0), index=pd.date_range("2020-01-01", periods=10))
    # cutoff Jan 10, lag 3 -> end Jan 7 (exclusive), days 3 -> Jan 4, 5, 6
    assert indices.window_mean(s, pd.Timestamp("2020-01-10"), days=3, lag=3) == pytest.approx(4.0)


def test_window_mean_empty_is_nan():
    s = pd.Series([1.0], index=pd.to_datetime(["2020-01-01"]))
    assert np.isnan(indices.window_mean(s, pd.Timestamp("2019-01-01")))


@pytest.mark.network
def test_live_indices_cover_sample():
    frame = indices.load_indices(refresh=True)
    assert list(frame.columns) == ["news", "gpr", "epu"]
    assert frame.index.min() <= pd.Timestamp("1993-01-01")
    assert frame.loc["1994":, ["news", "gpr", "epu"]].notna().mean().min() > 0.95
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/test_indices.py -v`
Expected: FAIL with `ImportError: cannot import name 'indices'`

- [ ] **Step 3: Write `src/fed_news/indices.py`**

```python
"""Three daily newspaper-based indices (spec section 6, F4-F6).

news: SF Fed Daily News Sentiment Index (sheet "Data", columns date / News Sentiment).
gpr:  Caldara-Iacoviello daily Geopolitical Risk index (column GPRD).
epu:  Baker-Bloom-Davis daily Economic Policy Uncertainty (day, month, year, daily_policy_index).
Each feature is a mean over a 30-day window that ends three days before the cutoff.
"""
import io

import numpy as np
import pandas as pd

from fed_news import config
from fed_news.http import get

SF_URL = "https://www.frbsf.org/wp-content/uploads/news_sentiment_data.xlsx"
GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
EPU_URL = "https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv"


def load_indices(refresh: bool = False) -> pd.DataFrame:
    path = config.DATA / "indices.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    sf = pd.read_excel(io.BytesIO(get(SF_URL, binary=True, cache=False)), sheet_name="Data")
    news = pd.Series(sf["News Sentiment"].to_numpy(float), index=pd.to_datetime(sf["date"]))
    gpr_raw = pd.read_excel(io.BytesIO(get(GPR_URL, binary=True, cache=False)), engine="xlrd")
    gpr_raw["date"] = pd.to_datetime(gpr_raw["date"], errors="coerce")
    gpr_raw = gpr_raw.dropna(subset=["date"])
    gpr = pd.Series(gpr_raw["GPRD"].to_numpy(float), index=gpr_raw["date"])
    epu_raw = pd.read_csv(io.StringIO(get(EPU_URL, cache=False)))
    epu_idx = pd.to_datetime(epu_raw[["year", "month", "day"]])
    epu = pd.Series(epu_raw["daily_policy_index"].to_numpy(float), index=epu_idx)
    frame = pd.DataFrame({
        "news": news[~news.index.duplicated(keep="last")],
        "gpr": gpr[~gpr.index.duplicated(keep="last")],
        "epu": epu[~epu.index.duplicated(keep="last")],
    }).sort_index()
    frame.to_pickle(path)
    return frame


def window_mean(series: pd.Series, cutoff: pd.Timestamp, days: int = config.TEXT_WINDOW_DAYS,
                lag: int = config.INDEX_LAG_DAYS) -> float:
    end = cutoff - pd.Timedelta(days=lag)
    start = end - pd.Timedelta(days=days)
    sub = series[(series.index >= start) & (series.index < end)].dropna()
    return float(sub.mean()) if len(sub) else np.nan
```

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_indices.py -v` then `$PY -m pytest tests/test_indices.py -v -m network`
Expected: 2 passed; then 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fed_news/indices.py tests/test_indices.py
git commit -m "Add SF Fed sentiment, GPR and EPU daily indices

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Panels for T1 and T2 (targets, S, M, Taylor inputs)

**Files:**
- Create: `src/fed_news/panel.py`
- Test: `tests/test_panel.py`

**Interfaces:**
- Consumes: `market.load_daily`, `market.load_fedfunds`, `market.asof`, `alfred.SERIES`, `alfred.asof_series`, `alfred.load_all`, `meetings.load_meetings`
- Produces: `panel.S_COLS`, `panel.M_COLS`, `panel.N_COLS`, `panel.TAYLOR_COLS` (lists of str); `panel.BLOCKS: dict[str, list[str]]`; `panel.feature_columns(blocks: str) -> list[str]` (e.g. `"SN"`); `panel.macro_at(alfred_obs: dict, cutoff: pd.Timestamp) -> tuple[dict, int]`; `panel.market_at(daily: pd.DataFrame, cutoff: pd.Timestamp) -> dict`; `panel.build_t1(daily, fedfunds, alfred_obs) -> pd.DataFrame` and `panel.build_t2(meet, daily, alfred_obs) -> pd.DataFrame`, both indexed by date with columns `cutoff, win_start, y_level, y_change, s_rate_lag`, the S, M, Taylor columns, `fallback_n`, and for T2 also `y_class`; `panel.build_panels() -> tuple[pd.DataFrame, pd.DataFrame]` saving `D:/fed_news/t1.pkl` and `t2.pkl`. For T1, `y_class` is also added (from `y_change` with the 0.02 tolerance) so evaluation code is shared.

- [ ] **Step 1: Write the failing test `tests/test_panel.py`**

```python
import numpy as np
import pandas as pd
import pytest

from fed_news import panel

T = pd.Timestamp


def flat_vintage(level_by_month: dict, released: str):
    dates = list(level_by_month)
    return pd.DataFrame({"date": [T(d) for d in dates], "realtime_start": T(released),
                         "realtime_end": T("2262-04-01"), "value": list(level_by_month.values())})


def test_feature_columns_never_include_targets():
    cols = panel.feature_columns("SMN")
    assert cols == panel.S_COLS + panel.M_COLS + panel.N_COLS
    assert not any(c.startswith("y_") for c in cols)


def test_s_cols_match_paper_table_1():
    assert panel.S_COLS == [
        "s_rate_lag", "s_m2_mom", "s_m2_yoy", "s_pce_yoy", "s_pce_core_yoy", "s_payems_chg",
        "s_earn_yoy", "s_gdp_yoy", "s_u6_mom", "s_u6_yoy", "s_ppi_yoy", "s_cpi_yoy",
        "s_cpi_core_yoy", "s_hpi_yoy", "s_slope_10y2y", "s_slope_1y6m"]


def test_macro_at_computes_yoy_within_one_vintage():
    months = pd.date_range("1999-01-01", "2000-01-01", freq="MS")
    obs = flat_vintage({m: 100.0 + i for i, m in enumerate(months)}, "2000-02-15")
    alfred_obs = {name: obs for name in ["m2", "pce", "pce_core", "payems", "earn", "u6", "u3",
                                         "ppi", "cpi", "cpi_core", "hpi"]}
    q = pd.date_range("1999-01-01", "2000-01-01", freq="QS")
    alfred_obs["gdp"] = flat_vintage({d: 200.0 + i for i, d in enumerate(q)}, "2000-02-15")
    out, fallbacks = panel.macro_at(alfred_obs, T("2000-02-16"))
    assert out["s_cpi_yoy"] == pytest.approx((112.0 / 100.0 - 1) * 100)
    assert out["s_payems_chg"] == pytest.approx(1.0)
    assert out["s_u6_yoy"] == pytest.approx(12.0)
    assert out["t_u3"] == pytest.approx(112.0)
    assert out["s_gdp_yoy"] == pytest.approx((204.0 / 200.0 - 1) * 100)
    assert fallbacks == 0
    # cutoff on the first vintage's release day: that vintage is invisible, so all 12 series
    # fall back to release-lag gating (strictness itself is tested in test_alfred.py)
    _, fallbacks = panel.macro_at(alfred_obs, T("2000-02-15"))
    assert fallbacks == 12


def test_market_at_uses_target_before_cutoff():
    idx = pd.date_range("2020-01-01", "2020-01-10")
    daily = pd.DataFrame({"target": 1.0, "tbill_3m": 1.5, "y6m": 1.6, "y1": 1.7, "y2": 1.8,
                          "y10": 2.0, "vix": 15.0, "wti": 60.0}, index=idx)
    daily.loc["2020-01-10":, "target"] = 9.0
    out = panel.market_at(daily, T("2020-01-10"))
    assert out["m_sp_tbill_3m"] == pytest.approx(0.5)
    assert out["s_slope_10y2y"] == pytest.approx(0.2)
    assert out["s_slope_1y6m"] == pytest.approx(0.1)
    assert out["m_zero_bound"] == 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/test_panel.py -v`
Expected: FAIL with `ImportError: cannot import name 'panel'`

- [ ] **Step 3: Write `src/fed_news/panel.py`**

```python
"""Rows for the two tasks (spec sections 4-5).

T1: one row per month t from 1994; cutoff = first day of month t; y_level = FEDFUNDS(t);
    s_rate_lag = FEDFUNDS(t-1); text window [first day of t-1, cutoff).
T2: one row per FOMC decision; cutoff = day before the decision; y_level = target after;
    s_rate_lag = target before; text window [previous meeting's cutoff, cutoff).
S = the paper's Table 1 statistics, real-time; M = our market block; t_u3 feeds the Taylor rule.
"""
import numpy as np
import pandas as pd

from fed_news import alfred, config, market
from fed_news.meetings import load_meetings

S_COLS = ["s_rate_lag", "s_m2_mom", "s_m2_yoy", "s_pce_yoy", "s_pce_core_yoy", "s_payems_chg",
          "s_earn_yoy", "s_gdp_yoy", "s_u6_mom", "s_u6_yoy", "s_ppi_yoy", "s_cpi_yoy",
          "s_cpi_core_yoy", "s_hpi_yoy", "s_slope_10y2y", "s_slope_1y6m"]
M_COLS = ["m_sp_tbill_3m", "m_sp_y6m", "m_sp_y1", "m_sp_y2", "m_target_level", "m_zero_bound",
          "m_vix", "m_wti"]
N_COLS = ["n_stance", "n_stance_chg", "n_beige", "n_news", "n_gpr", "n_epu"]
TAYLOR_COLS = ["s_rate_lag", "s_pce_core_yoy", "t_u3"]
BLOCKS = {"S": S_COLS, "M": M_COLS, "N": N_COLS}
YOY = {"pce": "s_pce_yoy", "pce_core": "s_pce_core_yoy", "earn": "s_earn_yoy", "ppi": "s_ppi_yoy",
       "cpi": "s_cpi_yoy", "cpi_core": "s_cpi_core_yoy", "hpi": "s_hpi_yoy"}


def feature_columns(blocks: str) -> list[str]:
    cols = [c for b in blocks for c in BLOCKS[b]]
    if any(c.startswith("y_") for c in cols):
        raise ValueError("target column in feature list")
    return cols


def _last(series: pd.Series) -> float:
    series = series.dropna()
    return float(series.iloc[-1]) if len(series) else np.nan


def macro_at(alfred_obs: dict, cutoff: pd.Timestamp) -> tuple[dict, int]:
    out, fallbacks = {}, 0

    def get_series(name: str) -> pd.Series:
        nonlocal fallbacks
        _, freq, lag = alfred.SERIES[name]
        series, used_fallback = alfred.asof_series(alfred_obs[name], cutoff, freq, lag)
        fallbacks += int(used_fallback)
        return series.asfreq("QS" if freq == "Q" else "MS") if len(series) else series

    m2 = get_series("m2")
    out["s_m2_mom"] = _last(m2.pct_change() * 100)
    out["s_m2_yoy"] = _last(m2.pct_change(12) * 100)
    for name, col in YOY.items():
        out[col] = _last(get_series(name).pct_change(12) * 100)
    out["s_payems_chg"] = _last(get_series("payems").diff())
    out["s_gdp_yoy"] = _last(get_series("gdp").pct_change(4) * 100)
    u6 = get_series("u6")
    out["s_u6_mom"] = _last(u6.diff())
    out["s_u6_yoy"] = _last(u6.diff(12))
    out["t_u3"] = _last(get_series("u3"))
    return out, fallbacks


def market_at(daily: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    v = {col: market.asof(daily[col], cutoff)
         for col in ["target", "tbill_3m", "y6m", "y1", "y2", "y10", "vix", "wti"]}
    return {
        "s_slope_10y2y": v["y10"] - v["y2"], "s_slope_1y6m": v["y1"] - v["y6m"],
        "m_sp_tbill_3m": v["tbill_3m"] - v["target"], "m_sp_y6m": v["y6m"] - v["target"],
        "m_sp_y1": v["y1"] - v["target"], "m_sp_y2": v["y2"] - v["target"],
        "m_target_level": v["target"], "m_zero_bound": int(v["target"] <= 0.30),
        "m_vix": v["vix"], "m_wti": v["wti"],
    }


def _row(cutoff, win_start, y_level, rate_lag, daily, alfred_obs) -> dict:
    macro, fallbacks = macro_at(alfred_obs, cutoff)
    return {"cutoff": cutoff, "win_start": win_start, "y_level": y_level,
            "y_change": y_level - rate_lag, "s_rate_lag": rate_lag,
            **macro, **market_at(daily, cutoff), "fallback_n": fallbacks}


def _classes(change: pd.Series) -> pd.Series:
    return pd.Series(np.where(change > config.CLASS_EPS, 2, np.where(change < -config.CLASS_EPS, 0, 1)),
                     index=change.index)


def build_t1(daily, fedfunds, alfred_obs) -> pd.DataFrame:
    rows = {}
    months = fedfunds.loc[config.SAMPLE_START:].index
    for t in months:
        prev = t - pd.DateOffset(months=1)
        if prev not in fedfunds.index:
            continue
        rows[t] = _row(t, prev, float(fedfunds[t]), float(fedfunds[prev]), daily, alfred_obs)
    frame = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    frame["y_class"] = _classes(frame["y_change"])
    return frame


def build_t2(meet, daily, alfred_obs) -> pd.DataFrame:
    rows, prev_cutoff = {}, None
    for day, m in meet.iterrows():
        win_start = prev_cutoff if prev_cutoff is not None else m["cutoff"] - pd.Timedelta(days=45)
        rows[day] = _row(m["cutoff"], win_start, m["target_after"], m["target_before"], daily, alfred_obs)
        rows[day]["y_class"] = int(m["y_class"])
        rows[day]["unscheduled"] = int(m["unscheduled"])
        prev_cutoff = m["cutoff"]
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def build_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily, fedfunds, alfred_obs = market.load_daily(), market.load_fedfunds(), alfred.load_all()
    t1 = build_t1(daily, fedfunds, alfred_obs)
    t2 = build_t2(load_meetings(), daily, alfred_obs)
    t1.to_pickle(config.DATA / "t1.pkl")
    t2.to_pickle(config.DATA / "t2.pkl")
    return t1, t2


if __name__ == "__main__":
    t1, t2 = build_panels()
    for name, frame in (("T1", t1), ("T2", t2)):
        complete = frame[S_COLS + M_COLS + ["t_u3"]].notna().all(axis=1)
        print(f"{name}: rows {len(frame)}, complete S+M+Taylor {complete.sum()}, first complete "
              f"{frame.index[complete.to_numpy()].min().date()}, fallback rows {(frame.fallback_n > 0).sum()}")
```

- [ ] **Step 4: Run tests, then build the panels**

Run: `$PY -m pytest tests/test_panel.py -v` then `PYTHONPATH=src $PY -m fed_news.panel`
Expected: 4 passed; then two lines. T1 should have about 393 rows with the first complete row in 1995 (U-6 starts January 1994, so its year-on-year change starts January 1995); T2 about 270 rows. Save the printed numbers for `PREREG.md`.

- [ ] **Step 5: Commit**

```bash
git add src/fed_news/panel.py tests/test_panel.py
git commit -m "Add real-time T1 and T2 panels with S, M and Taylor inputs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Stance labels and the per-year ChronoBERT classifiers

**Files:**
- Create: `src/fed_news/labels.py`, `src/fed_news/stance.py`
- Test: `tests/test_labels.py`, `tests/test_stance.py`

**Interfaces:**
- Consumes: `http.get`, `corpus.build_corpus`, `corpus.split_sentences`, `config` stance settings
- Produces: `labels.LABELS = {"dovish": 0, "hawkish": 1, "neutral": 2}`; `labels.load_labels(refresh: bool = False) -> pd.DataFrame` (columns `sentence` lower-cased, `label` int, `year` int, `source`); `stance.STANCE_TYPES = ("statement", "minutes", "presconf", "speech")`; `stance.vintage_year(row_year: int) -> int`; `stance.classifier_year(row_year: int) -> int`; `stance.model_name(vintage: int) -> str`; `stance.eligible(labels: pd.DataFrame, cyear: int) -> pd.DataFrame`; `stance.train_and_score(train: pd.DataFrame, texts: list[str], vintage: int, seed: int) -> np.ndarray` (n x 3 probabilities); `stance.run_stance() -> pd.DataFrame` saving `D:/fed_news/doc_stance.pkl` with columns `doc_id, type, release, stance, n_sent` and `results/stance_accuracy.csv`.

- [ ] **Step 1: Write the failing tests `tests/test_labels.py` and `tests/test_stance.py`**

```python
# tests/test_labels.py
import pytest

from fed_news import labels


@pytest.mark.network
def test_labels_load_and_tdw_mapping_is_hawkish_for_tightening():
    lab = labels.load_labels(refresh=True)
    assert set(lab["label"]) == {0, 1, 2}
    assert lab["sentence"].is_unique
    assert (lab["sentence"] == lab["sentence"].str.lower()).all()
    tdw = lab[lab["source"] == "tdw"]
    assert len(tdw) > 2000
    tight = tdw[tdw["sentence"].str.contains(r"\btighten", regex=True)]
    assert tight["label"].mode().iloc[0] == labels.LABELS["hawkish"]
    assert lab["year"].min() <= 1997
```

```python
# tests/test_stance.py
import pandas as pd
import pytest

from fed_news import stance


def test_vintage_rule():
    assert stance.vintage_year(1994) == 1999
    assert stance.vintage_year(2000) == 1999
    assert stance.vintage_year(2016) == 2015
    assert stance.vintage_year(2026) == 2024


def test_classifier_year_maps_early_rows_to_2000():
    assert stance.classifier_year(1996) == 2000
    assert stance.classifier_year(2013) == 2013


def test_model_name():
    assert stance.model_name(2015) == "manelalab/chrono-bert-v1-20151231"


def test_eligible_is_strictly_before_year():
    lab = pd.DataFrame({"sentence": [f"s{i}" for i in range(120)], "label": 0,
                        "year": [1998] * 60 + [2000] * 60})
    assert set(stance.eligible(lab, 2000)["year"]) == {1998}


def test_eligible_raises_when_too_few():
    lab = pd.DataFrame({"sentence": ["a", "b"], "label": [0, 1], "year": [1998, 1999]})
    with pytest.raises(ValueError, match="labels"):
        stance.eligible(lab, 2000)


@pytest.mark.gpu
def test_train_and_score_smoke():
    train = pd.DataFrame({"sentence": ["the committee decided to raise rates"] * 16
                          + ["the committee decided to lower rates"] * 16,
                          "label": [1] * 16 + [0] * 16})
    probs = stance.train_and_score(train, ["inflation pressures warrant tightening"], 1999, 0)
    assert probs.shape == (1, 3)
    assert abs(probs.sum() - 1.0) < 1e-4
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest tests/test_stance.py -v`
Expected: FAIL with `ImportError: cannot import name 'stance'`

- [ ] **Step 3: Write `src/fed_news/labels.py`**

```python
"""Hawkish / dovish / neutral sentence labels (spec section 5).

Trillion Dollar Words (gtfintechlab/fomc_communication, 2,480 sentences, label 0 dovish,
1 hawkish, 2 neutral, with a year) plus the 1,000 Federal Reserve sentences of the World
Central Banks set (gtfintechlab/federal_reserve_system; the configs 5768/78516/944601 are three
splits of the same sentences, so one config's train+val+test is the full set).
"""
import io

import pandas as pd

from fed_news import config
from fed_news.http import get

LABELS = {"dovish": 0, "hawkish": 1, "neutral": 2}
TDW = "https://huggingface.co/datasets/gtfintechlab/fomc_communication/resolve/main/{}.csv"
WCB = ("https://huggingface.co/datasets/gtfintechlab/federal_reserve_system/resolve/main/"
       "5768/{}-00000-of-00001.parquet")


def load_labels(refresh: bool = False) -> pd.DataFrame:
    path = config.DATA / "labels.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    tdw = pd.concat([pd.read_csv(io.StringIO(get(TDW.format(s)))) for s in ("train", "test")])
    tdw = pd.DataFrame({"sentence": tdw["sentence"], "label": tdw["label"].astype(int),
                        "year": tdw["year"].astype(int), "source": "tdw"})
    wcb = pd.concat([pd.read_parquet(io.BytesIO(get(WCB.format(s), binary=True)))
                     for s in ("train", "val", "test")])
    wcb = wcb[wcb["stance_label"].isin(LABELS)]
    wcb = pd.DataFrame({"sentence": wcb["sentences"], "label": wcb["stance_label"].map(LABELS),
                        "year": wcb["year"].astype(int), "source": "wcb"})
    out = pd.concat([tdw, wcb], ignore_index=True)
    out["sentence"] = out["sentence"].astype(str).str.lower().str.strip()
    out = out.drop_duplicates("sentence").reset_index(drop=True)
    out.to_pickle(path)
    return out
```

- [ ] **Step 4: Write `src/fed_news/stance.py`**

```python
"""Per-year, lookahead-free stance classifiers (spec section 6).

For each classifier year y (2000 onward; documents from 1994-1999 use y = 2000), fine-tune
ChronoBERT vintage y-1 (clipped to 1999-2024) on labels with year < y, three seeds, and score
every sentence of that year's statements, minutes, press conferences and speeches. Sentence
stance = P(hawkish) - P(dovish); document stance = mean over its sentences. Models are not
saved (disk); per-year results are, so an interrupted run resumes.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from fed_news import config
from fed_news.corpus import build_corpus, split_sentences
from fed_news.labels import load_labels

STANCE_TYPES = ("statement", "minutes", "presconf", "speech")


def vintage_year(row_year: int) -> int:
    return min(max(row_year - 1, config.FIRST_VINTAGE), config.LAST_VINTAGE)


def classifier_year(row_year: int) -> int:
    return max(row_year, config.FIRST_CLASSIFIER_YEAR)


def model_name(vintage: int) -> str:
    return f"manelalab/chrono-bert-v1-{vintage}1231"


def eligible(labels: pd.DataFrame, cyear: int) -> pd.DataFrame:
    out = labels[labels["year"] < cyear]
    if len(out) < config.MIN_LABELS:
        raise ValueError(f"only {len(out)} labels before {cyear}; need {config.MIN_LABELS}")
    return out


class StanceNet(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.lm = AutoModel.from_pretrained(name)
        self.head = nn.Linear(self.lm.config.hidden_size, 3)

    def forward(self, ids, mask):
        h = self.lm(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1))


def _batches(tok, texts, size, order):
    for i in range(0, len(order), size):
        j = order[i:i + size]
        enc = tok([texts[k] for k in j], padding=True, truncation=True,
                  max_length=config.STANCE_MAXLEN, return_tensors="pt")
        yield j, enc["input_ids"].cuda(), enc["attention_mask"].cuda()


def train_and_score(train: pd.DataFrame, texts: list[str], vintage: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    name = model_name(vintage)
    tok = AutoTokenizer.from_pretrained(name)
    net = StanceNet(name).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=config.STANCE_LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler()
    sents, y = list(train["sentence"]), torch.tensor(train["label"].to_numpy(), dtype=torch.long)
    for _ in range(config.STANCE_EPOCHS):
        net.train()
        for j, ids, mask in _batches(tok, sents, config.STANCE_BATCH, rng.permutation(len(sents))):
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                logits = net(ids, mask)
            loss = nn.functional.cross_entropy(logits.float(), y[j].cuda())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
    net.eval()
    out = np.zeros((len(texts), 3), dtype=np.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for j, ids, mask in _batches(tok, texts, 128, np.arange(len(texts))):
            out[j] = torch.softmax(net(ids, mask).float(), dim=-1).cpu().numpy()
    del net
    torch.cuda.empty_cache()
    return out


def _sentence_table(docs: pd.DataFrame) -> pd.DataFrame:
    rows = [(d.doc_id, s.lower()) for d in docs.itertuples() for s in split_sentences(d.text)]
    return pd.DataFrame(rows, columns=["doc_id", "sentence"])


def run_stance() -> pd.DataFrame:
    labels, corpus = load_labels(), build_corpus()
    docs = corpus[corpus["type"].isin(STANCE_TYPES)].copy()
    docs["cyear"] = docs["release"].dt.year.map(classifier_year)
    parts_dir = config.DATA / "stance"
    parts_dir.mkdir(exist_ok=True)
    acc_rows, all_sent = [], []
    for cyear in sorted(docs["cyear"].unique()):
        part = parts_dir / f"{cyear}.pkl"
        if part.exists():
            sent, acc = pd.read_pickle(part)
        else:
            pool = eligible(labels, cyear)
            hold = np.random.default_rng(0).random(len(pool)) < config.STANCE_HOLDOUT
            train, holdout = pool[~hold], pool[hold]
            sent = _sentence_table(docs[docs["cyear"] == cyear])
            texts = list(holdout["sentence"]) + list(sent["sentence"])
            probs = np.mean([train_and_score(train, texts, vintage_year(cyear), seed)
                             for seed in config.STANCE_SEEDS], axis=0)
            pred = probs[:len(holdout)].argmax(1)
            truth = holdout["label"].to_numpy()
            f1 = np.mean([2 * ((pred == k) & (truth == k)).sum()
                          / max((pred == k).sum() + (truth == k).sum(), 1) for k in range(3)])
            acc = {"cyear": cyear, "vintage": vintage_year(cyear), "n_train": len(train),
                   "n_holdout": len(holdout), "accuracy": float((pred == truth).mean()),
                   "macro_f1": float(f1)}
            sp = probs[len(holdout):]
            sent = sent.assign(stance=sp[:, 1] - sp[:, 0])
            pd.to_pickle((sent, acc), part)
        acc_rows.append(acc)
        all_sent.append(sent)
        print(acc, flush=True)
    sentences = pd.concat(all_sent, ignore_index=True)
    doc_stance = (sentences.groupby("doc_id")["stance"].agg(["mean", "size"])
                  .rename(columns={"mean": "stance", "size": "n_sent"}).reset_index())
    doc_stance = doc_stance.merge(docs[["doc_id", "type", "release"]], on="doc_id")
    doc_stance.to_pickle(config.DATA / "doc_stance.pkl")
    pd.DataFrame(acc_rows).to_csv(config.RESULTS / "stance_accuracy.csv", index=False)
    return doc_stance


if __name__ == "__main__":
    out = run_stance()
    print(out.groupby(out["release"].dt.year)["stance"].describe().round(3).to_string())
```

- [ ] **Step 5: Run the tests**

Run: `$PY -m pytest tests/test_stance.py -v`, then `$PY -m pytest tests/test_labels.py -v -m network`, then `$PY -m pytest tests/test_stance.py -v -m gpu`
Expected: 5 passed; 1 passed; 1 passed. If the label test's `tighten` check fails, the TDW label mapping is not 0 dovish / 1 hawkish / 2 neutral: read the dataset README at `https://huggingface.co/datasets/gtfintechlab/fomc_communication`, fix the mapping in `load_labels`, and rerun.

- [ ] **Step 6: Run the full stance job in the background (about 2-3 hours on the RTX 2060)**

```bash
PYTHONPATH=src HF_HOME=D:/hf_cache nohup $PY -u -m fed_news.stance > logs/stance.log 2>&1 &
```
Check `logs/stance.log`. Expected: one accuracy line per classifier year 2000-2026, then a per-year stance summary. Accuracy around 0.55-0.70 is normal (the TDW paper's best model is in that range); a year below 0.40 means that classifier learned nothing, which goes in `RESULTS.md` but is not fixed by tuning (settings are fixed).

- [ ] **Step 7: Commit**

```bash
git add src/fed_news/labels.py src/fed_news/stance.py tests/test_labels.py tests/test_stance.py results/stance_accuracy.csv
git commit -m "Add stance labels and per-year lookahead-free ChronoBERT classifiers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: The six news features

**Files:**
- Create: `src/fed_news/text_features.py`
- Test: `tests/test_text_features.py`

**Interfaces:**
- Consumes: `corpus.build_corpus`, `indices.load_indices`, `indices.window_mean`, `panel.N_COLS`, `D:/fed_news/doc_stance.pkl`, `D:/fed_news/t1.pkl`, `t2.pkl`
- Produces: `text_features.GROUP: dict[str, str]`; `text_features.fed_stance(stance: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float`; `text_features.lm_tone(text: str) -> float`; `text_features.beige_tones(corpus: pd.DataFrame) -> pd.Series` (tone indexed by release date); `text_features.build_text_features(rows: pd.DataFrame, stance: pd.DataFrame, beige: pd.Series, idx: pd.DataFrame) -> pd.DataFrame` with exactly `panel.N_COLS`; `text_features.attach_all() -> None` writing `D:/fed_news/t1_full.pkl` and `t2_full.pkl`.

- [ ] **Step 1: Write the failing test `tests/test_text_features.py`**

```python
import numpy as np
import pandas as pd
import pytest

from fed_news import panel, text_features as tf

T = pd.Timestamp


def stance_docs():
    return pd.DataFrame({
        "release": [T("2020-01-05"), T("2020-01-06"), T("2020-01-07"), T("2020-01-10")],
        "type": ["statement", "minutes", "speech", "speech"],
        "stance": [0.6, 0.0, -0.3, 0.9],
    })


def test_fed_stance_is_mean_of_group_means_and_excludes_cutoff_day():
    # window [Jan 5, Jan 10): policy 0.6, minutes 0.0, speech -0.3 -> mean 0.1
    assert tf.fed_stance(stance_docs(), T("2020-01-05"), T("2020-01-10")) == pytest.approx(0.1)


def test_fed_stance_skips_empty_groups():
    assert tf.fed_stance(stance_docs(), T("2020-01-07"), T("2020-01-11")) == pytest.approx(0.3)


def test_empty_window_carries_forward():
    rows = pd.DataFrame({"win_start": [T("2020-01-05"), T("2020-01-11")],
                         "cutoff": [T("2020-01-10"), T("2020-01-20")]},
                        index=[T("2020-01-11"), T("2020-01-21")])
    idx = pd.DataFrame({"news": 0.1, "gpr": 100.0, "epu": 50.0},
                       index=pd.date_range("2019-11-01", "2020-01-31"))
    beige = pd.Series([0.2], index=[T("2019-12-01")])
    out = tf.build_text_features(rows, stance_docs(), beige, idx)
    assert list(out.columns) == panel.N_COLS
    assert out["n_stance"].tolist() == pytest.approx([0.1, 0.1])
    assert out["n_stance_chg"].iloc[1] == pytest.approx(0.0)
    assert out["n_beige"].tolist() == pytest.approx([0.2, 0.2])
    assert out["n_gpr"].iloc[0] == pytest.approx(np.log(100.0))


def test_beige_is_latest_strictly_before_cutoff():
    rows = pd.DataFrame({"win_start": [T("2020-01-01")], "cutoff": [T("2020-01-15")]}, index=[T("2020-01-16")])
    idx = pd.DataFrame({"news": 0.0, "gpr": 1.0, "epu": 1.0}, index=pd.date_range("2019-11-01", "2020-01-31"))
    beige = pd.Series([0.2, 0.9], index=[T("2019-12-01"), T("2020-01-15")])
    out = tf.build_text_features(rows, stance_docs(), beige, idx)
    assert out["n_beige"].iloc[0] == pytest.approx(0.2)


def test_lm_tone_sign():
    assert tf.lm_tone("Growth was strong and conditions improved, a positive gain.") > 0
    assert tf.lm_tone("Losses worsened and conditions were weak, with severe declines.") < 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/test_text_features.py -v`
Expected: FAIL with `ImportError: cannot import name 'text_features'`

- [ ] **Step 3: Write `src/fed_news/text_features.py`**

```python
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
```

- [ ] **Step 4: Run tests and attach the features**

Run: `$PY -m pytest tests/test_text_features.py -v` then `PYTHONPATH=src $PY -m fed_news.text_features`
Expected: 5 passed; then per-column counts for T1 and T2 close to the panel row counts.

- [ ] **Step 5: Commit**

```bash
git add src/fed_news/text_features.py tests/test_text_features.py
git commit -m "Add the six news features

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Models and statistical tests

**Files:**
- Create: `src/fed_news/models.py`, `src/fed_news/stats_tests.py`
- Test: `tests/test_models.py`, `tests/test_stats_tests.py`

**Interfaces:**
- Produces: `models.OrderedLogit(lam: float)` with `.fit(X: np.ndarray, y: np.ndarray) -> OrderedLogit` and `.predict_proba(X) -> np.ndarray` (n x 3), attribute `beta_`; `models.expanding_folds(n: int, k: int = CV_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]`; `models.choose_penalty(make, score, X, y, grid) -> float`; `models.ridge_score(model, X, y) -> float` (MSE); `models.logloss_score(model, X, y) -> float`; `models.move_sizes(y_class: np.ndarray, y_change: np.ndarray) -> np.ndarray` (length 3); `models.classes_from_change(change: np.ndarray, band: float = CLASS_BAND) -> np.ndarray`; `models.taylor_predict(X_train, y_train, x_test) -> float`; `stats_tests.dm_hln(loss_base, loss_new, h: int = 1) -> tuple[float, float]` (one-sided p that the new model has lower loss); `stats_tests.mcnemar_greater(correct_base, correct_new) -> tuple[int, int, float]` (fixes, breaks, one-sided p); `stats_tests.holm(pvalues: dict[str, float]) -> dict[str, float]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models.py
import numpy as np
import pytest
from scipy.optimize import check_grad
from statsmodels.miscmodels.ordinal_model import OrderedModel

from fed_news import models


def simulate(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    eta = X @ np.array([1.0, -0.5]) + rng.logistic(size=n)
    y = np.digitize(eta, [-1.0, 1.0])
    return X, y


def test_ordered_logit_matches_statsmodels_without_penalty():
    X, y = simulate()
    ours = models.OrderedLogit(0.0).fit(X, y)
    ref = OrderedModel(y, X, distr="logit").fit(method="bfgs", disp=False)
    assert np.allclose(ours.beta_, ref.params[:2], atol=5e-3)
    assert np.allclose(ours.predict_proba(X).sum(1), 1.0)


def test_ordered_logit_gradient_is_correct():
    X, y = simulate(100)
    model = models.OrderedLogit(0.3)
    w = np.array([0.2, -0.1, -0.5, 0.3])
    assert check_grad(lambda v: model._loss(v, X, y)[0], lambda v: model._loss(v, X, y)[1], w) < 1e-5


def test_penalty_shrinks_coefficients():
    X, y = simulate()
    weak = models.OrderedLogit(0.0).fit(X, y).beta_
    strong = models.OrderedLogit(3.0).fit(X, y).beta_
    assert np.abs(strong).sum() < np.abs(weak).sum()


def test_expanding_folds_are_chronological():
    folds = models.expanding_folds(60, 5)
    assert len(folds) == 5
    for train, val in folds:
        assert train.max() < val.min() and train.min() == 0


def test_choose_penalty_prefers_strong_penalty_on_noise():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(200, 20)), rng.integers(0, 3, 200)
    lam = models.choose_penalty(models.OrderedLogit, models.logloss_score, X, y, (0.01, 3.0))
    assert lam == 3.0


def test_move_sizes_and_class_band():
    sizes = models.move_sizes(np.array([0, 0, 1, 2]), np.array([-0.25, -0.75, 0.0, 0.25]))
    assert sizes.tolist() == pytest.approx([-0.5, 0.0, 0.25])
    assert models.move_sizes(np.array([1, 2]), np.array([0.0, 0.25]))[0] == pytest.approx(-0.25)
    assert models.classes_from_change(np.array([-0.2, 0.1, 0.13])).tolist() == [0, 1, 2]


def test_taylor_predict_recovers_linear_rule():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 3))
    y = 0.5 + X @ np.array([0.9, 0.3, -0.2])
    assert models.taylor_predict(X, y, np.array([1.0, 1.0, 1.0])) == pytest.approx(1.5)
```

```python
# tests/test_stats_tests.py
import numpy as np
import pytest

from fed_news import stats_tests as st


def test_dm_hln_hand_computed():
    base = np.array([1.0, 2.0, 3.0, 4.0])
    new = np.array([0.5, 1.5, 2.0, 4.0])
    # d = [0.5, 0.5, 1.0, 0.0], mean 0.5, var (population) 0.125, DM = 0.5/sqrt(0.125/4) = 2.828
    # HLN factor sqrt((n - 1)/n) = sqrt(0.75); stat = 2.449; one-sided t(3) p = 0.0459
    stat, p = st.dm_hln(base, new)
    assert stat == pytest.approx(2.4495, abs=1e-3)
    assert p == pytest.approx(0.0459, abs=1e-3)


def test_dm_hln_identical_losses_is_p_one_half_or_more():
    x = np.ones(10)
    assert st.dm_hln(x, x)[1] >= 0.5


def test_mcnemar_exact_one_sided():
    base = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    new = np.array([1, 1, 1, 1, 1, 1, 1, 0])
    fixes, breaks, p = st.mcnemar_greater(base.astype(bool), new.astype(bool))
    assert (fixes, breaks) == (6, 1)
    assert p == pytest.approx(8 / 128)


def test_holm():
    adj = st.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest tests/test_models.py tests/test_stats_tests.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write `src/fed_news/models.py`**

```python
"""Primary models (spec section 7): penalised ordered logit (T2), ridge on the change (T1),
expanding-window penalty choice, move sizes, the class band, and the inertial Taylor rule."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from fed_news import config


class OrderedLogit:
    """Cumulative logit for cut < hold < hike. Thresholds t1 = a, t2 = a + exp(b).
    Loss = mean negative log-likelihood + lam/2 * ||beta||^2 (slopes only)."""

    def __init__(self, lam: float):
        self.lam = lam

    @staticmethod
    def _probs(X, beta, t1, t2):
        eta = X @ beta
        c1, c2 = expit(t1 - eta), expit(t2 - eta)
        return np.column_stack([c1, c2 - c1, 1.0 - c2])

    def _loss(self, w, X, y):
        p = X.shape[1]
        beta, a, b = w[:p], w[p], w[p + 1]
        t1, t2 = a, a + np.exp(b)
        eta = X @ beta
        c1, c2 = expit(t1 - eta), expit(t2 - eta)
        g1, g2 = c1 * (1 - c1), c2 * (1 - c2)
        prob = np.clip(np.select([y == 0, y == 1], [c1, c2 - c1], 1.0 - c2), 1e-12, None)
        d_eta = np.select([y == 0, y == 1], [-g1, g1 - g2], g2) / prob
        d_t1 = np.select([y == 0, y == 1], [g1, -g1], 0.0) / prob
        d_t2 = np.select([y == 0, y == 1], [0.0, g2], -g2) / prob
        n = len(y)
        loss = -np.mean(np.log(prob)) + 0.5 * self.lam * beta @ beta
        grad = np.empty_like(w)
        grad[:p] = -(X.T @ d_eta) / n + self.lam * beta
        grad[p] = -np.mean(d_t1 + d_t2)
        grad[p + 1] = -np.mean(d_t2) * np.exp(b)
        return loss, grad

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        p = X.shape[1]
        frac = np.clip(np.cumsum(np.bincount(y, minlength=3) / len(y))[:2], 1e-3, 1 - 1e-3)
        gap = max(logit(frac[1]) - logit(frac[0]), 1e-3)
        w0 = np.concatenate([np.zeros(p), [logit(frac[0]), np.log(gap)]])
        res = minimize(self._loss, w0, args=(X, y), jac=True, method="L-BFGS-B")
        self.beta_ = res.x[:p]
        self.t1_, self.t2_ = res.x[p], res.x[p] + np.exp(res.x[p + 1])
        return self

    def predict_proba(self, X):
        return self._probs(X, self.beta_, self.t1_, self.t2_)


def expanding_folds(n: int, k: int = config.CV_FOLDS):
    edges = np.linspace(0, n, k + 2).astype(int)
    return [(np.arange(0, edges[i]), np.arange(edges[i], edges[i + 1])) for i in range(1, k + 1)]


def ridge_score(model, X, y) -> float:
    return float(np.mean((model.predict(X) - y) ** 2))


def logloss_score(model, X, y) -> float:
    proba = np.clip(model.predict_proba(X), 1e-12, None)
    return float(-np.mean(np.log(proba[np.arange(len(y)), np.asarray(y, dtype=int)])))


def choose_penalty(make, score, X, y, grid) -> float:
    best, best_score = grid[0], np.inf
    for value in grid:
        s = np.mean([score(make(value).fit(X[tr], y[tr]), X[va], y[va])
                     for tr, va in expanding_folds(len(y))])
        if s < best_score:
            best, best_score = value, s
    return best


def move_sizes(y_class: np.ndarray, y_change: np.ndarray) -> np.ndarray:
    sizes = np.array([-0.25, 0.0, 0.25])
    for k in (0, 2):
        if (y_class == k).any():
            sizes[k] = float(np.mean(y_change[y_class == k]))
    return sizes


def classes_from_change(change: np.ndarray, band: float = config.CLASS_BAND) -> np.ndarray:
    change = np.asarray(change, dtype=float)
    return np.where(change > band, 2, np.where(change < -band, 0, 1))


def taylor_predict(X_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(X_train)), X_train])
    coef = np.linalg.lstsq(design, y_train, rcond=None)[0]
    return float(coef[0] + x_test @ coef[1:])
```

- [ ] **Step 4: Write `src/fed_news/stats_tests.py`**

```python
"""Pre-registered tests (spec section 8)."""
import numpy as np
from scipy import stats


def dm_hln(loss_base, loss_new, h: int = 1) -> tuple[float, float]:
    """Harvey-Leybourne-Newbold corrected Diebold-Mariano; one-sided p that loss_new < loss_base."""
    d = np.asarray(loss_base, float) - np.asarray(loss_new, float)
    n = len(d)
    centred = d - d.mean()
    var = np.mean(centred ** 2) + 2 * sum(np.mean(centred[k:] * centred[:-k]) for k in range(1, h))
    if var <= 0:
        return 0.0, 1.0 if d.mean() <= 0 else 0.0
    dm = d.mean() / np.sqrt(var / n)
    stat = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(stat), float(1 - stats.t.cdf(stat, df=n - 1))


def mcnemar_greater(correct_base, correct_new) -> tuple[int, int, float]:
    base, new = np.asarray(correct_base, bool), np.asarray(correct_new, bool)
    fixes, breaks = int((~base & new).sum()), int((base & ~new).sum())
    if fixes + breaks == 0:
        return 0, 0, 1.0
    return fixes, breaks, float(stats.binomtest(fixes, fixes + breaks, 0.5, alternative="greater").pvalue)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    order = sorted(pvalues, key=pvalues.get)
    m, running, out = len(order), 0.0, {}
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[key]))
        out[key] = running
    return out
```

The identical-loss case returns `(0.0, 1.0)` because the variance is zero and the mean difference is not positive, which satisfies the `>= 0.5` test.

- [ ] **Step 5: Run the tests**

Run: `$PY -m pytest tests/test_models.py tests/test_stats_tests.py -v`
Expected: 11 passed

- [ ] **Step 6: Commit**

```bash
git add src/fed_news/models.py src/fed_news/stats_tests.py tests/test_models.py tests/test_stats_tests.py
git commit -m "Add ordered logit, penalty CV, Taylor rule and pre-registered tests

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Walk-forward engine, ladder and the pre-registration guard

**Files:**
- Create: `src/fed_news/walkforward.py`
- Test: `tests/test_walkforward.py`

**Interfaces:**
- Consumes: `models.*`, `panel.S_COLS`, `panel.M_COLS`, `panel.N_COLS`, `panel.TAYLOR_COLS`, `config`
- Produces: `walkforward.Spec` dataclass (`name: str, task: str, kind: str, cols: list[str], placebo: bool = False`), `kind` in `{"rw", "taylor", "primary"}`; `walkforward.ladder(task: str) -> list[Spec]` with names `rw, taylor, S, S+N, placebo, S+M, S+M+N`; `walkforward.common_rows(df: pd.DataFrame) -> pd.DataFrame`; `walkforward.train_mask(n: int, i: int, task: str) -> np.ndarray`; `walkforward.placebo_matrix(X, train, i, ncols, rng) -> np.ndarray`; `walkforward.prereg_ok(repo=config.REPO) -> bool`; `walkforward.run_spec(df: pd.DataFrame, spec: Spec, test_start: str, test_end: str | None = None, seed: int = 0, require_prereg: bool = True) -> pd.DataFrame` with columns `date, y_true, y_pred, lag, class_true, class_pred`.

- [ ] **Step 1: Write the failing test `tests/test_walkforward.py`**

```python
import subprocess

import numpy as np
import pandas as pd
import pytest

from fed_news import panel, walkforward as wf


def synthetic(n=80, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1995-01-01", periods=n, freq="MS")
    df = pd.DataFrame(rng.normal(size=(n, len(panel.S_COLS))), index=idx, columns=panel.S_COLS)
    for c in panel.M_COLS + panel.N_COLS:
        df[c] = rng.normal(size=n)
    df["t_u3"] = rng.normal(size=n)
    df["s_rate_lag"] = 2.0
    change = 0.25 * np.sign(df["n_stance"].to_numpy()) * (np.abs(df["n_stance"].to_numpy()) > 0.5)
    df["y_change"], df["y_level"] = change, 2.0 + change
    df["y_class"] = np.where(change > 0.02, 2, np.where(change < -0.02, 0, 1))
    return df


def test_train_mask_t1_ends_two_rows_back_and_t2_one():
    assert np.flatnonzero(wf.train_mask(10, 6, "T1")).tolist() == [0, 1, 2, 3, 4]
    assert np.flatnonzero(wf.train_mask(10, 6, "T2")).tolist() == [0, 1, 2, 3, 4, 5]


def test_common_rows_drops_any_missing():
    df = synthetic()
    df.iloc[3, df.columns.get_loc("n_epu")] = np.nan
    df.iloc[5, df.columns.get_loc("t_u3")] = np.nan
    out = wf.common_rows(df)
    assert len(out) == len(df) - 2 and df.index[3] not in out.index


def test_placebo_touches_only_n_columns_and_uses_training_rows():
    X = np.arange(20.0).reshape(5, 4)
    train = np.array([True, True, True, False, False])
    out = wf.placebo_matrix(X, train, 3, [2, 3], np.random.default_rng(0))
    assert np.array_equal(out[:, :2], X[:, :2])
    assert sorted(out[:3, 2].tolist()) == sorted(X[:3, 2].tolist())
    assert out[3, 2] in X[:3, 2]
    assert np.array_equal(out[4], X[4])


def test_ladder_names_and_blocks():
    names = [s.name for s in wf.ladder("T2")]
    assert names == ["rw", "taylor", "S", "S+N", "placebo", "S+M", "S+M+N"]
    spec = {s.name: s for s in wf.ladder("T2")}
    assert spec["S+N"].cols == panel.S_COLS + panel.N_COLS and spec["placebo"].placebo


def test_text_spec_refused_without_prereg(monkeypatch):
    monkeypatch.setattr(wf, "prereg_ok", lambda repo=None: False)
    spec = {s.name: s for s in wf.ladder("T2")}["S+N"]
    with pytest.raises(PermissionError):
        wf.run_spec(synthetic(), spec, "2000-01-01")


def test_primary_t2_learns_signal_and_rw_holds():
    df = synthetic(120)
    spec = {s.name: s for s in wf.ladder("T2")}
    sn = wf.run_spec(df, spec["S+N"], "2002-01-01", require_prereg=False)
    rw = wf.run_spec(df, spec["rw"], "2002-01-01", require_prereg=False)
    assert (sn["class_pred"] == sn["class_true"]).mean() > 0.6
    assert (rw["class_pred"] == 1).all() and (rw["y_pred"] == 2.0).all()


def test_prereg_ok_reads_git_tag(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                       check=True, capture_output=True)
    git("init", "-q")
    git("commit", "-q", "--allow-empty", "-m", "x")
    assert wf.prereg_ok(tmp_path) is False
    git("tag", "prereg")
    assert wf.prereg_ok(tmp_path) is True
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/test_walkforward.py -v`
Expected: FAIL with `ImportError: cannot import name 'walkforward'`

- [ ] **Step 3: Write `src/fed_news/walkforward.py`**

```python
"""Walk-forward engine (spec sections 7-8).

Each test row is forecast from a model fit only on earlier rows: T1 on rows up to t-2 (the
paper's protocol), T2 on all earlier meetings. Features are standardised on the training rows
and penalties chosen by expanding-window CV inside them. Any spec that uses an N column on a
test window is refused unless the git tag `prereg` exists.
"""
import subprocess
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from fed_news import config, models
from fed_news.panel import M_COLS, N_COLS, S_COLS, TAYLOR_COLS


@dataclass
class Spec:
    name: str
    task: str
    kind: str
    cols: list[str] = field(default_factory=list)
    placebo: bool = False


def ladder(task: str) -> list[Spec]:
    return [
        Spec("rw", task, "rw"),
        Spec("taylor", task, "taylor", TAYLOR_COLS),
        Spec("S", task, "primary", S_COLS),
        Spec("S+N", task, "primary", S_COLS + N_COLS),
        Spec("placebo", task, "primary", S_COLS + N_COLS, placebo=True),
        Spec("S+M", task, "primary", S_COLS + M_COLS),
        Spec("S+M+N", task, "primary", S_COLS + M_COLS + N_COLS),
    ]


def common_rows(df: pd.DataFrame) -> pd.DataFrame:
    needed = list(dict.fromkeys(S_COLS + M_COLS + N_COLS + TAYLOR_COLS + ["y_level", "y_change", "y_class"]))
    return df[df[needed].notna().all(axis=1)]


def train_mask(n: int, i: int, task: str) -> np.ndarray:
    idx = np.arange(n)
    return idx <= i - 2 if task == "T1" else idx < i


def placebo_matrix(X, train, i, ncols, rng) -> np.ndarray:
    X = X.copy()
    rows = np.flatnonzero(train)
    X[np.ix_(rows, ncols)] = X[np.ix_(rng.permutation(rows), ncols)]
    X[i, ncols] = X[rng.choice(rows), ncols]
    return X


def prereg_ok(repo=config.REPO) -> bool:
    out = subprocess.run(["git", "-C", str(repo), "tag", "-l", config.PREREG_TAG],
                         capture_output=True, text=True)
    return out.stdout.strip() == config.PREREG_TAG


def _standardise(X, train):
    mu, sd = X[train].mean(0), X[train].std(0)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def run_spec(df: pd.DataFrame, spec: Spec, test_start: str, test_end: str | None = None,
             seed: int = 0, require_prereg: bool = True) -> pd.DataFrame:
    if require_prereg and any(c in N_COLS for c in spec.cols) and not prereg_ok():
        raise PermissionError(f"{spec.name} uses news columns; tag '{config.PREREG_TAG}' first")
    dates = df.index
    test = np.flatnonzero((dates >= pd.Timestamp(test_start))
                          & (dates <= pd.Timestamp(test_end) if test_end else True))
    y_change, y_class = df["y_change"].to_numpy(float), df["y_class"].to_numpy(int)
    lag, y_level = df["s_rate_lag"].to_numpy(float), df["y_level"].to_numpy(float)
    X = df[spec.cols].to_numpy(float) if spec.cols else None
    ncols = [spec.cols.index(c) for c in spec.cols if c in N_COLS]
    rng = np.random.default_rng(seed)
    out = []
    for i in test:
        train = train_mask(len(df), i, spec.task)
        cls = None
        if spec.kind == "rw":
            change = 0.0
        elif spec.kind == "taylor":
            change = models.taylor_predict(X[train], y_level[train], X[i]) - lag[i]
        else:
            Xi = placebo_matrix(X, train, i, ncols, rng) if spec.placebo else X
            Z = _standardise(Xi, train)
            if spec.task == "T1":
                alpha = models.choose_penalty(lambda a: Ridge(alpha=a), models.ridge_score,
                                              Z[train], y_change[train], config.RIDGE_GRID)
                change = float(Ridge(alpha=alpha).fit(Z[train], y_change[train]).predict(Z[i:i + 1])[0])
            else:
                lam = models.choose_penalty(models.OrderedLogit, models.logloss_score,
                                            Z[train], y_class[train], config.ORDLOGIT_GRID)
                proba = models.OrderedLogit(lam).fit(Z[train], y_class[train]).predict_proba(Z[i:i + 1])[0]
                change = float(proba @ models.move_sizes(y_class[train], y_change[train]))
                cls = int(proba.argmax())
        if cls is None:
            cls = int(models.classes_from_change(np.array([change]))[0])
        out.append({"date": dates[i], "y_true": y_level[i], "y_pred": lag[i] + change, "lag": lag[i],
                    "class_true": int(y_class[i]), "class_pred": cls})
    return pd.DataFrame(out)
```

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_walkforward.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/fed_news/walkforward.py tests/test_walkforward.py
git commit -m "Add walk-forward engine, ladder, placebo and pre-registration guard

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Statistics-only dry run, then pre-registration

**Files:**
- Create: `src/fed_news/run_models.py`, `PREREG.md`
- Test: `tests/test_run_models.py`

**Interfaces:**
- Consumes: `walkforward.ladder`, `walkforward.run_spec`, `walkforward.common_rows`, `D:/fed_news/t1_full.pkl`, `t2_full.pkl`
- Produces: `run_models.load_task(task: str) -> pd.DataFrame` (common rows only); `run_models.main(argv: list[str] | None = None) -> None` writing `results/preds_{task}_{name}.csv`; flag `--no-text` runs only specs without N columns.

- [ ] **Step 1: Write the failing test `tests/test_run_models.py`**

```python
from fed_news import run_models, walkforward as wf


def test_no_text_selection_excludes_every_news_spec():
    names = [s.name for s in run_models.select_specs("T2", no_text=True)]
    assert names == ["rw", "taylor", "S", "S+M"]


def test_full_selection_is_the_whole_ladder():
    assert [s.name for s in run_models.select_specs("T1", no_text=False)] == [s.name for s in wf.ladder("T1")]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/test_run_models.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_models'`

- [ ] **Step 3: Write `src/fed_news/run_models.py`**

```python
"""Run the Plan 1 ladder on both tasks and write one prediction file per row."""
import argparse
import time

import pandas as pd

from fed_news import config
from fed_news.panel import N_COLS
from fed_news.walkforward import common_rows, ladder, run_spec


def load_task(task: str) -> pd.DataFrame:
    return common_rows(pd.read_pickle(config.DATA / f"{task.lower()}_full.pkl"))


def select_specs(task: str, no_text: bool):
    specs = ladder(task)
    return [s for s in specs if not any(c in N_COLS for c in s.cols)] if no_text else specs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-text", action="store_true")
    args = ap.parse_args(argv)
    config.RESULTS.mkdir(exist_ok=True)
    for task in ("T1", "T2"):
        df = load_task(task)
        print(f"{task}: {len(df)} common rows, {df.index.min().date()} .. {df.index.max().date()}", flush=True)
        for spec in select_specs(task, args.no_text):
            t0 = time.time()
            preds = run_spec(df, spec, config.TEST_START)
            preds.to_csv(config.RESULTS / f"preds_{task}_{spec.name}.csv", index=False)
            print(f"  {spec.name:8s} {len(preds)} forecasts ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, then the statistics-only dry run on the real data**

```bash
$PY -m pytest tests/test_run_models.py -v
PYTHONPATH=src $PY -m fed_news.run_models --no-text
```
Expected: 2 passed; then T1 about 308 forecasts and T2 about 210 forecasts for each of `rw, taylor, S, S+M`. This checks the pipeline end to end. It makes no news forecast, so it does not touch H1. Do not read accuracy numbers from these files yet beyond checking they are finite.

- [ ] **Step 5: Write `PREREG.md`**

Fill every bracketed value from this session's outputs (Task 3 Step 6, Task 4 Step 8, Task 5 Step 5, Task 7 Step 4, Task 8 Step 6, Task 12 Step 4), then remove the brackets. The file must contain no bracket placeholders when committed.

```markdown
# Pre-registration: news + statistics forecasts of the federal funds rate

Date: [today's date]. Code commit: [output of `git rev-parse HEAD` before this commit].
Design: docs/superpowers/specs/2026-10-08-fed-news-design.md (all sections binding).

## Data as built
- Meetings: [count] from [first date] to [last date]; unscheduled: [list].
- T1 common rows: [count], first [date]; T2 common rows: [count], first [date].
- ALFRED first vintages: [one line per series]; rows using the fallback: T1 [n], T2 [n].
- Corpus documents by type: [counts]; dropped: [dropped_docs.json contents].
- Stance classifier held-out accuracy by year: see results/stance_accuracy.csv (min [x], median [y]).

## Primary hypothesis H1 and win rule
S = the 16 columns in panel.S_COLS. N = n_stance, n_stance_chg, n_beige, n_news, n_gpr, n_epu.
Test windows: T1 2001-01 to [last month]; T2 meetings 2001-01 to [last meeting].
H1 holds only if all three are one-sided p < 0.05:
1. T1: S+N beats S on squared error, DM-HLN, h = 1.
2. T2: S+N beats S on squared rate error, DM-HLN, h = 1.
3. T2: S+N beats S on class correctness, exact McNemar.
Void if the placebo beats S at p < 0.05 on any of the three. If H1 fails it is reported as a
null; target, windows and features are not changed afterwards.

## Fixed settings
Ordered logit L2 grid {0.01, 0.03, 0.1, 0.3, 1, 3}; ridge grid 7 log-spaced 0.1-1000; 5-fold
expanding inner CV; T1 trains on rows up to t-2, T2 on earlier meetings; class band ±0.125.

## Secondary hypotheses (Plans 2 and 3)
H2a-H2g exactly as in spec sections 3, 7 and 9, with Holm adjustment across the H2 family.
They cannot rescue a failed H1.
```

- [ ] **Step 6: Commit and tag**

```bash
git add src/fed_news/run_models.py tests/test_run_models.py PREREG.md
git commit -m "Pre-register H1: frozen data, settings and win rule

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git tag prereg
git tag -l prereg
```
Expected: `prereg`. From here on, nothing in `src/` that affects H1 may change; a bug fix after the tag is recorded in `RESULTS.md` under "Deviations" with its commit hash.

---

### Task 13: Full ladder, evaluation and RESULTS.md

**Files:**
- Create: `src/fed_news/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `results/preds_{task}_{name}.csv`, `stats_tests.*`, `config.PAPER_T1`, `config.PAPER_T2`, `results/stance_accuracy.csv`, `results/coverage.csv`
- Produces: `evaluate.metrics(preds: pd.DataFrame) -> dict` (mse, rmse, mae, mbe, accuracy, macro_f1, n); `evaluate.decide(main_p: list[float], placebo_p: list[float], alpha: float = ALPHA) -> str` returning `"supported"`, `"not supported"` or `"void"`; `evaluate.main() -> None` writing `RESULTS.md`, `results/metrics.csv`, `results/h1_tests.csv`, `results/h2_pvalues.csv`.

- [ ] **Step 1: Write the failing test `tests/test_evaluate.py`**

```python
import pandas as pd
import pytest

from fed_news import evaluate


def test_metrics_on_toy_predictions():
    preds = pd.DataFrame({"y_true": [1.0, 2.0], "y_pred": [1.5, 2.0],
                          "class_true": [2, 1], "class_pred": [2, 0]})
    m = evaluate.metrics(preds)
    assert m["mse"] == pytest.approx(0.125) and m["mae"] == pytest.approx(0.25)
    assert m["mbe"] == pytest.approx(0.25) and m["accuracy"] == pytest.approx(0.5)


def test_decide_requires_all_three():
    assert evaluate.decide([0.01, 0.02, 0.03], [0.5, 0.6, 0.7]) == "supported"
    assert evaluate.decide([0.01, 0.20, 0.03], [0.5, 0.6, 0.7]) == "not supported"


def test_decide_void_when_placebo_passes_any():
    assert evaluate.decide([0.01, 0.02, 0.03], [0.5, 0.04, 0.7]) == "void"
    assert evaluate.decide([0.30, 0.02, 0.03], [0.5, 0.04, 0.7]) == "void"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/test_evaluate.py -v`
Expected: FAIL with `ImportError: cannot import name 'evaluate'`

- [ ] **Step 3: Write `src/fed_news/evaluate.py`**

```python
"""Metrics, the pre-registered H1 decision, and RESULTS.md (spec section 8)."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from fed_news import config
from fed_news.stats_tests import dm_hln, holm, mcnemar_greater

PAPER_ROWS = [  # spec ladder row 1: the paper's own reported numbers
    ("paper LSTM (Table 2)", 0.2376 ** 2, 0.2376, 0.1495, -0.0006),
    ("paper GRU (Table 2)", 0.2705 ** 2, 0.2705, 0.1327, 0.0091),
    ("paper LLM, lookahead (Table 3)", 0.0311, 0.1762, 0.1483, np.nan),
]
NAMES = ["rw", "taylor", "S", "S+N", "placebo", "S+M", "S+M+N"]


def metrics(preds: pd.DataFrame) -> dict:
    err = preds["y_pred"] - preds["y_true"]
    return {"n": len(preds), "mse": float(np.mean(err ** 2)), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))), "mbe": float(np.mean(err)),
            "accuracy": float((preds["class_pred"] == preds["class_true"]).mean()),
            "macro_f1": float(f1_score(preds["class_true"], preds["class_pred"], average="macro"))}


def decide(main_p, placebo_p, alpha: float = config.ALPHA) -> str:
    if any(p < alpha for p in placebo_p):
        return "void"
    return "supported" if all(p < alpha for p in main_p) else "not supported"


def _load(task):
    return {n: pd.read_csv(config.RESULTS / f"preds_{task}_{n}.csv", parse_dates=["date"]) for n in NAMES}


def _window(p, window):
    return p[(p["date"] >= window[0]) & (p["date"] <= window[1])] if window else p


def _sq(p):
    return ((p["y_pred"] - p["y_true"]) ** 2).to_numpy()


def _correct(p):
    return (p["class_pred"] == p["class_true"]).to_numpy()


def _three_tests(t1, t2, new):
    p1 = dm_hln(_sq(t1["S"]), _sq(t1[new]))[1]
    p2 = dm_hln(_sq(t2["S"]), _sq(t2[new]))[1]
    p3 = mcnemar_greater(_correct(t2["S"]), _correct(t2[new]))[2]
    return [p1, p2, p3]


def main() -> None:
    t1, t2 = _load("T1"), _load("T2")
    rows = []
    for task, preds, paper in (("T1", t1, config.PAPER_T1), ("T2", t2, config.PAPER_T2)):
        for window_name, window in (("primary", None), ("paper window", paper)):
            for name in NAMES:
                rows.append({"task": task, "window": window_name, "model": name,
                             **metrics(_window(preds[name], window))})
    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS / "metrics.csv", index=False)

    main_p, placebo_p = _three_tests(t1, t2, "S+N"), _three_tests(t1, t2, "placebo")
    verdict = decide(main_p, placebo_p)
    tests = pd.DataFrame({"test": ["T1 DM-HLN", "T2 DM-HLN", "T2 McNemar"],
                          "S+N vs S": main_p, "placebo vs S": placebo_p})
    tests.to_csv(config.RESULTS / "h1_tests.csv", index=False)

    h2 = {"H2c T1 S+N vs rw": dm_hln(_sq(t1["rw"]), _sq(t1["S+N"]))[1],
          "H2c T2 S+N vs rw": dm_hln(_sq(t2["rw"]), _sq(t2["S+N"]))[1],
          "H2d T1 S+M+N vs S+M": dm_hln(_sq(t1["S+M"]), _sq(t1["S+M+N"]))[1],
          "H2d T2 S+M+N vs S+M": dm_hln(_sq(t2["S+M"]), _sq(t2["S+M+N"]))[1],
          "H2d T2 McNemar S+M+N vs S+M": mcnemar_greater(_correct(t2["S+M"]), _correct(t2["S+M+N"]))[2]}
    h2_frame = pd.DataFrame({"hypothesis": list(h2), "p": list(h2.values()),
                             "holm_within_plan1": [holm(h2)[k] for k in h2]})
    h2_frame.to_csv(config.RESULTS / "h2_pvalues.csv", index=False)

    paper = pd.DataFrame(PAPER_ROWS, columns=["model", "mse", "rmse", "mae", "mbe"])
    stance_acc = pd.read_csv(config.RESULTS / "stance_accuracy.csv")
    lines = [
        "# Results: Plan 1 (H1)", "",
        f"**H1 verdict: {verdict}.**", "",
        "## Pre-registered tests (one-sided p)", "", tests.to_markdown(index=False, floatfmt=".4f"), "",
        "## Metrics", "", table.to_markdown(index=False, floatfmt=".4f"), "",
        "## The paper's reported numbers (T1 window Jan 2016 - Apr 2025; LLM has lookahead)", "",
        paper.to_markdown(index=False, floatfmt=".4f"), "",
        "## Secondary (Holm within Plan 1 only; final Holm after Plans 2-3)", "",
        h2_frame.to_markdown(index=False, floatfmt=".4f"), "",
        "## Stance classifier held-out accuracy", "", stance_acc.to_markdown(index=False, floatfmt=".3f"), "",
        "## Deviations", "", "None recorded.", "",
    ]
    (config.REPO / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:8]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_evaluate.py -v`
Expected: 3 passed

- [ ] **Step 5: Run the full ladder once (after the `prereg` tag), then evaluate**

```bash
git tag -l prereg   # must print prereg
PYTHONPATH=src nohup $PY -u -m fed_news.run_models > logs/run_models.log 2>&1 &
# when logs/run_models.log shows all 14 rows finished:
PYTHONPATH=src $PY -m fed_news.evaluate
```
Expected: `RESULTS.md` starts with `**H1 verdict: supported.**`, `not supported.` or `void.` Whatever it says is the result. Do not rerun with changed settings.

- [ ] **Step 6: Run the whole offline test suite, then commit the results**

```bash
$PY -m pytest -v
git add src/fed_news/evaluate.py tests/test_evaluate.py RESULTS.md results/metrics.csv results/h1_tests.csv results/h2_pvalues.csv results/preds_*.csv
git commit -m "Run the pre-registered H1 ladder and record results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
Expected: all offline tests pass; the commit records the verdict.

---

## Self-review notes (done while writing)

- Spec coverage: sections 4 (tasks, windows, cutoffs: Tasks 4, 7, 11), 5 (S: Task 3/7; M: Task 7; N sources: Tasks 5, 6, 8), 6 (vintage rule, classifier, F1-F6: Tasks 8, 9), 7 (primary models, rows 0, 3, 4, 5, 8, 9: Tasks 10-12), 8 (win, void, null rules: Tasks 12-13), 11 (coverage stop, leak tests, y_ guard, ordered logit vs statsmodels, DM/McNemar/Holm values: Tasks 5, 7, 10, 11). Rows 2, 6, 7 and section 9 belong to Plans 2-3, as the spec now states.
- Row 1 (the paper's numbers) is reported in `RESULTS.md` (Task 13).
- Names checked across tasks: `S_COLS`, `M_COLS`, `N_COLS`, `TAYLOR_COLS`, `run_spec`, `ladder`, `common_rows`, `prereg_ok`, `load_task`, `select_specs`, `metrics`, `decide`.
