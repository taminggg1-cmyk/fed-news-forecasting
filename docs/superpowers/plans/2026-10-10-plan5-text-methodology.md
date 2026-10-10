# Plan 5: Evidence for Look-Ahead-Free Text Features (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the Plan 5 analyses: a pre-registered event study (H5), a look-ahead comparison with a standard language model, a Treasury-bill market benchmark and a text-feature ablation, then compile `RESULTS_TEXT.md`.

**Architecture:** New modules only. `events.py` builds event days from the stored document stance and the daily 2-year yield; `event_study.py` runs walk-forward OLS and the Clark-West test; `market_bench.py` builds bill forecasts; `ablation.py` reruns the Plan 1 primary model on feature groups; `naive_stance.py` scores documents with ModernBERT trained on all labels; `evaluate_text.py` writes the report. Plan 1 and Plan 2 modules are imported, never edited.

**Tech Stack:** Python 3.13 (`/d/gold_lm/venv/Scripts/python.exe`), pandas 3.0, numpy, statsmodels, scipy, scikit-learn, torch CUDA, transformers 5.17, pytest.

**Spec:** `docs/superpowers/specs/2026-10-10-text-methodology-design.md` (binding). Market benchmark origin: `docs/superpowers/specs/2026-10-10-replication-market-design.md`, section 3.

## Global Constraints

- Never edit Plan 1 or Plan 2 files: `src/fed_news/{config,http,market,alfred,fomc_calendar,meetings,corpus,indices,panel,labels,stance,text_features,models,stats_tests,walkforward,run_models,evaluate,tft,neural,neural_walk,run_neural,evaluate_h2a,frozen,diag_tft,clark_west}.py` and their tests. Import only.
- Event study (binding, pre-registered in `PREREG_TEXT.md`): event types statement, minutes, presconf, speech; each document keyed to its outcome trading day (first date with a y2 value on or after release; same key = one event); test outcome days 2001-01-01 to 2026-09-30; surprise = event stance minus the document-mean stance of earlier events' documents released within 30 calendar days before the outcome day, fallback = mean stance of the latest earlier event's documents; outcome = 2-year yield change in basis points (`market_daily.pkl` column `y2`) on the outcome day vs the previous trading day; walk-forward OLS on all strictly earlier event days (from 1994), small model `dy2 = a`, large `dy2 = a + b * surprise`; one-sided Clark-West, alpha 0.05; void if the same test on `dy2_before` gives p < 0.05.
- No code may compute event-day yield changes against stance on real data before tag `prereg-text` exists (tests use synthetic data only).
- Look-ahead comparison: `answerdotai/ModernBERT-base`, all labels, `config.STANCE_*` settings, seeds `config.STANCE_SEEDS`, through the unmodified `stance.train_and_score`.
- Market benchmark: bill value = `market.asof(bill, cutoff)` (last value strictly before the row's cutoff); spread = mean of `y_level - bill` over the row's `walkforward.train_mask` rows that have a bill value; at least 12 such rows, else no forecast. Bills: `bill3m` (= `market_daily.pkl` `tbill_3m`, FRED DTB3) and `bill4w` (FRED DTB4WK).
- Outputs go to `results/text/`; data caches to `D:/fed_news/`. Never print or commit the FRED key; network errors are re-raised without the URL.
- Every commit message ends with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Stage only your own files.

## Review Focus

1. A surprise or a walk-forward fit that sees documents or outcomes from the event day or later (must fail the perturbation tests).
2. Weekend or holiday releases: the outcome must use the next business day's yield minus the last earlier yield, not NaN.
3. Several documents on one day: one event row per day, mean stance; the day category follows statement > minutes > speech.
4. The bill spread must use only training rows; rows without a bill value never enter the spread or the test set.
5. `stance.model_name` must be restored after the naive run even if training raises.

## File Structure

| file | responsibility |
|---|---|
| `src/fed_news/events.py` | event days, day stance, surprise, yield changes |
| `src/fed_news/event_study.py` | walk-forward forecasts, Clark-West, void rule, secondary tests, CLI |
| `src/fed_news/market_bench.py` | bill series, bill forecasts, benchmark table, CLI |
| `src/fed_news/ablation.py` | feature-group runs of the Plan 1 primary model, CLI |
| `src/fed_news/naive_stance.py` | ModernBERT scoring, naive panels, pipeline comparison, CLI |
| `src/fed_news/evaluate_text.py` | `RESULTS_TEXT.md` |
| `tests/test_events.py`, `tests/test_event_study.py`, `tests/test_market_bench.py`, `tests/test_ablation.py`, `tests/test_naive_stance.py`, `tests/test_evaluate_text.py` | offline tests |

---

### Task 1: Event days and the event study engine

**Files:** Create `src/fed_news/events.py`, `src/fed_news/event_study.py`, `tests/test_events.py`, `tests/test_event_study.py`.

**Interfaces:**
- Consumes: `clark_west.clark_west(y, pred_small, pred_large) -> (t, p)`; `config.DATA`, `config.RESULTS`.
- Produces: `events.build_events(doc_stance, y2, end=TEST_END) -> DataFrame` (index `day`; columns `stance, surprise, dy2, dy2_before, n_docs, category`); `events.load_events()`; `event_study.walk_forward(events, outcome, regressor="surprise", test_start, test_end) -> DataFrame` (index `day`; `y, pred_small, pred_large, category`); `event_study.decide(primary_p, void_p) -> str`; `event_study.run(events, regressor="surprise", label="chronobert") -> (dict, DataFrame)`; `event_study.main()` writing `results/text/events.csv`, `event_forecasts.csv`, `event_study.csv`.

- [ ] **Step 1: Write `tests/test_events.py`**

```python
import numpy as np
import pandas as pd

from fed_news import events as ev


def docs(rows):
    return pd.DataFrame(rows, columns=["doc_id", "type", "release", "stance", "n_sent"]).assign(
        release=lambda d: pd.to_datetime(d["release"]))


def test_surprise_uses_30_day_window_before_the_day():
    d = ev._docs(docs([("a", "speech", "2005-01-01", 0.9, 5),      # 40 days before: outside window
                       ("b", "speech", "2005-01-20", 0.2, 5),
                       ("c", "minutes", "2005-02-01", 0.4, 5),
                       ("d", "statement", "2005-02-10", 0.1, 5),   # event day
                       ("e", "speech", "2005-02-10", 0.3, 5)]))    # same day: excluded from the baseline
    assert np.isclose(ev.surprise_at(d, pd.Timestamp("2005-02-10")), 0.2 - 0.3)


def test_surprise_falls_back_to_latest_earlier_document():
    d = ev._docs(docs([("a", "speech", "2005-01-01", 0.5, 5), ("b", "speech", "2005-03-15", 0.1, 5)]))
    assert np.isclose(ev.surprise_at(d, pd.Timestamp("2005-03-15")), 0.1 - 0.5)


def test_surprise_ignores_later_documents():
    rows = [("a", "speech", "2005-01-10", 0.5, 5), ("b", "speech", "2005-01-20", 0.1, 5)]
    base = ev.surprise_at(ev._docs(docs(rows)), pd.Timestamp("2005-01-20"))
    later = ev.surprise_at(ev._docs(docs(rows + [("c", "speech", "2005-01-25", 9.0, 5)])), pd.Timestamp("2005-01-20"))
    assert base == later


def test_yield_change_weekend_and_day_before():
    y = pd.Series([1.00, 1.05, 1.20], index=pd.to_datetime(["2005-01-06", "2005-01-07", "2005-01-10"]))
    assert np.isclose(ev.yield_change(y, pd.Timestamp("2005-01-08")), 15.0)   # Saturday: Monday - Friday
    assert np.isclose(ev.yield_change(y, pd.Timestamp("2005-01-07")), 5.0)
    assert np.isclose(ev.yield_change_before(y, pd.Timestamp("2005-01-10")), 5.0)


def test_day_category_priority():
    assert ev.day_category(["speech", "minutes"]) == "minutes"
    assert ev.day_category(["speech", "presconf"]) == "statement"
    assert ev.day_category(["speech"]) == "speech"


def test_build_events_one_row_per_day_and_drops_nan():
    d = docs([("a", "speech", "2005-01-03", 0.5, 5), ("b", "speech", "2005-01-04", 0.1, 5),
              ("c", "minutes", "2005-01-04", 0.3, 5), ("z", "beige", "2005-01-04", 0.9, 5)])
    y = pd.Series([1.0, 1.1, 1.3, 1.2], index=pd.to_datetime(["2004-12-31", "2005-01-03", "2005-01-04", "2005-01-05"]))
    out = ev.build_events(d, y)
    assert list(out.index) == [pd.Timestamp("2005-01-04")]              # first day has no surprise
    row = out.iloc[0]
    assert np.isclose(row["stance"], 0.2) and row["n_docs"] == 2 and row["category"] == "minutes"
    assert np.isclose(row["dy2"], 20.0) and np.isclose(row["dy2_before"], 10.0)
```

- [ ] **Step 2: Write `tests/test_event_study.py`**

```python
import numpy as np
import pandas as pd

from fed_news import event_study as es


def synthetic(n=400, slope=5.0, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("1998-01-01", periods=n)
    s = rng.normal(size=n)
    return pd.DataFrame({"surprise": s, "dy2": slope * s + rng.normal(size=n),
                         "dy2_before": rng.normal(size=n), "category": "speech",
                         "stance": s, "n_docs": 1}, index=pd.Index(days, name="day"))


def test_walk_forward_uses_only_earlier_days():
    e = synthetic()
    k = 250
    base = es.walk_forward(e, "dy2", test_start=str(e.index[200].date()))
    pert = e.copy()
    pert.iloc[k:, pert.columns.get_loc("dy2")] = 100.0
    again = es.walk_forward(pert, "dy2", test_start=str(e.index[200].date()))
    cut = e.index[k]
    cols = ["pred_small", "pred_large"]
    assert np.allclose(base.loc[:cut, cols], again.loc[:cut, cols])


def test_decide_rule():
    assert es.decide(0.01, 0.20) == "supported"
    assert es.decide(0.20, 0.20) == "not supported"
    assert es.decide(0.01, 0.01) == "void"


def test_run_detects_signal_and_void_is_null():
    summary, wf = es.run(synthetic(), test_start=str(synthetic().index[200].date()))
    assert summary["verdict"] == "supported"
    assert summary["cw_p"] < 0.01 and summary["void_cw_p"] > 0.05
    assert summary["n_test"] == len(wf) == 200


def test_walk_forward_refuses_tiny_training_set():
    e = synthetic(n=60)
    try:
        es.walk_forward(e, "dy2", test_start=str(e.index[10].date()))
    except ValueError as err:
        assert "training events" in str(err)
    else:
        raise AssertionError("must refuse fewer than MIN_TRAIN training events")
```

- [ ] **Step 3: Run both test files, expect ImportError.**

Run: `/d/gold_lm/venv/Scripts/python.exe -m pytest tests/test_events.py tests/test_event_study.py -q`

- [ ] **Step 4: Write `src/fed_news/events.py`**

```python
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
```

- [ ] **Step 5: Write `src/fed_news/event_study.py`**

```python
"""Plan 5 event study (H5): does the Fed stance surprise predict the same-day 2-year yield move, out of sample?

Primary test (PREREG_TEXT.md): walk-forward OLS on all strictly earlier event days; small model dy2 = a,
large model dy2 = a + b * surprise; one-sided Clark-West, p < 0.05. Void if the same test on the day-before
yield change gives p < 0.05. Secondary: full-sample OLS with Newey-West errors (5 lags), Clark-West by day
category, out-of-sample sign accuracy.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from fed_news import config
from fed_news.clark_west import clark_west
from fed_news.events import TEST_END, TEST_START, load_events

ALPHA = 0.05
MIN_TRAIN = 50
OUT = config.RESULTS / "text"


def walk_forward(events: pd.DataFrame, outcome: str, regressor: str = "surprise",
                 test_start: str = TEST_START, test_end: str = TEST_END) -> pd.DataFrame:
    ev = events.sort_index()
    days = ev.index
    y, s = ev[outcome].to_numpy(float), ev[regressor].to_numpy(float)
    rows = []
    for i in np.flatnonzero((days >= pd.Timestamp(test_start)) & (days <= pd.Timestamp(test_end))):
        tr = np.flatnonzero(days < days[i])
        if len(tr) < MIN_TRAIN:
            raise ValueError(f"only {len(tr)} training events before {days[i].date()}")
        a, b = np.linalg.lstsq(np.column_stack([np.ones(len(tr)), s[tr]]), y[tr], rcond=None)[0]
        rows.append({"day": days[i], "y": y[i], "pred_small": float(y[tr].mean()),
                     "pred_large": float(a + b * s[i]), "category": ev["category"].iloc[i]})
    return pd.DataFrame(rows).set_index("day")


def decide(primary_p: float, void_p: float, alpha: float = ALPHA) -> str:
    if void_p < alpha:
        return "void"
    return "supported" if primary_p < alpha else "not supported"


def full_sample_ols(events: pd.DataFrame, regressor: str = "surprise", test_start: str = TEST_START,
                    test_end: str = TEST_END) -> dict:
    ev = events[(events.index >= pd.Timestamp(test_start)) & (events.index <= pd.Timestamp(test_end))]
    fit = sm.OLS(ev["dy2"].to_numpy(float), sm.add_constant(ev[regressor].to_numpy(float))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5})
    t = float(fit.tvalues[1])
    return {"ols_n": len(ev), "ols_slope": float(fit.params[1]), "ols_t_hac": t,
            "ols_p_one_sided": float(1 - stats.norm.cdf(t))}


def sign_accuracy(wf: pd.DataFrame) -> float:
    nz = wf[wf["y"] != 0]
    return float((np.sign(nz["pred_large"]) == np.sign(nz["y"])).mean())


def run(events: pd.DataFrame, regressor: str = "surprise", label: str = "chronobert",
        test_start: str = TEST_START, test_end: str = TEST_END) -> tuple[dict, pd.DataFrame]:
    wf = walk_forward(events, "dy2", regressor, test_start, test_end)
    wf_void = walk_forward(events, "dy2_before", regressor, test_start, test_end)
    t, p = clark_west(wf["y"], wf["pred_small"], wf["pred_large"])
    tv, pv = clark_west(wf_void["y"], wf_void["pred_small"], wf_void["pred_large"])
    summary = {"pipeline": label, "n_test": len(wf), "cw_t": t, "cw_p": p, "void_cw_t": tv, "void_cw_p": pv,
               "verdict": decide(p, pv), "sign_accuracy": sign_accuracy(wf)}
    for cat, g in wf.groupby("category"):
        summary[f"cw_p_{cat}"] = clark_west(g["y"], g["pred_small"], g["pred_large"])[1]
        summary[f"n_{cat}"] = len(g)
    summary.update(full_sample_ols(events, regressor, test_start, test_end))
    return summary, wf


def main() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    events = load_events()
    events.to_csv(OUT / "events.csv")
    summary, wf = run(events)
    wf.to_csv(OUT / "event_forecasts.csv")
    table = pd.DataFrame([summary])
    table.to_csv(OUT / "event_study.csv", index=False)
    print(table.T.to_string())
    return table


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the two test files (expect 10 passed), then the full offline suite (`-m pytest -q`).**
- [ ] **Step 7: Commit** `src/fed_news/events.py src/fed_news/event_study.py tests/test_events.py tests/test_event_study.py` with message `Add Plan 5 event study engine (events, walk-forward Clark-West, void rule)`. Do NOT run `event_study.main()` on real data.

---

### Task 2: Market benchmark and ablation

**Files:** Create `src/fed_news/market_bench.py`, `src/fed_news/ablation.py`, `tests/test_market_bench.py`, `tests/test_ablation.py`.

**Interfaces:**
- Consumes: `market.asof`, `walkforward.train_mask`, `walkforward.Spec`, `walkforward.run_spec(df, spec, test_start)`, `models.classes_from_change`, `run_models.load_task`, `panel.S_COLS`, `panel.N_COLS`, `evaluate.metrics`, `clark_west.compare(small, large) -> dict` (keys `n, dm_hln_p, clark_west_t, clark_west_p`), `stats_tests.dm_hln`, `alfred.api_key`, `http.get`.
- Produces: `market_bench.bill_forecast(df, bill, task, test_start) -> DataFrame` (standard prediction frame); `market_bench.load_bills(refresh=False) -> DataFrame` (columns `bill3m`, `bill4w`); `market_bench.main()` writing `results/text/preds_{task}_{bill}.csv` and `results/text/market_benchmark.csv`; `ablation.GROUPS`; `ablation.specs(task) -> list[Spec]`; `ablation.main()` writing `results/text/preds_{task}_{name}.csv` and `results/text/ablation.csv`.

- [ ] **Step 1: Write `tests/test_market_bench.py`**

```python
import numpy as np
import pandas as pd

from fed_news import market_bench as mb


def panel(n=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1996-01-01", periods=n, freq="MS")
    y = 3 + np.cumsum(rng.normal(scale=0.1, size=n))
    return pd.DataFrame({"cutoff": idx, "y_level": y, "s_rate_lag": np.r_[y[0], y[:-1]],
                         "y_class": 1}, index=idx)


def test_bill_forecast_is_bill_plus_training_spread():
    df = panel()
    bill = pd.Series(df["y_level"].to_numpy() - 0.25, index=df.index - pd.Timedelta(days=1))  # dated before cutoff
    out = mb.bill_forecast(df, bill, "T1", "1998-01-01")
    assert np.allclose(out["y_pred"], out["y_true"])            # constant spread 0.25 recovered exactly
    assert {"date", "y_true", "y_pred", "lag", "class_true", "class_pred"} == set(out.columns)


def test_spread_uses_only_training_rows():
    df = panel()
    bill = pd.Series(df["y_level"].to_numpy() - 0.25, index=df.index - pd.Timedelta(days=1))
    base = mb.bill_forecast(df, bill, "T1", "1998-01-01")
    pert = df.copy()
    k = 40
    pert.iloc[k:, pert.columns.get_loc("y_level")] += 5.0      # later targets change
    again = mb.bill_forecast(pert, bill, "T1", "1998-01-01")
    upto = df.index[k + 1]                                       # T1 trains on rows <= i-2
    assert np.allclose(base.set_index("date").loc[:upto, "y_pred"], again.set_index("date").loc[:upto, "y_pred"])


def test_rows_without_bill_are_skipped():
    df = panel()
    bill = pd.Series(df["y_level"].to_numpy() - 0.25, index=df.index - pd.Timedelta(days=1))
    bill = bill[bill.index >= pd.Timestamp("1999-01-01")]       # no bill before 1999
    out = mb.bill_forecast(df, bill, "T1", "1998-01-01")
    assert out["date"].min() > pd.Timestamp("1999-12-01")        # needs a bill and 12 training rows with bills


def test_fetch_error_hides_url(monkeypatch):
    def boom(url, **kw):
        raise FileNotFoundError(url)
    monkeypatch.setattr(mb.http, "get", boom)
    monkeypatch.setattr(mb, "api_key", lambda: "SECRETKEY")
    try:
        mb.fetch_series("DTB4WK")
    except RuntimeError as err:
        assert "SECRETKEY" not in str(err) and err.__cause__ is None and err.__suppress_context__
    else:
        raise AssertionError("must raise")
```

- [ ] **Step 2: Write `tests/test_ablation.py`**

```python
from fed_news import ablation as ab
from fed_news.panel import N_COLS, S_COLS


def test_specs_cover_each_group_alone_and_left_out():
    specs = {s.name: s for s in ab.specs("T1")}
    assert set(specs) == {"S+stance", "S+beige", "S+newspapers", "S+N-stance", "S+N-beige", "S+N-newspapers"}
    assert specs["S+stance"].cols == S_COLS + ["n_stance", "n_stance_chg"]
    assert set(specs["S+N-beige"].cols) == set(S_COLS + [c for c in N_COLS if c != "n_beige"])
    assert all(s.kind == "primary" and s.task == "T1" for s in specs.values())
```

- [ ] **Step 3: Run, expect ImportError.**

- [ ] **Step 4: Write `src/fed_news/market_bench.py`**

```python
"""Plan 5 market benchmark (descriptive): Treasury-bill forecasts of the policy rate.

forecast(row) = bill yield at the last date strictly before the row's cutoff + spread, where spread is the
mean of (y_level - bill) over the row's training rows (walkforward.train_mask) that have a bill value; at
least 12 such rows are required. Bills: bill3m = market_daily.pkl tbill_3m (FRED DTB3) and bill4w = FRED
DTB4WK (from 2001-07-31). Rows without a bill value get no forecast.
"""
import json

import numpy as np
import pandas as pd

from fed_news import config, http, models
from fed_news.alfred import api_key
from fed_news.clark_west import compare
from fed_news.evaluate import metrics
from fed_news.market import asof
from fed_news.run_models import load_task
from fed_news.walkforward import train_mask

FRED_OBS = "https://api.stlouisfed.org/fred/series/observations?series_id={}&api_key={}&file_type=json"
BILLS = ("bill3m", "bill4w")
MIN_SPREAD_ROWS = 12
COLUMNS = ["date", "y_true", "y_pred", "lag", "class_true", "class_pred"]
OUT = config.RESULTS / "text"


def fetch_series(sid: str) -> pd.Series:
    try:
        payload = json.loads(http.get(FRED_OBS.format(sid, api_key()), cache=False))
    except Exception:
        raise RuntimeError(f"FRED request failed for {sid}") from None  # never echo the URL: it holds the key
    obs = pd.DataFrame(payload["observations"])
    return pd.Series(pd.to_numeric(obs["value"], errors="coerce").to_numpy(),
                     index=pd.to_datetime(obs["date"]), name=sid).dropna()


def load_bills(refresh: bool = False) -> pd.DataFrame:
    path = config.DATA / "market_bill.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    daily = pd.read_pickle(config.DATA / "market_daily.pkl")
    bills = daily[["tbill_3m"]].rename(columns={"tbill_3m": "bill3m"}).join(
        fetch_series("DTB4WK").rename("bill4w"), how="outer")
    bills.to_pickle(path)
    return bills


def bill_forecast(df: pd.DataFrame, bill: pd.Series, task: str, test_start: str) -> pd.DataFrame:
    b = np.array([asof(bill, c) for c in df["cutoff"]])
    y, lag = df["y_level"].to_numpy(float), df["s_rate_lag"].to_numpy(float)
    gap = y - b
    rows = []
    for i in np.flatnonzero(df.index >= pd.Timestamp(test_start)):
        tr = train_mask(len(df), i, task) & ~np.isnan(gap)
        if np.isnan(b[i]) or tr.sum() < MIN_SPREAD_ROWS:
            continue
        pred = float(b[i] + gap[tr].mean())
        rows.append({"date": df.index[i], "y_true": y[i], "y_pred": pred, "lag": lag[i],
                     "class_true": int(df["y_class"].iloc[i]),
                     "class_pred": int(models.classes_from_change(np.array([pred - lag[i]]))[0])})
    return pd.DataFrame(rows, columns=COLUMNS)


def _on(dates, frame):
    return frame[frame["date"].isin(dates)].sort_values("date").reset_index(drop=True)


def main() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    bills = load_bills()
    load = lambda task, name: pd.read_csv(config.RESULTS / f"preds_{task}_{name}.csv", parse_dates=["date"])
    rows = []
    for task in ("T1", "T2"):
        df = load_task(task)
        preds = {name: load(task, name) for name in ("rw", "S", "S+N", "S+M")}
        for bill in BILLS:
            p = bill_forecast(df, bills[bill], task, config.TEST_START)
            p.to_csv(OUT / f"preds_{task}_{bill}.csv", index=False)
            preds[bill] = p
        windows = {"full": None, "paper": ("2016-01-01", "2025-04-01")} if task == "T1" else {"full": None}
        for wname, w in windows.items():
            for bill in BILLS:
                dates = set(preds[bill]["date"])
                if w:
                    dates = {d for d in dates if pd.Timestamp(w[0]) <= d <= pd.Timestamp(w[1])}
                for name, p in preds.items():
                    if name in BILLS and name != bill:
                        continue
                    pw, rw, bw = _on(dates, p), _on(dates, preds["rw"]), _on(dates, preds[bill])
                    rows.append({"task": task, "window": wname, "bill_dates": bill, "model": name, **metrics(pw),
                                 "p_vs_no_change": compare(rw, pw)["dm_hln_p"] if name != "rw" else np.nan,
                                 "p_vs_bill": compare(bw, pw)["dm_hln_p"] if name != bill else np.nan})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "market_benchmark.csv", index=False)
    print(table[["task", "window", "bill_dates", "model", "n", "rmse", "mae", "accuracy", "p_vs_no_change",
                 "p_vs_bill"]].round(4).to_string(index=False))
    return table


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write `src/fed_news/ablation.py`**

```python
"""Plan 5 ablation (exploratory): which text features carry the signal?

The Plan 1 primary model (T1 ridge, T2 ordered logit) with S plus one feature group, and with S+N minus one
group. Each is compared with the stored S forecasts: RMSE, change against S, DM-HLN p and Clark-West p.
"""
import pandas as pd

from fed_news import config
from fed_news.clark_west import compare
from fed_news.evaluate import metrics
from fed_news.panel import N_COLS, S_COLS
from fed_news.run_models import load_task
from fed_news.walkforward import Spec, run_spec

GROUPS = {"stance": ["n_stance", "n_stance_chg"], "beige": ["n_beige"], "newspapers": ["n_news", "n_gpr", "n_epu"]}
OUT = config.RESULTS / "text"


def specs(task: str) -> list[Spec]:
    out = []
    for group, cols in GROUPS.items():
        out.append(Spec(f"S+{group}", task, "primary", S_COLS + cols))
        out.append(Spec(f"S+N-{group}", task, "primary", S_COLS + [c for c in N_COLS if c not in cols]))
    return out


def main() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in ("T1", "T2"):
        df = load_task(task)
        base = pd.read_csv(config.RESULTS / f"preds_{task}_S.csv", parse_dates=["date"])
        full = pd.read_csv(config.RESULTS / f"preds_{task}_S+N.csv", parse_dates=["date"])
        results = {"S+N (all)": full}
        for spec in specs(task):
            p = run_spec(df, spec, config.TEST_START)
            p.to_csv(OUT / f"preds_{task}_{spec.name}.csv", index=False)
            results[spec.name] = p
        rmse_s = metrics(base)["rmse"]
        for name, p in results.items():
            m = metrics(p)
            rows.append({"task": task, "model": name, "rmse": m["rmse"], "accuracy": m["accuracy"],
                         "change_vs_S_pct": 100 * (m["rmse"] / rmse_s - 1), **compare(base, p)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "ablation.csv", index=False)
    print(table.round(4).to_string(index=False))
    return table


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the two test files (expect 5 passed) and the full offline suite.**
- [ ] **Step 7: Commit** the four files, message `Add Plan 5 T-bill market benchmark and text-feature ablation`. Do NOT run either `main()` on real data and do NOT fetch from FRED.

---

### Task 3: Naive (look-ahead) stance pipeline

**Files:** Create `src/fed_news/naive_stance.py`, `tests/test_naive_stance.py`.

**Interfaces:**
- Consumes: `stance.train_and_score`, `stance.model_name`, `stance.STANCE_TYPES`, `stance._sentence_table`, `labels.load_labels`, `corpus.build_corpus`, `text_features.fed_stance`, `walkforward.common_rows`, `walkforward.Spec`, `walkforward.run_spec`, `events.build_events`, `event_study.run`, `clark_west.compare`, `evaluate.metrics`.
- Produces: `naive_stance.NAIVE_MODEL`; `score_naive(name=NAIVE_MODEL) -> DataFrame` (columns `doc_id, type, release, stance, n_sent`, saved to `D:/fed_news/doc_stance_naive.pkl`, model name to `results/text/naive_model.txt`); `naive_panel(name, doc_stance) -> DataFrame`; `compare_pipelines() -> DataFrame` writing `results/text/naive_comparison.csv`, `results/text/naive_event_study.csv`, `results/text/preds_{task}_S+N_naive.csv`; CLI `python -m fed_news.naive_stance [--score] [--compare]`.

- [ ] **Step 1: Write `tests/test_naive_stance.py`**

```python
import numpy as np
import pandas as pd

from fed_news import naive_stance as ns, stance


def test_score_naive_uses_named_model_and_restores(monkeypatch, tmp_path):
    docs = pd.DataFrame({"doc_id": ["a", "b"], "type": ["speech", "minutes"],
                         "release": pd.to_datetime(["2005-01-03", "2005-01-04"]),
                         "text": ["Rates may rise again later this year if inflation stays high. "
                                  "Inflation remains well above the committee objective.",
                                  "Policy will remain accommodative for an extended period of time."]})
    labels = pd.DataFrame({"sentence": ["x", "y", "z"], "label": [0, 1, 2], "year": [2001, 2010, 2020]})
    seen = []

    def fake(train, texts, vintage, seed):
        seen.append((stance.model_name(vintage), len(train)))
        out = np.zeros((len(texts), 3), dtype=np.float32)
        out[:, 1] = 0.7
        out[:, 0] = 0.2
        out[:, 2] = 0.1
        return out

    monkeypatch.setattr(ns, "build_corpus", lambda: docs)
    monkeypatch.setattr(ns, "load_labels", lambda: labels)
    monkeypatch.setattr(stance, "train_and_score", fake)
    monkeypatch.setattr(ns, "OUT_PATH", tmp_path / "naive.pkl")
    monkeypatch.setattr(ns, "OUT", tmp_path)
    out = ns.score_naive("some/model")
    assert all(name == "some/model" and n == 3 for name, n in seen)       # all labels, the named model
    assert stance.model_name(2010) == "manelalab/chrono-bert-v1-20101231"  # restored
    assert np.allclose(out["stance"], 0.5) and list(out["n_sent"]) == [2, 1]


def test_model_name_restored_when_training_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(ns, "build_corpus", lambda: pd.DataFrame(
        {"doc_id": ["a"], "type": ["speech"], "release": pd.to_datetime(["2005-01-03"]),
         "text": ["Policy will remain accommodative for an extended period of time."]}))
    monkeypatch.setattr(ns, "load_labels", lambda: pd.DataFrame({"sentence": ["x"], "label": [0], "year": [2001]}))

    def boom(*a, **k):
        raise RuntimeError("gpu")
    monkeypatch.setattr(stance, "train_and_score", boom)
    try:
        ns.score_naive("some/model")
    except RuntimeError:
        pass
    assert stance.model_name(2010) == "manelalab/chrono-bert-v1-20101231"


def test_naive_panel_replaces_only_stance_columns(monkeypatch, tmp_path):
    idx = pd.date_range("2005-01-01", periods=4, freq="MS")
    rows = pd.DataFrame({"win_start": idx - pd.offsets.MonthBegin(1), "cutoff": idx}, index=idx)
    full = rows.assign(n_stance=9.0, n_stance_chg=9.0, n_beige=1.0, s_rate_lag=2.0)
    rows.to_pickle(tmp_path / "t1.pkl")
    full.to_pickle(tmp_path / "t1_full.pkl")
    monkeypatch.setattr(ns.config, "DATA", tmp_path)
    ds = pd.DataFrame({"doc_id": ["a", "b"], "type": ["speech", "speech"],
                       "release": pd.to_datetime(["2004-12-15", "2005-02-10"]), "stance": [0.2, 0.6], "n_sent": [1, 1]})
    out = ns.naive_panel("t1", ds)
    assert np.allclose(out["n_stance"], [0.2, 0.2, 0.6, 0.6])             # carried forward when a month is empty
    assert np.isnan(out["n_stance_chg"].iloc[0]) and np.allclose(out["n_stance_chg"].iloc[1:], [0.0, 0.4, 0.0])
    assert (out["n_beige"] == 1.0).all()
```

- [ ] **Step 2: Run, expect ImportError.**

- [ ] **Step 3: Write `src/fed_news/naive_stance.py`**

```python
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
from fed_news.event_study import run as run_event_study
from fed_news.events import build_events
from fed_news.labels import load_labels
from fed_news.panel import N_COLS, S_COLS
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
                         "change_vs_S_pct": 100 * (m["rmse"] / rmse_s - 1), **compare(on(base), on(preds))})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "naive_comparison.csv", index=False)
    y2 = pd.read_pickle(config.DATA / "market_daily.pkl")["y2"]
    summary, _ = run_event_study(build_events(naive, y2), label="naive")
    pd.DataFrame([summary]).to_csv(OUT / "naive_event_study.csv", index=False)
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
```

- [ ] **Step 4: Run the test file (expect 3 passed) and the full offline suite.**
- [ ] **Step 5: Commit** both files, message `Add Plan 5 naive look-ahead stance pipeline (ModernBERT, all labels)`. Do NOT run scoring or the comparison.

---

### Task 4: Report

**Files:** Create `src/fed_news/evaluate_text.py`, `tests/test_evaluate_text.py`.

**Interfaces:**
- Consumes: the CSVs in `results/text/` named above (`event_study.csv`, `naive_event_study.csv`, `naive_comparison.csv`, `market_benchmark.csv`, `ablation.csv`, `naive_model.txt`); `config.REPO`, `config.RESULTS`.
- Produces: `evaluate_text.main()` writing `RESULTS_TEXT.md`; missing inputs give the line `not run` in their section.

- [ ] **Step 1: Write `tests/test_evaluate_text.py`**

```python
import pandas as pd

from fed_news import config, evaluate_text as et


def test_report_with_all_inputs(tmp_path, monkeypatch):
    out = tmp_path / "text"
    out.mkdir()
    pd.DataFrame([{"pipeline": "chronobert", "n_test": 1800, "cw_t": 2.5, "cw_p": 0.006, "void_cw_t": 0.1,
                   "void_cw_p": 0.46, "verdict": "supported", "sign_accuracy": 0.55}]).to_csv(out / "event_study.csv", index=False)
    pd.DataFrame([{"task": "T1", "pipeline": "naive", "rmse": 0.15}]).to_csv(out / "naive_comparison.csv", index=False)
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    monkeypatch.setattr(config, "REPO", tmp_path)
    monkeypatch.setattr(et, "_git", lambda *a: "abc123")
    et.main()
    text = (tmp_path / "RESULTS_TEXT.md").read_text(encoding="utf-8")
    assert "**H5 verdict: supported.**" in text
    assert "exploratory" in text.lower() and "not run" in text      # missing ablation / market sections
```

- [ ] **Step 2: Run, expect ImportError.**

- [ ] **Step 3: Write `src/fed_news/evaluate_text.py`**

```python
"""Plan 5 report: RESULTS_TEXT.md (event study verdict, look-ahead comparison, market benchmark, ablation)."""
import subprocess

import pandas as pd

from fed_news import config

TAG = "prereg-text"


def _git(*args) -> str:
    out = subprocess.run(["git", "-C", str(config.REPO), *args], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else ""


def _table(name: str) -> str:
    path = config.RESULTS / "text" / name
    if not path.exists():
        return "not run"
    return pd.read_csv(path).round(4).to_markdown(index=False)


def main() -> None:
    text_dir = config.RESULTS / "text"
    es = text_dir / "event_study.csv"
    verdict = pd.read_csv(es)["verdict"].iloc[0] if es.exists() else "not run"
    model = (text_dir / "naive_model.txt").read_text(encoding="utf-8").strip() if (text_dir / "naive_model.txt").exists() else "not run"
    lines = [
        "# Results: Plan 5 (text as a forecasting method)", "",
        f"**H5 verdict: {verdict}.**", "",
        "H5 (pre-registered in PREREG_TEXT.md, tag prereg-text): the Fed stance surprise predicts the same-day "
        "2-year Treasury yield change out of sample (one-sided Clark-West, p < 0.05); void if the same test on the "
        "day-before change gives p < 0.05. It does not change the H1 or H2a verdicts.", "",
        "## Event study (confirmatory)", "", _table("event_study.csv"), "",
        "## Look-ahead comparison (exploratory, added after results)", "",
        f"Naive model: {model}, fine-tuned on all labels at once (deliberate look-ahead).", "",
        _table("naive_comparison.csv"), "", "Event study with the naive stance:", "", _table("naive_event_study.csv"), "",
        "## Market benchmark (descriptive)", "", _table("market_benchmark.csv"), "",
        "## Ablation (exploratory, added after results)", "", _table("ablation.csv"), "",
        "## Provenance", "",
        f"- {TAG} tag commit: {_git('rev-list', '-n', '1', TAG) or 'no tag'}",
        f"- HEAD: {_git('rev-parse', '--short', 'HEAD')}", "",
    ]
    (config.REPO / "RESULTS_TEXT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:4]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test (expect 1 passed) and the full offline suite.**
- [ ] **Step 5: Commit** both files, message `Add Plan 5 report writer (RESULTS_TEXT.md)`.

---

### Task 5: Pre-register, run, report (controller)

- [ ] **Step 1: Write `PREREG_TEXT.md`** (exact content; fill the commit hash):

```markdown
# Pre-registration addendum: Plan 5 event study (H5)

Date: 2026-10-10. Code commit: <git rev-parse --short HEAD before this commit>.
Binding: docs/superpowers/specs/2026-10-10-text-methodology-design.md, section 2; code src/fed_news/events.py
and src/fed_news/event_study.py at this commit. No event-day yield change had been computed against stance
before this tag.

## H5
The Fed stance surprise predicts the same-day change in the 2-year Treasury yield out of sample.

- Events: each statement, minutes, press-conference transcript or speech (document stance from
  D:/fed_news/doc_stance.pkl, the Plan 1 time-consistent classifiers) is keyed to its outcome trading day,
  the first date with a 2-year yield on or after its release day; documents with the same key form one
  event (weekend and holiday releases join the next trading day). Test events: outcome days 2001-01-01 to
  2026-09-30. Event stance = mean document stance.
- Surprise: event stance minus the document-mean stance of earlier events' documents released within the
  30 calendar days before the outcome day; if none, minus the mean stance of the latest earlier event's
  documents.
- Outcome: 2-year constant-maturity yield (FRED DGS2, market_daily.pkl y2), basis points, outcome day minus
  previous trading day. Void-rule outcome: the change on the previous trading day.
- Test: walk-forward OLS on all strictly earlier event days (from 1994); small model dy2 = a; large model
  dy2 = a + b * surprise; one-sided Clark-West; supported if p < 0.05.
- Void: the same test with the day-before change as outcome; if p < 0.05, H5 is void.
- Secondary (not decisive): full-sample OLS with Newey-West (5 lags), Clark-West by day category
  (statement > minutes > speech), sign accuracy.

## Exploratory analyses (not pre-registered tests)
Look-ahead comparison (answerdotai/ModernBERT-base on all labels), T-bill market benchmark, feature ablation,
as in the spec sections 3-5. The market benchmark is descriptive: both bills are shown with no selection,
the 3-month bill (full sample) is the headline row, and its DM p-values are not tests.
```

- [ ] **Step 2: Commit and tag**

```bash
git add PREREG_TEXT.md
git commit -m "Pre-register Plan 5 event study (H5)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git tag prereg-text
```

- [ ] **Step 3: Start the naive scoring on the GPU in the background** (harness background command so its exit notifies): `PYTHONPATH=src PYTHONIOENCODING=utf-8 $PY -u -m fed_news.naive_stance --score > logs/naive_score.log 2>&1`. If the model fails to load, rerun with `roberta-base` (edit nothing: call `score_naive("roberta-base")` from `-c`) and record the substitution in DEVIATIONS.md.
- [ ] **Step 4: While it runs (CPU):** `PYTHONPATH=src $PY -m fed_news.event_study`, then `-m fed_news.market_bench`, then `-m fed_news.ablation`. Sanity check before the comparison: `naive_panel("t1", pd.read_pickle(DATA/"doc_stance.pkl"))` reproduces `t1_full` `n_stance` and `n_stance_chg` exactly (same for t2).
- [ ] **Step 5: After scoring:** `-m fed_news.naive_stance --compare`, then `-m fed_news.evaluate_text`.
- [ ] **Step 6:** full suite; `git diff prereg -- <Plan 1 files>` and `git diff prereg-h2a -- <Plan 2 files>` empty; add DEVIATIONS.md lines for any substitution; commit `RESULTS_TEXT.md results/text/ DEVIATIONS.md` with message `Record Plan 5 results (event study H5, look-ahead comparison, market benchmark, ablation)`.

---

## Self-review notes

- Spec coverage: section 2 (Task 1, Task 5 steps 1-4), section 3 (Task 3, Task 5 step 5), section 4 (Task 2 market_bench; bills = 3-month throughout and 4-week where available, each with its own training spread, which refines the Plan 4 wording of a 4-week series with a 3-month fallback), section 5 (Task 2 ablation), section 7 order (Task 5).
- Names checked across tasks: `build_events`, `load_events`, `walk_forward`, `decide`, `run`, `bill_forecast`, `load_bills`, `specs`, `GROUPS`, `score_naive`, `naive_panel`, `compare_pipelines`, `clark_west.compare` keys `n, dm_hln_p, clark_west_t, clark_west_p`.
