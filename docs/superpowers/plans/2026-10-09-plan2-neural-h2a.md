# Fed News Plan 2: Neural Rows of the Ladder (H2a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the pre-registered neural forecasters (the paper's LSTM and GRU, a meeting-level LSTM, a TFT, and a linear-anchor + TFT-correction hybrid), each with statistics only (S) and statistics + news (S+N), on the same leakage-free rows as Plan 1, and report RMSE/MAE/accuracy tables plus the pre-registered tests for H2a.

**Architecture:** New modules only. `neural.py` holds the networks and their training loops; `tft.py` is the TFT vendored unchanged from the gold project; `neural_walk.py` is a walk-forward engine that reuses Plan 1's training masks, standardisation rule and linear models; `run_neural.py` runs every spec x 8 seeds (resumable) and averages seeds into ensembles; `evaluate_h2a.py` writes `RESULTS_H2A.md`. Plan 1's H1 code is imported, never edited.

**Tech Stack:** Python 3.13 in `D:/gold_lm/venv` (torch 2.11 + CUDA, numpy, pandas 3.0, scikit-learn, scipy, statsmodels, tabulate, pytest). RTX 2060, 6 GB.

**Spec:** `docs/superpowers/specs/2026-10-08-fed-news-design.md` (sections 3 H2a, 7 "Secondary architectures", 8 variance rule, 11) and `PREREG.md` (tag `prereg` = 5d0c31c).

## Global Constraints

- Python: `PY=/d/gold_lm/venv/Scripts/python.exe`; run commands from `/c/Users/User/fed_news` with `PYTHONPATH=src`. Branch `plan1-h1` (Plan 2 continues on it).
- Never modify Plan 1 files: `src/fed_news/{config,http,market,alfred,fomc_calendar,meetings,corpus,indices,panel,labels,stance,text_features,models,stats_tests,walkforward,run_models,evaluate}.py` and their tests. `git diff prereg -- src/fed_news/walkforward.py src/fed_news/evaluate.py src/fed_news/run_models.py src/fed_news/models.py` must stay empty.
- Inputs: `run_models.load_task("T1"|"T2")` (the Plan 1 common rows: T1 378 rows, T2 260 rows); test windows from `config.TEST_START` ("2001-01-01"): T1 308 months, T2 212 meetings.
- Training rows exactly as Plan 1: `walkforward.train_mask(n, i, task)` (T1 rows <= i-2, T2 rows < i). Features standardised on each refit's training rows only.
- Paper LSTM (T1): 12 steps, 240 units, one layer, no dropout, learning rate 1e-4, 300 epochs, predicts the level, refit every month (spec section 7).
- Paper GRU (T1): 12 steps, 188 units, one layer, no dropout, learning rate 2e-4, 449 epochs, predicts the level, refit every month.
- If one seed-pass of a paper LSTM/GRU spec takes longer than one hour, rerun the paper LSTM/GRU specs with refit every 3 months and record the deviation in `DEVIATIONS.md`.
- Meeting LSTM (T2): lookback 4 meetings, refit every 8 meetings; other settings as the paper LSTM (240 units, lr 1e-4, 300 epochs, level target).
- TFT: `CompactTFT` from the gold project, hidden size 32, 4 heads, dropout 0.2, lookback 12 months (T1) or 8 meetings (T2), early stopping on the last 20% of each training window (patience 20, at most 300 epochs, Adam lr 1e-3), refit every 12 months (T1) or 8 meetings (T2), predicts the change in the rate.
- Hybrid: the Plan 1 primary linear model (T1 ridge on the change; T2 penalised ordered logit, expected change) is the anchor, refitted at every test row exactly as in Plan 1; a TFT with the same settings, refitted on its own schedule, is trained on the anchor's residuals computed out-of-fold by expanding-window CV inside the training window (folds whose training part has fewer than 20 rows are skipped). Forecast = lag + anchor change + TFT correction.
- Training for LSTM/GRU/meeting LSTM is mini-batch Adam on MSE, batch size 32 (Keras default; the paper did not report its batch size), shuffled each epoch by a generator seeded with the run seed; the TFT trains full batch with early stopping. Every neural training target (level, change or hybrid residual) is standardised with the mean and sd of that refit's fit rows and inverted at prediction. 8 seeds (0-7) per spec; the seed sets torch's initialisation and the batch order.
- Continuous forecasts become classes with the +/-0.125 band (`models.classes_from_change`). Ensembles = mean of the 8 seeds' forecasts.
- Variance rule (spec section 8): a neural S+N vs S comparison counts only if mean RMSE(S) - mean RMSE(S+N) exceeds 2 x the larger of the two seed standard deviations (ddof = 1).
- Tests per architecture on the seed-mean ensembles: one-sided DM-HLN on squared error (T1 and T2); exact McNemar on correct classes (T2). Holm adjustment across all H2 p-values so far (Plan 1 `results/h2_pvalues.csv` plus H2a); final Holm after Plan 3.
- No S+N neural forecast on a test window until both git tags `prereg` and `prereg-h2a` exist (`prereg-h2a` is created in Task 4 after `PREREG_H2A.md` is committed).
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. The first test rows have short histories: every test row must still get exactly one forecast (no silently skipped rows). Test: Task 2 `test_one_forecast_per_test_row`.
2. The hybrid's earliest refits have few out-of-fold folds with >= 20 rows; it must still run and give finite forecasts. Test: Task 2 `test_hybrid_runs_at_earliest_refit`.
3. Seed files produced over different dates must never be averaged into one ensemble. Test: Task 3 `test_ensemble_refuses_mismatched_dates`.
4. A non-finite feature value must stop the run, not train a NaN network. Test: Task 2 `test_rejects_non_finite_inputs`.
5. A text spec run before the `prereg-h2a` tag must be refused. Test: Task 2 `test_text_spec_refused_without_h2a_tag`.

## File Structure

```
src/fed_news/
  tft.py              CompactTFT vendored unchanged from C:/Users/User/gold_gpt2/src/tft.py
  neural.py           RNNRegressor, build_model, make_sequences, fit_fixed, fit_early_stopping, predict
  neural_walk.py      NeuralSpec, neural_ladder, prereg_h2a_ok, standardise, oof_residuals, run_neural_spec
  run_neural.py       CLI: specs x seeds (resumable), run log, seed-mean ensembles
  evaluate_h2a.py     seed metrics, ensemble metrics, variance rule, tests, Holm, RESULTS_H2A.md
  frozen.py           frozen-model analysis (train once before 2016 / 2020, never update): run + RESULTS_FROZEN.md
tests/
  test_neural.py, test_neural_walk.py, test_run_neural.py, test_evaluate_h2a.py
PREREG_H2A.md         settings fixed before any S+N neural run (Task 4)
results/h2a/          per-seed and ensemble predictions, run_log.json, metric and test tables
RESULTS_H2A.md        Plan 2 report
```

---

### Task 1: Networks and training loops

**Files:**
- Create: `src/fed_news/tft.py` (copy), `src/fed_news/neural.py`
- Test: `tests/test_neural.py`

**Interfaces:**
- Produces: `tft.CompactTFT(n_features, hidden=32, n_heads=4, dropout=0.2)` (forward `(batch, time, features) -> (batch,)`); `neural.RNNRegressor(n_features, hidden, cell)`; `neural.build_model(arch: str, n_features: int, hidden: int, dropout: float, seed: int) -> nn.Module` (`arch` in `lstm`, `gru`, `tft`); `neural.make_sequences(Z: np.ndarray, rows, lookback: int) -> np.ndarray` (shape `(len(rows), lookback, n_features)`, window ending at each row); `neural.fit_fixed(model, X, y, lr, epochs, device) -> model`; `neural.fit_early_stopping(model, X, y, lr, max_epochs, patience, val_frac, device) -> model`; `neural.predict(model, X, device) -> np.ndarray`.

- [ ] **Step 1: Vendor the TFT**

```bash
cp /c/Users/User/gold_gpt2/src/tft.py src/fed_news/tft.py
```
Then insert one line at the very top of `src/fed_news/tft.py` (above the existing docstring): `# Vendored unchanged from gold_gpt2/src/tft.py (Plan 2, spec section 7).` Change nothing else.

- [ ] **Step 2: Write the failing test `tests/test_neural.py`**

```python
import numpy as np
import pytest
import torch

from fed_news import neural


def test_make_sequences_windows_end_at_each_row():
    Z = np.arange(20, dtype=float).reshape(10, 2)
    seq = neural.make_sequences(Z, np.array([3, 9]), 4)
    assert seq.shape == (2, 4, 2)
    assert seq[0, -1].tolist() == [6.0, 7.0] and seq[0, 0].tolist() == [0.0, 1.0]
    assert seq[1, -1].tolist() == [18.0, 19.0]


def test_make_sequences_refuses_rows_without_enough_history():
    with pytest.raises(ValueError, match="history"):
        neural.make_sequences(np.zeros((10, 2)), np.array([2]), 4)


def test_build_model_shapes_and_seed_determinism():
    x = torch.zeros((5, 12, 3))
    for arch in ("lstm", "gru", "tft"):
        a = neural.build_model(arch, 3, 16, 0.0, seed=7)
        b = neural.build_model(arch, 3, 16, 0.0, seed=7)
        a.eval(); b.eval()
        assert a(x).shape == (5,)
        assert torch.equal(a(x), b(x))


def test_fit_fixed_reduces_loss_on_learnable_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 6, 2)).astype(np.float32)
    y = X[:, -1, 0] * 2.0
    model = neural.build_model("lstm", 2, 16, 0.0, seed=0)
    before = float(np.mean((neural.predict(model, X, "cpu") - y) ** 2))
    neural.fit_fixed(model, X, y, lr=1e-2, epochs=150, device="cpu")
    after = float(np.mean((neural.predict(model, X, "cpu") - y) ** 2))
    assert after < 0.5 * before


class Counter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x[:, -1, 0] * self.w


def test_fit_early_stopping_stops_after_patience_without_improvement():
    X = np.ones((20, 3, 1), dtype=np.float32)
    y = np.ones(20, dtype=np.float32)
    model = Counter()
    neural.fit_early_stopping(model, X, y, lr=0.0, max_epochs=50, patience=3, val_frac=0.2, device="cpu")
    assert model.calls == 1 + 2 * 3   # untrained validation pass, then 3 epochs (train + val) without improvement


def test_fit_early_stopping_never_returns_worse_than_start_on_validation():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 4, 2)).astype(np.float32)
    y = X[:, -1, 1].astype(np.float32)
    model = neural.build_model("tft", 2, 8, 0.0, seed=0)
    val_x, val_y = X[40:], y[40:]
    start = float(np.mean((neural.predict(model, val_x, "cpu") - val_y) ** 2))
    neural.fit_early_stopping(model, X, y, lr=1e-2, max_epochs=40, patience=5, val_frac=0.2, device="cpu")
    end = float(np.mean((neural.predict(model, val_x, "cpu") - val_y) ** 2))
    assert end <= start + 1e-9
```

- [ ] **Step 3: Run it to verify it fails**

Run: `$PY -m pytest tests/test_neural.py -v`
Expected: FAIL with `ImportError: cannot import name 'neural'`

- [ ] **Step 4: Write `src/fed_news/neural.py`**

```python
"""Neural forecasters for Plan 2 (spec section 7, secondary architectures).

The paper's LSTM/GRU and the meeting LSTM train full-batch Adam on MSE for a fixed number of
epochs; the TFT (vendored in tft.py) trains with early stopping on the chronologically last part
of its training rows and keeps the best epoch's weights.
"""
import copy

import numpy as np
import torch
import torch.nn as nn

from fed_news.tft import CompactTFT


class RNNRegressor(nn.Module):
    """One recurrent layer (LSTM or GRU) and a linear head on the last time step."""

    def __init__(self, n_features: int, hidden: int, cell: str):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(n_features, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def build_model(arch: str, n_features: int, hidden: int, dropout: float, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    if arch in ("lstm", "gru"):
        return RNNRegressor(n_features, hidden, arch)
    if arch == "tft":
        return CompactTFT(n_features, hidden=hidden, n_heads=4, dropout=dropout)
    raise ValueError(f"unknown arch {arch}")


def make_sequences(Z: np.ndarray, rows, lookback: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=int)
    if len(rows) == 0:
        return np.empty((0, lookback, Z.shape[1]), dtype=np.float32)
    if rows.min() < lookback - 1:
        raise ValueError(f"row {rows.min()} has less than {lookback} rows of history")
    return np.stack([Z[j - lookback + 1: j + 1] for j in rows]).astype(np.float32)


def _tensor(a, device: str) -> torch.Tensor:
    return torch.tensor(np.asarray(a), dtype=torch.float32, device=device)


def fit_fixed(model, X, y, lr: float, epochs: int, device: str):
    model.to(device).train()
    xt, yt = _tensor(X, device), _tensor(y, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(xt), yt)
        loss.backward()
        opt.step()
    return model


def fit_early_stopping(model, X, y, lr: float, max_epochs: int, patience: int, val_frac: float, device: str):
    n = len(y)
    n_val = max(1, int(round(n * val_frac)))
    if n - n_val < 1:
        raise ValueError("too few rows for an early-stopping split")
    model.to(device)
    xt, yt = _tensor(X[: n - n_val], device), _tensor(y[: n - n_val], device)
    xv, yv = _tensor(X[n - n_val:], device), _tensor(y[n - n_val:], device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def val_loss() -> float:
        model.eval()
        with torch.no_grad():
            v = float(nn.functional.mse_loss(model(xv), yv))
        if not np.isfinite(v):
            raise ValueError("non-finite validation loss")
        return v

    # The untrained weights count as epoch 0, so the result is never worse than the start.
    best, best_state, wait = val_loss(), copy.deepcopy(model.state_dict()), 0
    for _ in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(xt), yt)
        loss.backward()
        opt.step()
        v = val_loss()
        if v < best - 1e-12:
            best, best_state, wait = v, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model


def predict(model, X, device: str) -> np.ndarray:
    model.to(device).eval()
    with torch.no_grad():
        return model(_tensor(X, device)).float().cpu().numpy()
```

- [ ] **Step 5: Run tests**

Run: `$PY -m pytest tests/test_neural.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add src/fed_news/tft.py src/fed_news/neural.py tests/test_neural.py
git commit -m "Add neural forecasters (LSTM/GRU/TFT) and training loops for Plan 2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Neural walk-forward engine

**Files:**
- Create: `src/fed_news/neural_walk.py`
- Test: `tests/test_neural_walk.py`

**Interfaces:**
- Consumes: `neural.*` (Task 1); `walkforward.train_mask(n, i, task)`, `walkforward.prereg_ok()`; `models.choose_penalty`, `models.OrderedLogit`, `models.ridge_score`, `models.logloss_score`, `models.move_sizes`, `models.classes_from_change`, `models.expanding_folds`; `config.RIDGE_GRID`, `config.ORDLOGIT_GRID`, `config.REPO`; `panel.S_COLS`, `panel.N_COLS`.
- Produces: `neural_walk.NeuralSpec` (fields `name, task, arch, cols, lookback, target, hidden, lr, epochs, refit_every, early_stop=False, dropout=0.0, anchor=False`); `neural_walk.neural_ladder(task: str, paper_refit_every: int = 1) -> list[NeuralSpec]` with names T1 `lstm-S, gru-S, tft-S, hybrid-S, lstm-S+N, gru-S+N, tft-S+N, hybrid-S+N`, T2 `mlstm-S, tft-S, hybrid-S, mlstm-S+N, tft-S+N, hybrid-S+N`; `neural_walk.prereg_h2a_ok(repo=config.REPO) -> bool`; `neural_walk.standardise(X, train) -> np.ndarray`; `neural_walk.oof_residuals(Z, y_change, y_class, rows, task) -> tuple[np.ndarray, np.ndarray]`; `neural_walk.run_neural_spec(df, spec, seed, test_start, test_end=None, device="cuda", require_prereg=True, epochs=None) -> pd.DataFrame` with columns `date, y_true, y_pred, lag, class_true, class_pred`.

- [ ] **Step 1: Write the failing test `tests/test_neural_walk.py`**

```python
import numpy as np
import pandas as pd
import pytest

from fed_news import models, neural_walk as nw, panel


def synthetic(n=90, seed=0):
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


def spec(task, name):
    return {s.name: s for s in nw.neural_ladder(task)}[name]


def run(df, sp, start, end=None, **kw):
    return nw.run_neural_spec(df, sp, 0, start, end, device="cpu", require_prereg=False, epochs=kw.pop("epochs", 3), **kw)


def test_ladder_matches_preregistered_settings():
    t1 = {s.name: s for s in nw.neural_ladder("T1")}
    assert (t1["lstm-S"].lookback, t1["lstm-S"].hidden, t1["lstm-S"].lr, t1["lstm-S"].epochs, t1["lstm-S"].refit_every) == (12, 240, 1e-4, 300, 1)
    assert (t1["gru-S+N"].lookback, t1["gru-S+N"].hidden, t1["gru-S+N"].lr, t1["gru-S+N"].epochs) == (12, 188, 2e-4, 449)
    assert (t1["tft-S"].hidden, t1["tft-S"].dropout, t1["tft-S"].lookback, t1["tft-S"].refit_every, t1["tft-S"].target) == (32, 0.2, 12, 12, "change")
    assert t1["hybrid-S+N"].anchor and t1["hybrid-S+N"].cols == panel.S_COLS + panel.N_COLS
    t2 = {s.name: s for s in nw.neural_ladder("T2")}
    assert (t2["mlstm-S"].lookback, t2["mlstm-S"].refit_every, t2["mlstm-S"].target) == (4, 8, "level")
    assert (t2["tft-S+N"].lookback, t2["tft-S+N"].refit_every) == (8, 8)
    assert sorted(t2) == ["hybrid-S", "hybrid-S+N", "mlstm-S", "mlstm-S+N", "tft-S", "tft-S+N"]
    assert nw.neural_ladder("T1", paper_refit_every=3)[0].refit_every == 3


def test_one_forecast_per_test_row():
    df = synthetic()
    for task, name in (("T1", "lstm-S"), ("T2", "tft-S"), ("T2", "hybrid-S")):
        out = run(df, spec(task, name), "2001-01-01")
        assert len(out) == int((df.index >= "2001-01-01").sum())
        assert np.isfinite(out["y_pred"]).all()


def test_t1_forecast_ignores_rows_outside_training_window():
    df = synthetic()
    i = int(np.flatnonzero(df.index >= "2001-01-01")[0])
    day = df.index[i]
    base = run(df, spec("T1", "lstm-S+N"), day, day)
    pert = df.copy()
    pert.iloc[i - 1:i + 1, pert.columns.get_indexer(["y_level", "y_change"])] = 7.0
    pert.iloc[i - 1:i + 1, pert.columns.get_loc("y_class")] = 2
    pert.iloc[i + 1:, :] = pert.iloc[i + 1:, :] * -3 + 5
    again = run(pert, spec("T1", "lstm-S+N"), day, day)
    assert again["y_pred"].iloc[0] == base["y_pred"].iloc[0]


def test_t2_forecast_ignores_current_and_future_targets():
    df = synthetic()
    i = int(np.flatnonzero(df.index >= "2001-01-01")[0])
    day = df.index[i]
    for name in ("tft-S+N", "hybrid-S+N", "mlstm-S+N"):
        base = run(df, spec("T2", name), day, day)
        pert = df.copy()
        pert.iloc[i, pert.columns.get_indexer(["y_level", "y_change"])] = 7.0
        pert.iloc[i, pert.columns.get_loc("y_class")] = 2
        pert.iloc[i + 1:, :] = pert.iloc[i + 1:, :] * -3 + 5
        again = run(pert, spec("T2", name), day, day)
        assert again["y_pred"].iloc[0] == base["y_pred"].iloc[0], name


def test_refit_schedule_builds_one_model_per_block(monkeypatch):
    df = synthetic()
    calls = []
    real = nw.build_model
    monkeypatch.setattr(nw, "build_model", lambda *a, **k: calls.append(1) or real(*a, **k))
    n_test = int((df.index >= "2001-01-01").sum())   # 18
    run(df, spec("T2", "tft-S"), "2001-01-01")
    assert len(calls) == -(-n_test // 8)


def test_oof_residuals_come_from_anchors_fit_on_earlier_rows(monkeypatch):
    df = synthetic()
    X = df[panel.S_COLS].to_numpy(float)
    rows = np.arange(60)
    seen = []
    real = nw._anchor
    monkeypatch.setattr(nw, "_anchor", lambda Z, yc, ycl, r, task: seen.append(np.asarray(r)) or real(Z, yc, ycl, r, task))
    res_rows, res = nw.oof_residuals(X, df["y_change"].to_numpy(), df["y_class"].to_numpy(), rows, "T1")
    folds = [(tr, va) for tr, va in models.expanding_folds(60) if len(tr) >= nw.MIN_ANCHOR_ROWS]
    assert len(seen) == len(folds) > 0
    for used, (tr, va) in zip(seen, folds):
        assert used.max() < va.min()               # each anchor saw only rows before the ones it scores
    assert np.array_equal(res_rows, np.concatenate([va for _, va in folds]))
    assert len(res) == len(res_rows)


def test_hybrid_runs_at_earliest_refit():
    df = synthetic()
    out = run(df, spec("T2", "hybrid-S+N"), "1999-01-01", "1999-02-01", epochs=5)
    assert len(out) == 2 and np.isfinite(out["y_pred"]).all()


def test_rejects_non_finite_inputs():
    df = synthetic()
    df.iloc[5, df.columns.get_loc("n_epu")] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        run(df, spec("T1", "tft-S+N"), "2001-01-01")


def test_rejects_unsorted_index():
    df = synthetic().iloc[::-1]
    with pytest.raises(ValueError, match="chronological"):
        run(df, spec("T1", "tft-S"), "2001-01-01")


def test_text_spec_refused_without_h2a_tag(monkeypatch):
    monkeypatch.setattr(nw, "prereg_ok", lambda repo=None: True)
    monkeypatch.setattr(nw, "prereg_h2a_ok", lambda repo=None: False)
    with pytest.raises(PermissionError):
        nw.run_neural_spec(synthetic(), spec("T1", "lstm-S+N"), 0, "2001-01-01", device="cpu", epochs=1)
    out = nw.run_neural_spec(synthetic(), spec("T1", "lstm-S"), 0, "2002-05-01", device="cpu", epochs=1)
    assert len(out) == 2
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/test_neural_walk.py -v`
Expected: FAIL with `ImportError: cannot import name 'neural_walk'`

- [ ] **Step 3: Write `src/fed_news/neural_walk.py`**

```python
"""Walk-forward for the neural rows of the ladder (spec section 7, secondary architectures).

Same rows, cutoffs and training masks as the primary engine (walkforward.train_mask): T1 trains on
rows up to t-2, T2 on all earlier meetings. Features are standardised on each refit's training rows.
The paper's LSTM/GRU and the meeting LSTM predict the level; the TFT predicts the change; the hybrid
adds a TFT correction, trained on out-of-fold residuals, to the primary linear model's change.
Between refits the last model (and its standardisation) forecasts the following rows.
"""
import subprocess
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from fed_news import config, models
from fed_news.neural import build_model, fit_early_stopping, fit_fixed, make_sequences, predict
from fed_news.panel import N_COLS, S_COLS
from fed_news.walkforward import prereg_ok, train_mask

H2A_TAG = "prereg-h2a"
TFT_HIDDEN, TFT_DROPOUT, TFT_LR, TFT_MAX_EPOCHS, TFT_PATIENCE, TFT_VAL_FRAC = 32, 0.2, 1e-3, 300, 20, 0.2
MIN_ANCHOR_ROWS = 20
TARGET_COLS = ["y_level", "y_change", "y_class", "s_rate_lag"]


@dataclass
class NeuralSpec:
    name: str
    task: str
    arch: str
    cols: list
    lookback: int
    target: str
    hidden: int
    lr: float
    epochs: int
    refit_every: int
    early_stop: bool = False
    dropout: float = 0.0
    anchor: bool = False


def neural_ladder(task: str, paper_refit_every: int = 1) -> list[NeuralSpec]:
    out = []
    for label, cols in (("S", S_COLS), ("S+N", S_COLS + N_COLS)):
        if task == "T1":
            out += [
                NeuralSpec(f"lstm-{label}", "T1", "lstm", cols, 12, "level", 240, 1e-4, 300, paper_refit_every),
                NeuralSpec(f"gru-{label}", "T1", "gru", cols, 12, "level", 188, 2e-4, 449, paper_refit_every),
                NeuralSpec(f"tft-{label}", "T1", "tft", cols, 12, "change", TFT_HIDDEN, TFT_LR, TFT_MAX_EPOCHS, 12,
                           True, TFT_DROPOUT),
                NeuralSpec(f"hybrid-{label}", "T1", "tft", cols, 12, "change", TFT_HIDDEN, TFT_LR, TFT_MAX_EPOCHS, 12,
                           True, TFT_DROPOUT, True),
            ]
        else:
            out += [
                NeuralSpec(f"mlstm-{label}", "T2", "lstm", cols, 4, "level", 240, 1e-4, 300, 8),
                NeuralSpec(f"tft-{label}", "T2", "tft", cols, 8, "change", TFT_HIDDEN, TFT_LR, TFT_MAX_EPOCHS, 8,
                           True, TFT_DROPOUT),
                NeuralSpec(f"hybrid-{label}", "T2", "tft", cols, 8, "change", TFT_HIDDEN, TFT_LR, TFT_MAX_EPOCHS, 8,
                           True, TFT_DROPOUT, True),
            ]
    return out


def prereg_h2a_ok(repo=config.REPO) -> bool:
    out = subprocess.run(["git", "-C", str(repo), "tag", "-l", H2A_TAG], capture_output=True, text=True)
    return out.stdout.strip() == H2A_TAG


def standardise(X, train) -> np.ndarray:
    mu, sd = X[train].mean(0), X[train].std(0)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def _anchor(Z, y_change, y_class, rows, task):
    """Fit the primary linear model on `rows`; return a function giving its forecast of the change."""
    rows = np.asarray(rows)
    if task == "T1":
        alpha = models.choose_penalty(lambda a: Ridge(alpha=a), models.ridge_score, Z[rows], y_change[rows],
                                      config.RIDGE_GRID)
        fitted = Ridge(alpha=alpha).fit(Z[rows], y_change[rows])
        return lambda idx: fitted.predict(Z[idx])
    lam = models.choose_penalty(models.OrderedLogit, models.logloss_score, Z[rows], y_class[rows],
                                config.ORDLOGIT_GRID)
    fitted = models.OrderedLogit(lam).fit(Z[rows], y_class[rows])
    sizes = models.move_sizes(y_class[rows], y_change[rows])
    return lambda idx: fitted.predict_proba(Z[idx]) @ sizes


def oof_residuals(Z, y_change, y_class, rows, task) -> tuple[np.ndarray, np.ndarray]:
    """y_change minus the anchor's forecast, each row's anchor fit only on strictly earlier training rows."""
    rows = np.asarray(rows)
    res_rows, res = [], []
    for tr, va in models.expanding_folds(len(rows)):
        if len(tr) < MIN_ANCHOR_ROWS:
            continue
        f = _anchor(Z, y_change, y_class, rows[tr], task)
        res_rows.append(rows[va])
        res.append(y_change[rows[va]] - f(rows[va]))
    if not res_rows:
        return np.array([], dtype=int), np.array([], dtype=float)
    return np.concatenate(res_rows), np.concatenate(res)


def _check_inputs(df: pd.DataFrame, spec: NeuralSpec) -> None:
    if not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError("rows must be in chronological order with a unique index")
    values = df[list(dict.fromkeys(spec.cols + TARGET_COLS))].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("non-finite value in features or targets")


def run_neural_spec(df: pd.DataFrame, spec: NeuralSpec, seed: int, test_start, test_end=None, device: str = "cuda",
                    require_prereg: bool = True, epochs: int | None = None) -> pd.DataFrame:
    uses_text = any(c in N_COLS for c in spec.cols)
    if require_prereg and uses_text and not (prereg_ok() and prereg_h2a_ok()):
        raise PermissionError(f"{spec.name} uses news columns; tags 'prereg' and '{H2A_TAG}' must exist first")
    _check_inputs(df, spec)
    dates, n = df.index, len(df)
    in_window = dates >= pd.Timestamp(test_start)
    if test_end is not None:
        in_window &= dates <= pd.Timestamp(test_end)
    test = np.flatnonzero(in_window)
    X = df[spec.cols].to_numpy(float)
    y_level, y_change, lag = (df[c].to_numpy(float) for c in ("y_level", "y_change", "s_rate_lag"))
    y_class = df["y_class"].to_numpy(int)
    n_epochs = spec.epochs if epochs is None else epochs
    out, model, anchor_f, Z = [], None, None, None
    for k, i in enumerate(test):
        if k % spec.refit_every == 0:
            train = train_mask(n, i, spec.task)
            Z = standardise(X, train)
            rows = np.flatnonzero(train)
            if spec.anchor:
                anchor_f = _anchor(Z, y_change, y_class, rows, spec.task)
                fit_rows, target = oof_residuals(Z, y_change, y_class, rows, spec.task)
            else:
                fit_rows = rows
                target = (y_level if spec.target == "level" else y_change)[rows]
            keep = fit_rows >= spec.lookback - 1
            fit_rows, target = fit_rows[keep], target[keep]
            if len(fit_rows) < 2:
                raise ValueError(f"{spec.name}: fewer than 2 training sequences at {dates[i].date()}")
            Xs = make_sequences(Z, fit_rows, spec.lookback)
            model = build_model(spec.arch, len(spec.cols), spec.hidden, spec.dropout, seed)
            if spec.early_stop:
                fit_early_stopping(model, Xs, target, spec.lr, n_epochs, TFT_PATIENCE, TFT_VAL_FRAC, device)
            else:
                fit_fixed(model, Xs, target, spec.lr, n_epochs, device)
        raw = float(predict(model, make_sequences(Z, np.array([i]), spec.lookback), device)[0])
        if spec.target == "level":
            change = raw - lag[i]
        else:
            change = raw + (float(anchor_f(np.array([i]))[0]) if spec.anchor else 0.0)
        out.append({"date": dates[i], "y_true": y_level[i], "y_pred": lag[i] + change, "lag": lag[i],
                    "class_true": int(y_class[i]),
                    "class_pred": int(models.classes_from_change(np.array([change]))[0])})
    return pd.DataFrame(out)
```

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_neural_walk.py -v`
Expected: 10 passed (CPU; a few seconds each).

- [ ] **Step 5: Confirm Plan 1 files are untouched, run the full suite, commit**

```bash
git diff prereg -- src/fed_news/walkforward.py src/fed_news/evaluate.py src/fed_news/run_models.py src/fed_news/models.py
$PY -m pytest -q
git add src/fed_news/neural_walk.py tests/test_neural_walk.py
git commit -m "Add neural walk-forward engine with hybrid out-of-fold anchor

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
Expected: the `git diff` prints nothing; the suite passes.

---

### Task 3: Runner, seeds and ensembles

**Files:**
- Create: `src/fed_news/run_neural.py`
- Test: `tests/test_run_neural.py`

**Interfaces:**
- Consumes: `neural_walk.neural_ladder`, `neural_walk.run_neural_spec`; `run_models.load_task`; `models.classes_from_change`; `config.RESULTS`, `config.TEST_START`.
- Produces: `run_neural.OUT` (`config.RESULTS / "h2a"`); `run_neural.SEEDS = tuple(range(8))`; `run_neural.seed_path(task, name, seed) -> Path`; `run_neural.ensemble_path(task, name) -> Path`; `run_neural.ensemble(frames: list[pd.DataFrame]) -> pd.DataFrame`; `run_neural.too_slow(seconds: float) -> bool` (> 3600); `run_neural.main(argv=None)` with options `--task T1|T2|all`, `--only name,name`, `--seeds 0-7`, `--paper-refit-every N`, `--device cuda|cpu`; writes `results/h2a/run_log.json` (`"<task>/<name>/seed<k>": {"seconds", "refit_every"}`).

- [ ] **Step 1: Write the failing test `tests/test_run_neural.py`**

```python
import json

import numpy as np
import pandas as pd
import pytest

from fed_news import run_neural


def preds(dates, y_pred, lag=2.0):
    return pd.DataFrame({"date": pd.to_datetime(dates), "y_true": 2.0, "y_pred": y_pred, "lag": lag,
                         "class_true": 1, "class_pred": 1})


def test_ensemble_is_seed_mean_with_band_class():
    d = ["2001-01-01", "2001-02-01"]
    out = run_neural.ensemble([preds(d, [2.0, 2.3]), preds(d, [2.2, 2.3])])
    assert out["y_pred"].tolist() == pytest.approx([2.1, 2.3])
    assert out["class_pred"].tolist() == [1, 2]       # changes 0.1 (hold) and 0.3 (> 0.125, hike)


def test_ensemble_refuses_mismatched_dates():
    with pytest.raises(ValueError, match="dates"):
        run_neural.ensemble([preds(["2001-01-01"], [2.0]), preds(["2001-02-01"], [2.0])])


def test_too_slow_is_more_than_one_hour():
    assert run_neural.too_slow(3600.1) and not run_neural.too_slow(3600.0)


def test_main_runs_selected_specs_resumes_and_ensembles(tmp_path, monkeypatch):
    monkeypatch.setattr(run_neural, "OUT", tmp_path)
    idx = pd.date_range("2001-01-01", periods=3, freq="MS")
    monkeypatch.setattr(run_neural, "load_task", lambda task: pd.DataFrame(index=idx))
    calls = []

    def fake(df, spec, seed, start, device):
        calls.append((spec.name, seed))
        return preds(idx, [2.0 + 0.01 * seed] * 3)

    monkeypatch.setattr(run_neural, "run_neural_spec", lambda df, spec, seed, start, device="cuda": fake(df, spec, seed, start, device))
    run_neural.main(["--task", "T2", "--only", "tft-S", "--seeds", "0-7", "--device", "cpu"])
    assert len(calls) == 8
    assert run_neural.ensemble_path("T2", "tft-S").exists()
    log = json.loads((tmp_path / "run_log.json").read_text())
    assert log["T2/tft-S/seed3"]["refit_every"] == 8
    run_neural.main(["--task", "T2", "--only", "tft-S", "--seeds", "0-7", "--device", "cpu"])
    assert len(calls) == 8                                 # resumed: nothing rerun


def test_no_ensemble_until_all_eight_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(run_neural, "OUT", tmp_path)
    idx = pd.date_range("2001-01-01", periods=2, freq="MS")
    monkeypatch.setattr(run_neural, "load_task", lambda task: pd.DataFrame(index=idx))
    monkeypatch.setattr(run_neural, "run_neural_spec", lambda df, spec, seed, start, device="cuda": preds(idx, [2.0, 2.0]))
    run_neural.main(["--task", "T1", "--only", "lstm-S", "--seeds", "0-2", "--device", "cpu"])
    assert not run_neural.ensemble_path("T1", "lstm-S").exists()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/test_run_neural.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_neural'`

- [ ] **Step 3: Write `src/fed_news/run_neural.py`**

```python
"""Run the Plan 2 neural ladder: every spec x 8 seeds (resumable), then seed-mean ensembles.

Per-seed predictions: results/h2a/preds_<task>_<name>_seed<k>.csv; ensembles (mean of the 8 seeds'
forecasts, class from the +/-0.125 band): results/h2a/preds_<task>_<name>.csv; timings and refit
schedule: results/h2a/run_log.json (the one-hour rule for the paper's LSTM/GRU is read from it).
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from fed_news import config, models
from fed_news.neural_walk import neural_ladder, run_neural_spec
from fed_news.run_models import load_task

OUT = config.RESULTS / "h2a"
SEEDS = tuple(range(8))
ONE_HOUR = 3600.0


def seed_path(task: str, name: str, seed: int):
    return OUT / f"preds_{task}_{name}_seed{seed}.csv"


def ensemble_path(task: str, name: str):
    return OUT / f"preds_{task}_{name}.csv"


def ensemble(frames: list[pd.DataFrame]) -> pd.DataFrame:
    base = frames[0][["date", "y_true", "lag", "class_true"]].copy()
    for f in frames[1:]:
        if len(f) != len(base) or not (f["date"].to_numpy() == base["date"].to_numpy()).all():
            raise ValueError("seed files cover different dates")
    base["y_pred"] = np.mean([f["y_pred"].to_numpy(float) for f in frames], axis=0)
    base["class_pred"] = models.classes_from_change(base["y_pred"].to_numpy() - base["lag"].to_numpy(float))
    return base[["date", "y_true", "y_pred", "lag", "class_true", "class_pred"]]


def too_slow(seconds: float) -> bool:
    return seconds > ONE_HOUR


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["T1", "T2", "all"], default="all")
    ap.add_argument("--only", default="")
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--paper-refit-every", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    lo, hi = (int(x) for x in args.seeds.split("-"))
    only = {s for s in args.only.split(",") if s}
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "run_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    for task in (["T1", "T2"] if args.task == "all" else [args.task]):
        df = load_task(task)
        for spec in neural_ladder(task, args.paper_refit_every):
            if only and spec.name not in only:
                continue
            for seed in range(lo, hi + 1):
                path = seed_path(task, spec.name, seed)
                if path.exists():
                    continue
                t0 = time.time()
                preds = run_neural_spec(df, spec, seed, config.TEST_START, device=args.device)
                seconds = time.time() - t0
                preds.to_csv(path, index=False)
                log[f"{task}/{spec.name}/seed{seed}"] = {"seconds": round(seconds, 1), "refit_every": spec.refit_every}
                log_path.write_text(json.dumps(log, indent=1))
                flag = "  SLOWER THAN ONE HOUR" if too_slow(seconds) else ""
                print(f"{task} {spec.name} seed {seed}: {len(preds)} forecasts, {seconds:.0f}s{flag}", flush=True)
            files = [seed_path(task, spec.name, s) for s in SEEDS]
            if all(p.exists() for p in files):
                frames = [pd.read_csv(p, parse_dates=["date"]) for p in files]
                ensemble(frames).to_csv(ensemble_path(task, spec.name), index=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_run_neural.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/fed_news/run_neural.py tests/test_run_neural.py
git commit -m "Add Plan 2 runner: seeds, resume, run log, seed-mean ensembles

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Pre-register the Plan 2 settings, tag, and time the paper LSTM

**Files:**
- Create: `PREREG_H2A.md`

**Interfaces:**
- Consumes: Tasks 1-3; the settings in Global Constraints.
- Produces: git tag `prereg-h2a`; `results/h2a/run_log.json` entry for `T1/lstm-S/seed0`; the decision `paper_refit_every` = 1 or 3.

- [ ] **Step 1: Write `PREREG_H2A.md`** (exact content, filling only the commit hash)

```markdown
# Pre-registration addendum: Plan 2 neural rows (H2a)

Date: 2026-10-09. Code commit: <output of `git rev-parse HEAD` before this commit>.
Binding: spec sections 3 (H2a), 7 (Secondary architectures), 8 (variance rule) and PREREG.md (tag `prereg`).
This file fixes the settings the spec leaves open, before any neural forecast that uses news.

## Settings
- Paper LSTM (T1): 12 steps, 240 units, 1 layer, no dropout, Adam lr 1e-4, 300 epochs, batch 32 shuffled per epoch (seeded), MSE on the standardised target, predicts the level, refit every month.
- Paper GRU (T1): 12 steps, 188 units, 1 layer, no dropout, Adam lr 2e-4, 449 epochs, batch 32 shuffled per epoch (seeded), MSE on the standardised target, predicts the level, refit every month.
- One-hour rule: `T1/lstm-S/seed0` is timed first, then (only if it took at most 3,600 s) `T1/gru-S/seed0`; if either takes longer than 3,600 s, all paper LSTM/GRU T1 specs are run with refit every 3 months instead, recorded in DEVIATIONS.md.
- Meeting LSTM (T2): lookback 4 meetings, refit every 8 meetings, otherwise as the paper LSTM.
- TFT: CompactTFT (gold_gpt2/src/tft.py, vendored unchanged), hidden 32, 4 heads, dropout 0.2, lookback 12 months / 8 meetings, Adam lr 1e-3, full batch, at most 300 epochs, early stopping on the chronologically last 20% of training rows with patience 20 (best epoch kept; the untrained weights count as epoch 0), refit every 12 months / 8 meetings, predicts the change.
- Hybrid: anchor = Plan 1 primary model (T1 ridge on the change, T2 penalised ordered logit expected change, penalties by 5-fold expanding CV), refitted at every test row as in Plan 1; TFT (refitted on its schedule) trained on out-of-fold anchor residuals (5 expanding folds inside the training window, folds whose training part has fewer than 20 rows skipped); forecast = lag + anchor change + TFT correction.
- Training rows: walkforward.train_mask (T1 rows <= t-2, T2 earlier meetings); features standardised on each refit's training rows; the model and standardisation of the last refit forecast the rows until the next refit.
- All neural targets standardised with the fit rows' mean and sd and inverted at prediction. Seeds 0-7 per spec (initialisation and batch order); ensemble = mean of the 8 seeds' forecasts; classes from the +/-0.125 band.

## Tests and decision rule (secondary; cannot change the H1 verdict)
- Per architecture (T1: lstm, gru, tft, hybrid; T2: mlstm, tft, hybrid), S+N vs S on the seed-mean ensembles: one-sided DM-HLN on squared error (T1, T2); exact one-sided McNemar on correct classes (T2).
- Variance rule: a gain counts only if mean RMSE(S) - mean RMSE(S+N) > 2 x max(sd RMSE(S), sd RMSE(S+N)) over the 8 seeds (ddof 1).
- Holm adjustment across all H2 p-values (Plan 1 results/h2_pvalues.csv plus these); the final family is fixed after Plan 3.
- Reported for every spec: per-seed mean +/- sd and ensemble RMSE, MAE, accuracy; the paper LSTM/GRU replication also on the paper window (Jan 2016 - Apr 2025) next to the paper's published values.

## Frozen-model analysis (descriptive, as the paper's section 4.1 fixed-rule exercise)
- Two cutoffs: train once on everything available before 2016-01-01 (T1: rows up to 2015-11, the t-2 rule; T2: meetings before 2016) and forecast every later month/meeting to Sep 2026 with no updating; the same with 2020-01-01.
- Models: "no change" (random walk / always-hold); the Plan 1 linear primary model with S and with S+N; every Plan 2 neural spec (T1 lstm, gru, tft, hybrid; T2 mlstm, tft, hybrid) with S and S+N, 8 seeds, seed-mean ensembles; neural settings as above except that the model is fitted once.
- Reported: RMSE, MAE, accuracy for each, next to the same model's walk-forward (updated) forecasts over the same months, and a one-sided DM p-value against "no change". These are descriptive: they are not part of the H2 Holm family and cannot change any verdict.
```

- [ ] **Step 2: Commit and tag**

```bash
git add PREREG_H2A.md
git commit -m "Pre-register Plan 2 neural settings and decision rule

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git tag prereg-h2a
git tag -l prereg-h2a
```
Expected: `prereg-h2a`.

- [ ] **Step 3: Time one paper-LSTM seed-pass (no news columns, GPU)**

```bash
PYTHONPATH=src $PY -m fed_news.run_neural --task T1 --only lstm-S --seeds 0-0 --device cuda
PYTHONPATH=src $PY -c "import json; print(json.load(open('results/h2a/run_log.json'))['T1/lstm-S/seed0'])"
```
Expected: one line `T1 lstm-S seed 0: 308 forecasts, <N>s`. If N > 3600, the one-hour rule applies: delete `results/h2a/preds_T1_lstm-S_seed0.csv`, append to `DEVIATIONS.md` the line `- Plan 2: T1/lstm-S/seed0 took <N> s (> 3,600 s); per PREREG_H2A.md all paper LSTM/GRU T1 specs run with refit every 3 months.`, commit it, and use `--paper-refit-every 3` for every later T1 run. Record N and the decision in the task report.

---

### Task 5: Full neural runs (GPU, background)

**Files:**
- Output only: `results/h2a/preds_*.csv`, `results/h2a/run_log.json`, `logs/run_neural.log`

**Interfaces:**
- Consumes: Task 4's tag and refit decision.
- Produces: 8 seed files and 1 ensemble file for each of the 14 specs (T1: 8, T2: 6).

- [ ] **Step 1: Launch (background; resumes from any finished seed files)**

```bash
mkdir -p logs
PYTHONPATH=src nohup $PY -u -m fed_news.run_neural --task all --device cuda --paper-refit-every <1 or 3 from Task 4> > logs/run_neural.log 2>&1 &
```

- [ ] **Step 2: Verify completion**

When the process has exited:
```bash
ls results/h2a/preds_T1_*_seed*.csv | wc -l     # 64
ls results/h2a/preds_T2_*_seed*.csv | wc -l     # 48
ls results/h2a/preds_T?_*.csv | grep -v seed | wc -l   # 14 ensembles
grep -i -E "error|traceback|nan" logs/run_neural.log | head
```
Expected: 64, 48, 14, and no errors. If the process died, relaunch the same command (finished seeds are skipped). If any T1 paper LSTM/GRU seed line says "SLOWER THAN ONE HOUR" while refit-every was 1, stop and apply the one-hour rule as in Task 4 Step 3 (delete those specs' seed files, record the deviation, relaunch with `--paper-refit-every 3`).

- [ ] **Step 3: Commit the predictions**

```bash
git add results/h2a/
git commit -m "Record Plan 2 neural predictions (14 specs x 8 seeds)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Evaluation and RESULTS_H2A.md

**Files:**
- Create: `src/fed_news/evaluate_h2a.py`
- Test: `tests/test_evaluate_h2a.py`

**Interfaces:**
- Consumes: `run_neural.OUT`, `run_neural.SEEDS`, `run_neural.seed_path`, `run_neural.ensemble_path` (always through the module, `run_neural.<name>`, so tests can redirect `OUT`); `evaluate.metrics(preds) -> dict` and `evaluate.PAPER_ROWS` (read-only import); `stats_tests.dm_hln`, `stats_tests.mcnemar_greater`, `stats_tests.holm`; `config.RESULTS`, `config.REPO`, `config.PAPER_T1`.
- Produces: `evaluate_h2a.ARCHS = {"T1": ["lstm", "gru", "tft", "hybrid"], "T2": ["mlstm", "tft", "hybrid"]}`; `evaluate_h2a.variance_rule(rmse_base, rmse_new) -> bool`; `evaluate_h2a.pair_tests(task, arch) -> dict[str, float]`; `evaluate_h2a.main()` writing `results/h2a/seed_metrics.csv`, `results/h2a/ensemble_metrics.csv`, `results/h2a/tests.csv` and `RESULTS_H2A.md`.

- [ ] **Step 1: Write the failing test `tests/test_evaluate_h2a.py`**

```python
import numpy as np
import pandas as pd
import pytest

from fed_news import config, evaluate_h2a as ev, run_neural


def test_variance_rule():
    assert ev.variance_rule([0.20, 0.21, 0.19, 0.20], [0.15, 0.16, 0.14, 0.15])
    assert not ev.variance_rule([0.20, 0.25, 0.15, 0.20], [0.18, 0.23, 0.13, 0.18])


def fake_preds(dates, err, cls_ok):
    y = np.full(len(dates), 2.0)
    return pd.DataFrame({"date": dates, "y_true": y, "y_pred": y + err, "lag": 2.0,
                         "class_true": 1, "class_pred": np.where(cls_ok, 1, 2)})


def write_all(tmp_path, rng):
    for task, archs in ev.ARCHS.items():
        dates = pd.date_range("2016-01-01", periods=40, freq="MS")   # inside the paper window
        for arch in archs:
            for info, scale in (("S", 0.3), ("S+N", 0.1)):
                for s in run_neural.SEEDS:
                    err = rng.normal(scale=scale, size=40)
                    fake_preds(dates, err, rng.random(40) < (0.5 if info == "S" else 0.9)).to_csv(
                        run_neural.seed_path(task, f"{arch}-{info}", s), index=False)
                err = rng.normal(scale=scale, size=40)
                fake_preds(dates, err, rng.random(40) < (0.5 if info == "S" else 0.9)).to_csv(
                    run_neural.ensemble_path(task, f"{arch}-{info}"), index=False)


def test_pair_tests_t2_has_dm_and_mcnemar(tmp_path, monkeypatch):
    monkeypatch.setattr(run_neural, "OUT", tmp_path)
    write_all(tmp_path, np.random.default_rng(0))
    out = ev.pair_tests("T2", "tft")
    assert set(out) == {"H2a T2 tft DM", "H2a T2 tft McNemar"}
    assert all(0 <= p <= 1 for p in out.values())
    assert set(ev.pair_tests("T1", "lstm")) == {"H2a T1 lstm DM"}


def test_main_writes_tables_and_report(tmp_path, monkeypatch):
    monkeypatch.setattr(run_neural, "OUT", tmp_path)
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    monkeypatch.setattr(config, "REPO", tmp_path)
    write_all(tmp_path, np.random.default_rng(1))
    pd.DataFrame({"hypothesis": ["H2c T1 S+N vs rw"], "p": [0.05], "holm_within_plan1": [0.05]}).to_csv(
        tmp_path / "h2_pvalues.csv", index=False)
    ev.main()
    for name in ("seed_metrics.csv", "ensemble_metrics.csv", "tests.csv"):
        assert (tmp_path / name).exists()
    tests = pd.read_csv(tmp_path / "tests.csv")
    assert {"hypothesis", "p", "holm_h2_so_far", "variance_rule"} <= set(tests.columns)
    assert "H2c T1 S+N vs rw" in set(tests["hypothesis"])
    report = (tmp_path / "RESULTS_H2A.md").read_text(encoding="utf-8")
    assert "Holm" in report and "variance rule" in report.lower()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/test_evaluate_h2a.py -v`
Expected: FAIL with `ImportError: cannot import name 'evaluate_h2a'`

- [ ] **Step 3: Write `src/fed_news/evaluate_h2a.py`**

```python
"""Plan 2 evaluation (spec sections 3 H2a and 8): metrics, the variance rule, tests, Holm, RESULTS_H2A.md.

Tests compare S+N with S for each architecture on the seed-mean ensembles (one-sided DM-HLN; McNemar
for T2). The variance rule uses the per-seed RMSEs. Holm runs across every H2 p-value so far (Plan 1
plus these); the final family is fixed after Plan 3. Nothing here touches the H1 verdict.
"""
import numpy as np
import pandas as pd

from fed_news import config, run_neural
from fed_news.evaluate import PAPER_ROWS, metrics
from fed_news.stats_tests import dm_hln, holm, mcnemar_greater

ARCHS = {"T1": ["lstm", "gru", "tft", "hybrid"], "T2": ["mlstm", "tft", "hybrid"]}
INFOS = ("S", "S+N")


def variance_rule(rmse_base, rmse_new) -> bool:
    base, new = np.asarray(rmse_base, float), np.asarray(rmse_new, float)
    return bool(base.mean() - new.mean() > 2 * max(base.std(ddof=1), new.std(ddof=1)))


def _load(path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def _seed_rows(task, arch) -> list[dict]:
    rows = []
    for info in INFOS:
        for s in run_neural.SEEDS:
            rows.append({"task": task, "arch": arch, "info": info, "seed": s,
                         **metrics(_load(run_neural.seed_path(task, f"{arch}-{info}", s)))})
    return rows


def pair_tests(task, arch) -> dict:
    base = _load(run_neural.ensemble_path(task, f"{arch}-S"))
    new = _load(run_neural.ensemble_path(task, f"{arch}-S+N"))
    if len(base) != len(new) or not (base["date"].to_numpy() == new["date"].to_numpy()).all():
        raise ValueError(f"prediction dates differ: {task} {arch}")
    sq = lambda p: ((p["y_pred"] - p["y_true"]) ** 2).to_numpy()
    out = {f"H2a {task} {arch} DM": dm_hln(sq(base), sq(new))[1]}
    if task == "T2":
        ok = lambda p: (p["class_pred"] == p["class_true"]).to_numpy()
        out[f"H2a {task} {arch} McNemar"] = mcnemar_greater(ok(base), ok(new))[2]
    return out


def main() -> None:
    out_dir = run_neural.OUT
    seeds = pd.DataFrame([r for t, archs in ARCHS.items() for a in archs for r in _seed_rows(t, a)])
    seeds.to_csv(out_dir / "seed_metrics.csv", index=False)
    summary = (seeds.groupby(["task", "arch", "info"])[["rmse", "mae", "accuracy"]]
               .agg(["mean", "std"]).round(4))
    summary.columns = [f"{m} {s}" for m, s in summary.columns]
    summary = summary.reset_index()

    ens_rows, paper_rows = [], []
    for task, archs in ARCHS.items():
        for arch in archs:
            for info in INFOS:
                p = _load(run_neural.ensemble_path(task, f"{arch}-{info}"))
                ens_rows.append({"task": task, "arch": arch, "info": info, **metrics(p)})
                if task == "T1" and arch in ("lstm", "gru"):
                    w = p[(p["date"] >= config.PAPER_T1[0]) & (p["date"] <= config.PAPER_T1[1])]
                    paper_rows.append({"model": f"our {arch.upper()} replication, {info}", **metrics(w)})
    ens = pd.DataFrame(ens_rows)
    ens.to_csv(out_dir / "ensemble_metrics.csv", index=False)
    paper = pd.concat([pd.DataFrame(PAPER_ROWS, columns=["model", "mse", "rmse", "mae", "mbe"]),
                       pd.DataFrame(paper_rows)[["model", "mse", "rmse", "mae", "mbe"]]], ignore_index=True)

    h2a, rule = {}, {}
    for task, archs in ARCHS.items():
        for arch in archs:
            tests = pair_tests(task, arch)
            h2a.update(tests)
            s = seeds[(seeds.task == task) & (seeds.arch == arch)]
            passed = variance_rule(s[s.info == "S"]["rmse"], s[s.info == "S+N"]["rmse"])
            rule.update({k: passed for k in tests})
    plan1 = pd.read_csv(config.RESULTS / "h2_pvalues.csv")
    family = dict(zip(plan1["hypothesis"], plan1["p"]))
    family.update(h2a)
    adjusted = holm(family)
    tests = pd.DataFrame({"hypothesis": list(family), "p": list(family.values()),
                          "holm_h2_so_far": [adjusted[k] for k in family],
                          "variance_rule": [rule.get(k, "") for k in family]})
    tests.to_csv(out_dir / "tests.csv", index=False)

    lines = [
        "# Results: Plan 2 (H2a, neural architectures)", "",
        "Secondary analyses; they cannot change the H1 verdict in RESULTS.md. "
        "Settings: PREREG_H2A.md (tag prereg-h2a).", "",
        "## Ensemble metrics (mean of 8 seeds' forecasts), test windows from 2001", "",
        ens.to_markdown(index=False, floatfmt=".4f"), "",
        "## Per-seed metrics, mean and standard deviation over 8 seeds", "",
        summary.to_markdown(index=False, floatfmt=".4f"), "",
        "## Paper replication on the paper's window (Jan 2016 - Apr 2025), T1", "",
        paper.to_markdown(index=False, floatfmt=".4f"), "",
        "## Tests: S+N vs S per architecture (one-sided), Holm across all H2 so far", "",
        "The variance rule: a gain counts only if mean RMSE(S) - mean RMSE(S+N) exceeds twice the "
        "larger seed standard deviation. The final Holm family is fixed after Plan 3.", "",
        tests.to_markdown(index=False, floatfmt=".4f"), "",
    ]
    (config.REPO / "RESULTS_H2A.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:4]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_evaluate_h2a.py -v`
Expected: 3 passed

- [ ] **Step 5: Evaluate the real runs, run the full suite, commit**

```bash
PYTHONPATH=src $PY -m fed_news.evaluate_h2a
$PY -m pytest -q
git diff prereg -- src/fed_news/walkforward.py src/fed_news/evaluate.py src/fed_news/run_models.py src/fed_news/models.py
git add src/fed_news/evaluate_h2a.py tests/test_evaluate_h2a.py RESULTS_H2A.md results/h2a/seed_metrics.csv results/h2a/ensemble_metrics.csv results/h2a/tests.csv
git commit -m "Evaluate Plan 2 neural rows and record RESULTS_H2A

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
Expected: `RESULTS_H2A.md` written; the suite passes; the `git diff` prints nothing.

---

### Task 7: Frozen-model analysis (train once, never update)

**Files:**
- Create: `src/fed_news/frozen.py`
- Test: `tests/test_frozen.py`

**Interfaces:**
- Consumes: `neural_walk.neural_ladder`, `neural_walk.run_neural_spec`, `neural_walk.standardise`, `neural_walk.prereg_h2a_ok`; `walkforward.train_mask`, `walkforward.prereg_ok`; `models.choose_penalty`, `models.OrderedLogit`, `models.ridge_score`, `models.logloss_score`, `models.move_sizes`, `models.classes_from_change`; `run_neural.SEEDS`, `run_neural.ensemble`, `run_neural.ensemble_path`; `run_models.load_task`; `evaluate.metrics`; `stats_tests.dm_hln`; `config.RESULTS`, `config.REPO`, `config.RIDGE_GRID`, `config.ORDLOGIT_GRID`.
- Produces: `frozen.STARTS = ("2016-01-01", "2020-01-01")`; `frozen.NEVER = 10**9`; `frozen.OUT` (`config.RESULTS / "frozen"`); `frozen.frozen_linear(df, cols, task, test_start) -> pd.DataFrame`; `frozen.frozen_rw(df, test_start) -> pd.DataFrame`; `frozen.run_all(device="cuda", seeds=run_neural.SEEDS)`; `frozen.report() -> pd.DataFrame` (writes `results/frozen/metrics.csv` and `RESULTS_FROZEN.md`); CLI `python -m fed_news.frozen [--device cuda] [--report-only]`. Prediction files: `results/frozen/preds_<task>_<model>_<yyyy>[_seed<k>].csv`, `<model>` in `rw`, `linear-S`, `linear-S+N` or a neural spec name, `<yyyy>` = 2016 or 2020.

- [ ] **Step 1: Write the failing test `tests/test_frozen.py`**

```python
import numpy as np
import pandas as pd

from fed_news import config, frozen, neural_walk as nw, panel, run_neural


def synthetic(n=90, seed=0):
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


def test_frozen_linear_never_sees_test_period_targets(monkeypatch):
    monkeypatch.setattr(frozen, "prereg_ok", lambda repo=None: True)
    monkeypatch.setattr(frozen, "prereg_h2a_ok", lambda repo=None: True)
    df = synthetic()
    start = "2000-01-01"
    i0 = int(np.flatnonzero(df.index >= start)[0])
    for task in ("T1", "T2"):
        first_unseen = i0 - 1 if task == "T1" else i0   # T1 trains on rows <= i0-2, T2 on rows < i0
        base = frozen.frozen_linear(df, panel.S_COLS + panel.N_COLS, task, start)
        pert = df.copy()
        cols = pert.columns.get_indexer(["y_level", "y_change", "y_class"])
        pert.iloc[first_unseen:, cols] = pert.iloc[first_unseen:, cols] * -3 + 5
        again = frozen.frozen_linear(pert, panel.S_COLS + panel.N_COLS, task, start)
        assert len(base) == int((df.index >= start).sum())
        assert np.allclose(base["y_pred"], again["y_pred"])


def test_frozen_rw_predicts_no_change():
    out = frozen.frozen_rw(synthetic(), "2001-01-01")
    assert (out["y_pred"] == out["lag"]).all() and (out["class_pred"] == 1).all()


def test_frozen_neural_fits_exactly_once(monkeypatch):
    df = synthetic()
    calls = []
    real = nw.build_model
    monkeypatch.setattr(nw, "build_model", lambda *a, **k: calls.append(1) or real(*a, **k))
    sp = {s.name: s for s in nw.neural_ladder("T1")}["tft-S"]
    from dataclasses import replace
    out = nw.run_neural_spec(df, replace(sp, refit_every=frozen.NEVER), 0, "2000-01-01", device="cpu",
                             require_prereg=False, epochs=3)
    assert len(calls) == 1 and len(out) == int((df.index >= "2000-01-01").sum())


def test_run_all_and_report_with_fakes(tmp_path, monkeypatch):
    monkeypatch.setattr(frozen, "OUT", tmp_path / "frozen")
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    monkeypatch.setattr(config, "REPO", tmp_path)
    monkeypatch.setattr(run_neural, "OUT", tmp_path / "h2a")
    df = synthetic(n=400)
    monkeypatch.setattr(frozen, "load_task", lambda task: df)
    monkeypatch.setattr(frozen, "prereg_ok", lambda repo=None: True)
    monkeypatch.setattr(frozen, "prereg_h2a_ok", lambda repo=None: True)

    def fake_neural(d, spec, seed, start, device="cuda"):
        rw = frozen.frozen_rw(d, start)
        rw["y_pred"] = rw["y_pred"] + 0.01 * seed
        return rw

    monkeypatch.setattr(frozen, "run_neural_spec", fake_neural)
    frozen.run_all(device="cpu", seeds=run_neural.SEEDS)
    for task in ("T1", "T2"):
        for yyyy in ("2016", "2020"):
            assert (tmp_path / "frozen" / f"preds_{task}_linear-S+N_{yyyy}.csv").exists()
            assert (tmp_path / "frozen" / f"preds_{task}_tft-S_{yyyy}.csv").exists()
    table = frozen.report()
    assert {"task", "start", "model", "rmse", "mae", "accuracy", "p_vs_no_change"} <= set(table.columns)
    assert (tmp_path / "RESULTS_FROZEN.md").exists()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/test_frozen.py -v`
Expected: FAIL with `ImportError: cannot import name 'frozen'`

- [ ] **Step 3: Write `src/fed_news/frozen.py`**

```python
"""Frozen-model analysis (PREREG_H2A.md; the paper's section 4.1 fixed-rule exercise).

Every model is trained once on what was available before 2016-01-01 (or 2020-01-01) and then forecasts
every later month or meeting without updating. Compared with the same models refitted every month in
the main analysis, this shows whether rules learned before a date still hold afterwards. Descriptive:
not part of the H2 Holm family.
"""
import argparse
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from fed_news import config, models, run_neural
from fed_news.evaluate import metrics
from fed_news.neural_walk import neural_ladder, prereg_h2a_ok, run_neural_spec, standardise
from fed_news.panel import N_COLS, S_COLS
from fed_news.run_models import load_task
from fed_news.stats_tests import dm_hln
from fed_news.walkforward import prereg_ok, train_mask

STARTS = ("2016-01-01", "2020-01-01")
NEVER = 10 ** 9
OUT = config.RESULTS / "frozen"
COLUMNS = ["date", "y_true", "y_pred", "lag", "class_true", "class_pred"]


def _arrays(df):
    return (df["y_level"].to_numpy(float), df["y_change"].to_numpy(float), df["y_class"].to_numpy(int),
            df["s_rate_lag"].to_numpy(float))


def _frame(df, test, y_pred, class_pred) -> pd.DataFrame:
    y_level, _, y_class, lag = _arrays(df)
    return pd.DataFrame({"date": df.index[test], "y_true": y_level[test], "y_pred": y_pred, "lag": lag[test],
                         "class_true": y_class[test], "class_pred": class_pred})[COLUMNS]


def frozen_rw(df, test_start) -> pd.DataFrame:
    test = np.flatnonzero(df.index >= pd.Timestamp(test_start))
    lag = df["s_rate_lag"].to_numpy(float)
    return _frame(df, test, lag[test], np.ones(len(test), dtype=int))


def frozen_linear(df, cols, task, test_start) -> pd.DataFrame:
    if any(c in N_COLS for c in cols) and not (prereg_ok() and prereg_h2a_ok()):
        raise PermissionError("news columns need tags 'prereg' and 'prereg-h2a'")
    test = np.flatnonzero(df.index >= pd.Timestamp(test_start))
    train = train_mask(len(df), test[0], task)
    rows = np.flatnonzero(train)
    Z = standardise(df[cols].to_numpy(float), train)
    _, y_change, y_class, lag = _arrays(df)
    if task == "T1":
        alpha = models.choose_penalty(lambda a: Ridge(alpha=a), models.ridge_score, Z[rows], y_change[rows],
                                      config.RIDGE_GRID)
        change = Ridge(alpha=alpha).fit(Z[rows], y_change[rows]).predict(Z[test])
        cls = models.classes_from_change(change)
    else:
        lam = models.choose_penalty(models.OrderedLogit, models.logloss_score, Z[rows], y_class[rows],
                                    config.ORDLOGIT_GRID)
        proba = models.OrderedLogit(lam).fit(Z[rows], y_class[rows]).predict_proba(Z[test])
        change = proba @ models.move_sizes(y_class[rows], y_change[rows])
        cls = proba.argmax(axis=1)
    return _frame(df, test, lag[test] + change, cls)


def _path(task, model, start, seed=None):
    suffix = "" if seed is None else f"_seed{seed}"
    return OUT / f"preds_{task}_{model}_{start[:4]}{suffix}.csv"


def run_all(device: str = "cuda", seeds=run_neural.SEEDS) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for task in ("T1", "T2"):
        df = load_task(task)
        for start in STARTS:
            frozen_rw(df, start).to_csv(_path(task, "rw", start), index=False)
            for label, cols in (("S", S_COLS), ("S+N", S_COLS + N_COLS)):
                frozen_linear(df, cols, task, start).to_csv(_path(task, f"linear-{label}", start), index=False)
            for spec in neural_ladder(task):
                for seed in seeds:
                    path = _path(task, spec.name, start, seed)
                    if not path.exists():
                        run_neural_spec(df, replace(spec, refit_every=NEVER), seed, start,
                                        device=device).to_csv(path, index=False)
                files = [_path(task, spec.name, start, s) for s in run_neural.SEEDS]
                if all(p.exists() for p in files):
                    frames = [pd.read_csv(p, parse_dates=["date"]) for p in files]
                    run_neural.ensemble(frames).to_csv(_path(task, spec.name, start), index=False)


def _updated(task, model, start) -> pd.DataFrame | None:
    """The same model's walk-forward forecasts, cut to the frozen test months."""
    if model.startswith("linear-"):
        path = config.RESULTS / f"preds_{task}_{model.split('-', 1)[1]}.csv"
    elif model == "rw":
        path = config.RESULTS / f"preds_{task}_rw.csv"
    else:
        path = run_neural.ensemble_path(task, model)
    if not path.exists():
        return None
    p = pd.read_csv(path, parse_dates=["date"])
    return p[p["date"] >= pd.Timestamp(start)]


def report() -> pd.DataFrame:
    rows = []
    for task in ("T1", "T2"):
        names = ["rw", "linear-S", "linear-S+N"] + [s.name for s in neural_ladder(task)]
        for start in STARTS:
            rw = pd.read_csv(_path(task, "rw", start), parse_dates=["date"])
            for model in names:
                path = _path(task, model, start)
                if not path.exists():
                    continue
                p = pd.read_csv(path, parse_dates=["date"])
                sq = lambda f: ((f["y_pred"] - f["y_true"]) ** 2).to_numpy()
                p_rw = np.nan if model == "rw" else dm_hln(sq(rw), sq(p))[1]
                row = {"task": task, "start": start[:4], "model": model, "version": "frozen",
                       **metrics(p), "p_vs_no_change": p_rw}
                rows.append(row)
                upd = _updated(task, model, start)
                if upd is not None and len(upd):
                    rows.append({"task": task, "start": start[:4], "model": model, "version": "updated",
                                 **metrics(upd), "p_vs_no_change": np.nan})
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "metrics.csv", index=False)
    lines = [
        "# Results: frozen models (train once, never update)", "",
        "Descriptive analysis from PREREG_H2A.md, modelled on the paper's fixed-rule exercise. Each model is "
        "trained once on data before 2016 (or 2020) and forecasts every later month or meeting; 'updated' is "
        "the same model refitted every month or meeting in the main analysis, over the same period. "
        "p_vs_no_change is a one-sided DM test against the no-change forecast; it is not part of any "
        "pre-registered decision.", "",
        table.round(4).to_markdown(index=False), "",
    ]
    (config.REPO / "RESULTS_FROZEN.md").write_text("\n".join(lines), encoding="utf-8")
    return table


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)
    if not args.report_only:
        run_all(device=args.device)
    print(report().round(4).to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `$PY -m pytest tests/test_frozen.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the frozen analysis (GPU, under an hour), the full suite, commit**

```bash
PYTHONPATH=src nohup $PY -u -m fed_news.frozen --device cuda > logs/frozen.log 2>&1 &
# when finished:
tail -40 logs/frozen.log
$PY -m pytest -q
git diff prereg -- src/fed_news/walkforward.py src/fed_news/evaluate.py src/fed_news/run_models.py src/fed_news/models.py
git add src/fed_news/frozen.py tests/test_frozen.py RESULTS_FROZEN.md results/frozen/
git commit -m "Add frozen-model analysis (train once before 2016/2020, never update)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
Expected: the log ends with the metrics table; `RESULTS_FROZEN.md` written; suite passes; the `git diff` prints nothing.

---

## Self-review notes

- Frozen analysis (PREREG_H2A.md section "Frozen-model analysis"): Task 7, reusing the Task 2 engine with `refit_every = NEVER` for neural models and a one-fit version of the Plan 1 linear models.
- Spec coverage: section 7 paper LSTM/GRU (Task 2 ladder, Task 4 one-hour rule), meeting LSTM, TFT, hybrid with out-of-fold residuals (Task 2); section 3 H2a tests and section 8 variance rule plus Holm (Task 6); section 11 leakage tests for the new engine (Task 2 perturbation tests, prereg guard, non-finite refusal).
- Settings the spec leaves open (batch size, TFT learning rate, epochs and patience, meeting-LSTM width, hybrid refit schedule, variance-rule formula, ensemble = seed mean) are fixed in `PREREG_H2A.md` and tagged before any S+N neural run.
- Plan 1 code is imported only; Tasks 2 and 6 check `git diff prereg` on the H1 files.
- Names checked across tasks: `NeuralSpec`, `neural_ladder`, `run_neural_spec`, `oof_residuals`, `_anchor`, `prereg_h2a_ok`, `seed_path`, `ensemble_path`, `ensemble`, `SEEDS`, `OUT`, `ARCHS`, `variance_rule`, `pair_tests`.
