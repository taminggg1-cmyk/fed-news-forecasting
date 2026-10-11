# Study 3: Decision-Trained ChronoBERT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a registered test of "16 statistics + a ChronoBERT text signal trained on Fed decisions" against "statistics only" on the per-meeting task.

**Architecture:** `decision_embed.py` turns every Fed document into a frozen ChronoBERT embedding per vintage (no outcome used). `decision_text.py` labels each document with the next FOMC decision, trains a per-year logistic regression only on labels known before that year, and scores that year's documents (P(hike) - P(cut)). `study3.py` turns document scores into a meeting signal with the existing `naive_panel`, runs the unchanged Plan 1 ordered logit with and without it on the same rows, and computes the registered tests; `evaluate_study3.py` writes `RESULTS_STUDY3.md`.

**Tech Stack:** Python 3.13, PyTorch 2.11 + CUDA, transformers 5.17 (ModernBERT-based ChronoBERT), scikit-learn 1.8, pandas 3.0, numpy, scipy, pytest.

**Spec:** `docs/superpowers/specs/2026-10-11-study3-decision-trained-text-design.md`

## Global Constraints

- Do not modify Plan 1-Study 2 code: `walkforward.py`, `evaluate.py`, `run_models.py`, `panel.py`, `models.py`, `stance.py`, `stance2.py`, `text_features.py`, `naive_stance.py`, `clark_west.py`, `stats_tests.py`. Only import from them.
- Python: `/d/gold_lm/venv/Scripts/python.exe`; run from the repo root `C:\Users\User\fed_news` with `PYTHONPATH=src`.
- Data root `config.DATA` (default `D:/fed_news`); Hugging Face cache `D:/hf_cache` (set by importing `fed_news.config`); all 26 ChronoBERT vintages are already cached.
- Tests run offline by default; tests that need the GPU carry `@pytest.mark.gpu` (deselected by default in `pyproject.toml`).
- C grid `(0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)`; fallback C `0.01`; minimum `100` training documents and `10` per class; window `512` tokens; vintage `v(y) = min(max(y - 1, 1999), 2024)`; test start `config.TEST_START` (`2001-01-01`); alpha `0.05`.
- No real-label training before the tag `prereg-study3` exists and `PREREG_STUDY3_OSF.txt` holds a URL starting with `https://osf.io/`.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A statement released on a meeting day must be labelled with the *next* meeting's decision, not that day's (pinned in Task 1, `test_label_is_first_meeting_strictly_after_release`).
2. Documents released after the last meeting have no label: never trained on, still scored (Task 1, same test plus `test_score_all_scores_only_years_with_enough_labels`).
3. Early cross-validation folds whose training part lacks a class must be skipped; if no fold is usable the fallback C is used, never a crash (Task 1, `test_fit_probe_falls_back_when_no_fold_has_all_classes`).
4. A changed document text must invalidate the embedding cache instead of silently reusing stale vectors (Task 2, `test_cache_reuses_and_recomputes_when_text_changes`).
5. Duplicate document ids must be refused, because the embedding lookup is keyed by id (Task 2, `test_duplicate_doc_ids_are_refused`; Task 3 run check).

---

### Task 1: Document labels, year rule, per-year probe, gate

**Files:**
- Create: `src/fed_news/decision_text.py`
- Test: `tests/test_decision_text.py`

**Interfaces:**
- Consumes: `fed_news.stance.vintage_year(row_year: int) -> int`; `fed_news.models.expanding_folds(n, k=5)`; `fed_news.config` (`DATA`, `RESULTS`, `REPO`).
- Produces:
  - `next_decision(docs: DataFrame[doc_id, release, ...], meetings: DataFrame[y_class] indexed by decision date) -> DataFrame` adding `label` (int, -1 when no later meeting) and `label_date` (datetime, NaT when none).
  - `training_docs(labelled, year: int) -> DataFrame`, `enough(train) -> bool`.
  - `fit_probe(X: ndarray, y: ndarray) -> (model, mu, sd, C)`, `doc_scores(model, mu, sd, X) -> ndarray`.
  - `score_all(labelled, embed, shuffle_seed: int | None = None) -> (scored DataFrame with year, stance; quality DataFrame)`, where `embed(vintage: int, doc_ids: list[str]) -> ndarray`.
  - `gate(repo=config.REPO) -> str | None`.
  - Constants `TAG, OSF_NAME, OUT_PATH, SHUFFLED_PATH, RESULTS, CLASSES, C_GRID, FALLBACK_C, MIN_DOCS, MIN_PER_CLASS`.

- [ ] **Step 1: Write the failing tests**

`tests/test_decision_text.py`:

```python
import subprocess

import numpy as np
import pandas as pd

from fed_news import decision_text as dt
from fed_news.stance import vintage_year


def meetings():
    idx = pd.to_datetime(["2000-03-21", "2000-06-28", "2001-01-31"])
    return pd.DataFrame({"y_class": [2, 1, 0]}, index=pd.Index(idx, name="date"))


def test_label_is_first_meeting_strictly_after_release():
    docs = pd.DataFrame({"doc_id": ["a", "b", "c", "d"],
                         "release": pd.to_datetime(["2000-01-05", "2000-03-21", "2000-06-27", "2001-02-01"])})
    out = dt.next_decision(docs, meetings())
    assert out["label"].tolist() == [2, 1, 1, -1]          # b is released on a meeting day: next meeting's label
    assert out["label_date"].iloc[1] == pd.Timestamp("2000-06-28")
    assert pd.isna(out["label_date"].iloc[3])


def test_training_docs_use_only_labels_known_before_the_year():
    docs = pd.DataFrame({"doc_id": list("abc"),
                         "release": pd.to_datetime(["2000-01-05", "2000-12-20", "2000-12-30"])})
    tr = dt.training_docs(dt.next_decision(docs, meetings()), 2001)
    assert tr["doc_id"].tolist() == ["a"]                  # b and c are labelled by the 2001-01-31 meeting


def test_enough_needs_100_docs_and_10_per_class():
    ok = pd.DataFrame({"label": [0] * 10 + [1] * 80 + [2] * 10})
    assert dt.enough(ok)
    assert not dt.enough(ok.iloc[1:])                      # 99 documents and 9 cuts
    assert not dt.enough(pd.DataFrame({"label": [1] * 95 + [2] * 15}))


def test_vintage_rule():
    assert [vintage_year(y) for y in (1996, 2000, 2001, 2025, 2026)] == [1999, 1999, 2000, 2024, 2024]


def separable(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    X = rng.normal(size=(n, 5))
    X[:, 0] += 3 * (y - 1)
    return X, y


def test_fit_probe_learns_signal_and_picks_c_from_grid():
    X, y = separable()
    model, mu, sd, c = dt.fit_probe(X, y)
    assert c in dt.C_GRID
    assert np.corrcoef(dt.doc_scores(model, mu, sd, X), y)[0, 1] > 0.8


def test_fit_probe_falls_back_when_no_fold_has_all_classes():
    y = np.array([1] * 90 + [0] * 10 + [2] * 10)           # every expanding training fold lacks the hikes
    X = np.random.default_rng(1).normal(size=(len(y), 3))
    _, _, _, c = dt.fit_probe(X, y)
    assert c == dt.FALLBACK_C


def world(seed=0, n=1200):
    rng = np.random.default_rng(seed)
    mdates = pd.date_range("1995-01-15", "2002-12-15", freq="45D")
    meet = pd.DataFrame({"y_class": np.resize([0, 1, 2, 1], len(mdates))},           # every class every year
                        index=pd.Index(mdates, name="date"))
    lo, hi = pd.Timestamp("1995-01-01").value // 10 ** 9, pd.Timestamp("2002-12-31").value // 10 ** 9
    release = pd.to_datetime(rng.integers(lo, hi, n), unit="s").normalize()
    docs = pd.DataFrame({"doc_id": [f"d{i}" for i in range(n)], "type": "speech", "release": release,
                         "source": "corpus"})
    table = {d: rng.normal(size=4) for d in docs["doc_id"]}
    return docs, meet, (lambda v, ids: np.stack([table[i] for i in ids]))


def test_score_all_scores_only_years_with_enough_labels():
    docs, meet, emb = world()
    scored, quality = dt.score_all(dt.next_decision(docs, meet), emb)
    assert scored.loc[scored["release"].dt.year == 1995, "stance"].isna().all()
    assert scored.loc[scored["release"].dt.year >= 1996, "stance"].notna().all()
    assert scored["stance"].dropna().between(-1, 1).all()
    assert {"year", "n_train", "classifier", "C", "accuracy", "majority_accuracy"} <= set(quality.columns)


def test_future_labels_do_not_change_past_scores():
    docs, meet, emb = world()
    a, _ = dt.score_all(dt.next_decision(docs, meet), emb)
    meet2 = meet.copy()
    later = meet2.index >= pd.Timestamp("2000-01-01")
    meet2.loc[later, "y_class"] = (meet2.loc[later, "y_class"] + 1) % 3
    b, _ = dt.score_all(dt.next_decision(docs, meet2), emb)
    early = (a["release"].dt.year < 2000).to_numpy()
    np.testing.assert_allclose(a["stance"].to_numpy()[early], b["stance"].to_numpy()[early])


def test_shuffled_labels_change_scores_but_not_coverage():
    docs, meet, emb = world()
    lab = dt.next_decision(docs, meet)
    real, _ = dt.score_all(lab, emb)
    shuf, _ = dt.score_all(lab, emb, shuffle_seed=0)
    assert (real["stance"].isna() == shuf["stance"].isna()).all()
    assert not np.allclose(real["stance"].dropna(), shuf["stance"].dropna())


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_gate_needs_tag_and_osf_url(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x")
    assert "tag" in dt.gate(tmp_path)
    git(tmp_path, "tag", dt.TAG)
    assert "OSF" in dt.gate(tmp_path)
    (tmp_path / dt.OSF_NAME).write_text("https://osf.io/abcd1/\n", encoding="utf-8")
    assert dt.gate(tmp_path) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src /d/gold_lm/venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_decision_text.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'fed_news.decision_text'` (or `ImportError`).

- [ ] **Step 3: Write the implementation**

`src/fed_news/decision_text.py`:

```python
"""Study 3 (PREREG_STUDY3.md; spec docs/superpowers/specs/2026-10-11-study3-decision-trained-text-design.md).

Each document is labelled with the class (0 cut, 1 hold, 2 hike) of the first FOMC decision strictly after its
release. The classifier of release year y is a multinomial logistic regression on frozen ChronoBERT embeddings
of vintage v(y) (decision_embed), trained only on documents whose label meeting is before 1 January of year y;
it scores the documents released in year y: score = P(hike) - P(cut). Every score is therefore out of sample
for its own label. Training on real labels is refused until the tag `prereg-study3` exists and
PREREG_STUDY3_OSF.txt holds the OSF registration URL; `--embed-only` uses no outcome.
"""
import argparse
import subprocess

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from fed_news import config, models
from fed_news.stance import vintage_year

TAG = "prereg-study3"
OSF_NAME = "PREREG_STUDY3_OSF.txt"
OUT_PATH = config.DATA / "doc_decision.pkl"
SHUFFLED_PATH = config.DATA / "doc_decision_shuffled.pkl"
RESULTS = config.RESULTS / "study3"
CLASSES = (0, 1, 2)
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
FALLBACK_C = 0.01
MIN_DOCS, MIN_PER_CLASS = 100, 10


def next_decision(docs: pd.DataFrame, meetings: pd.DataFrame) -> pd.DataFrame:
    """Add `label` (y_class of the first meeting strictly after release, -1 if none) and `label_date`."""
    if not meetings.index.is_monotonic_increasing:
        raise ValueError("meetings must be sorted by decision date")
    dates = meetings.index.to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(dates, docs["release"].to_numpy(dtype="datetime64[ns]"), side="right")
    has = pos < len(dates)
    label_date = np.full(len(docs), np.datetime64("NaT"), dtype="datetime64[ns]")
    label = np.full(len(docs), -1, dtype=int)
    label_date[has] = dates[pos[has]]
    label[has] = meetings["y_class"].to_numpy(int)[pos[has]]
    return docs.assign(label_date=label_date, label=label)


def training_docs(labelled: pd.DataFrame, year: int) -> pd.DataFrame:
    """Documents whose label meeting is before 1 January of `year`, in label-date order."""
    known = (labelled["label"] >= 0) & (labelled["label_date"] < pd.Timestamp(f"{year}-01-01"))
    return labelled[known].sort_values(["label_date", "release", "doc_id"])


def enough(train: pd.DataFrame) -> bool:
    counts = train["label"].value_counts()
    return len(train) >= MIN_DOCS and all(counts.get(k, 0) >= MIN_PER_CLASS for k in CLASSES)


def _make(c: float) -> LogisticRegression:
    return LogisticRegression(C=c, class_weight="balanced", max_iter=2000)


def _fold_loss(Z: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray, c: float) -> float:
    proba = np.clip(_make(c).fit(Z[tr], y[tr]).predict_proba(Z[va]), 1e-12, None)
    return float(-np.mean(np.log(proba[np.arange(len(va)), y[va]])))


def fit_probe(X: np.ndarray, y: np.ndarray):
    """Standardise, choose C by expanding-window CV over folds whose training part has every class, fit."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (X - mu) / sd
    folds = [(tr, va) for tr, va in models.expanding_folds(len(y)) if set(y[tr].tolist()) == set(CLASSES)]
    best, best_loss = FALLBACK_C, np.inf
    if folds:
        for c in C_GRID:
            loss = float(np.mean([_fold_loss(Z, y, tr, va, c) for tr, va in folds]))
            if loss < best_loss:
                best, best_loss = c, loss
    return _make(best).fit(Z, y), mu, sd, best


def doc_scores(model, mu: np.ndarray, sd: np.ndarray, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba((X - mu) / sd)
    return proba[:, 2] - proba[:, 0]


def score_all(labelled: pd.DataFrame, embed, shuffle_seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every document with the classifier of its release year. embed(vintage, doc_ids) -> (n, dim) array.

    With `shuffle_seed`, each year's training labels are permuted (the shuffled-label control)."""
    docs = labelled.assign(year=labelled["release"].dt.year)
    stance = np.full(len(docs), np.nan)
    rows = []
    for year in sorted(int(y) for y in docs["year"].unique()):
        train = training_docs(docs, year)
        here = np.flatnonzero(docs["year"].to_numpy() == year)
        row = {"year": year, "n_train": len(train), "n_scored": len(here),
               **{f"n_train_{k}": int((train["label"] == k).sum()) for k in CLASSES}}
        if not enough(train):
            rows.append({**row, "classifier": False})
            continue
        v = vintage_year(year)
        y = train["label"].to_numpy(int)
        if shuffle_seed is not None:
            y = np.random.default_rng([shuffle_seed, year]).permutation(y)
        model, mu, sd, c = fit_probe(embed(v, train["doc_id"].tolist()), y)
        ydocs = docs.iloc[here]
        proba = model.predict_proba((embed(v, ydocs["doc_id"].tolist()) - mu) / sd)
        stance[here] = proba[:, 2] - proba[:, 0]
        truth, pred = ydocs["label"].to_numpy(int), proba.argmax(axis=1)
        known = truth >= 0
        row.update({"classifier": True, "vintage": v, "C": c, "n_scored_labelled": int(known.sum())})
        if known.any():
            majority = int(np.bincount(y, minlength=3).argmax())
            row.update({"accuracy": float((pred[known] == truth[known]).mean()),
                        "macro_f1": float(f1_score(truth[known], pred[known], labels=list(CLASSES),
                                                   average="macro", zero_division=0)),
                        "majority_accuracy": float((truth[known] == majority).mean())})
        rows.append(row)
    return docs.assign(stance=stance), pd.DataFrame(rows)


def gate(repo=config.REPO) -> str | None:
    """None when real-label training is allowed, otherwise the reason it is refused."""
    tag = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", f"refs/tags/{TAG}"],
                         capture_output=True, text=True)
    if tag.returncode != 0:
        return f"tag {TAG} missing"
    osf = repo / OSF_NAME
    url = osf.read_text(encoding="utf-8").strip() if osf.exists() else ""
    if not url.startswith("https://osf.io/"):
        return f"{OSF_NAME} must hold the OSF registration URL"
    return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src /d/gold_lm/venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_decision_text.py`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fed_news/decision_text.py tests/test_decision_text.py
git commit -m "Study 3: document labels, year rule, per-year probe and registration gate

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Frozen ChronoBERT embeddings with a fingerprinted cache

**Files:**
- Create: `src/fed_news/decision_embed.py`
- Test: `tests/test_decision_embed.py`

**Interfaces:**
- Consumes: `fed_news.stance.model_name(vintage) -> str`, `fed_news.stance.vintage_year(row_year) -> int`, `fed_news.config.DATA`.
- Produces:
  - `window_bounds(n_tokens: int, size: int = WINDOW) -> list[tuple[int, int]]`, `mean_pool(hidden, mask) -> Tensor`.
  - `embed_texts(texts: list[str], vintage: int, batch: int = BATCH) -> ndarray (n, 768)` (GPU).
  - `needed_docs(docs, vintage) -> DataFrame`, `fingerprint(docs, vintage) -> str`.
  - `load_or_embed(docs, vintage, embed_fn=embed_texts, cache_dir=CACHE_DIR) -> dict[str, ndarray]`.
  - `cached_embedder(docs, embed_fn=embed_texts, cache_dir=CACHE_DIR)` returning `embed(vintage, doc_ids) -> ndarray` (the callable Task 1's `score_all` consumes).
  - `embed_all(docs, embed_fn=embed_texts, cache_dir=CACHE_DIR) -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_decision_embed.py`:

```python
import numpy as np
import pandas as pd
import pytest
import torch

from fed_news import decision_embed as de


def test_window_bounds_cover_every_token_once():
    assert de.window_bounds(0) == [(0, 0)]
    assert de.window_bounds(1020, 512) == [(0, 510), (510, 1020)]
    b = de.window_bounds(1021, 512)
    assert len(b) == 3 and b[-1] == (1020, 1021)


def test_mean_pool_ignores_padding():
    h = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    m = torch.tensor([[1, 1, 0]])
    assert torch.allclose(de.mean_pool(h, m), torch.tensor([[2.0, 2.0]]))


def test_needed_docs_cover_every_year_using_the_vintage():
    docs = pd.DataFrame({"doc_id": list("abcd"), "text": "x",
                         "release": pd.to_datetime(["1996-05-01", "2000-05-01", "2001-05-01", "2026-01-01"])})
    assert de.needed_docs(docs, 1999)["doc_id"].tolist() == ["a", "b"]
    assert de.needed_docs(docs, 2000)["doc_id"].tolist() == ["a", "b", "c"]
    assert de.needed_docs(docs, 2015)["doc_id"].tolist() == ["a", "b", "c"]
    assert de.needed_docs(docs, 2024)["doc_id"].tolist() == list("abcd")
    assert de.needed_docs(docs, 1990).empty


def fake_factory(calls):
    def fake(texts, vintage):
        calls.append(list(texts))
        return np.arange(len(texts) * 3, dtype=np.float32).reshape(len(texts), 3) + vintage
    return fake


def small_docs():
    return pd.DataFrame({"doc_id": ["a", "b"], "text": ["one", "two"],
                         "release": pd.to_datetime(["1999-01-01", "2000-01-01"])})


def test_cache_reuses_and_recomputes_when_text_changes(tmp_path):
    calls, docs = [], small_docs()
    first = de.load_or_embed(docs, 1999, fake_factory(calls), tmp_path)
    again = de.load_or_embed(docs, 1999, fake_factory(calls), tmp_path)
    assert len(calls) == 1 and np.array_equal(first["b"], again["b"])
    docs.loc[1, "text"] = "changed"
    de.load_or_embed(docs, 1999, fake_factory(calls), tmp_path)
    assert len(calls) == 2


def test_duplicate_doc_ids_are_refused(tmp_path):
    docs = pd.concat([small_docs(), small_docs().iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        de.load_or_embed(docs, 1999, fake_factory([]), tmp_path)


def test_cached_embedder_returns_rows_in_requested_order(tmp_path):
    calls, docs = [], small_docs()
    table = de.load_or_embed(docs, 1999, fake_factory(calls), tmp_path)
    out = de.cached_embedder(docs, fake_factory(calls), tmp_path)(1999, ["b", "a"])
    assert out.shape == (2, 3)
    assert np.array_equal(out[0], table["b"]) and np.array_equal(out[1], table["a"])


@pytest.mark.gpu
def test_embed_texts_gpu_smoke():
    out = de.embed_texts(["The Committee decided to raise the target range.", "word " * 1500], 1999, batch=4)
    assert out.shape == (2, 768) and np.isfinite(out).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src /d/gold_lm/venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_decision_embed.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'fed_news.decision_embed'`.

- [ ] **Step 3: Write the implementation**

`src/fed_news/decision_embed.py`:

```python
"""Frozen ChronoBERT document embeddings for Study 3 (no outcome is used here).

A document is tokenised in full and split into consecutive windows of at most WINDOW tokens
([CLS] + up to WINDOW - 2 tokens + [SEP]); each window's last hidden state is mean-pooled over its tokens and
the document embedding is the mean over its windows. One cache file per vintage, <data>/decision_emb/{v}.npz,
holds the embeddings of every document released up to the last year that uses the vintage, with a fingerprint
of the model name, the window size and the documents (ids and text hashes); a changed fingerprint recomputes.
"""
import gc
import hashlib

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from fed_news import config
from fed_news.stance import model_name, vintage_year

WINDOW = 512
BATCH = 16
CACHE_DIR = config.DATA / "decision_emb"


def window_bounds(n_tokens: int, size: int = WINDOW) -> list[tuple[int, int]]:
    body = size - 2
    return [(i, min(i + body, n_tokens)) for i in range(0, n_tokens, body)] or [(0, 0)]


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1)


def embed_texts(texts: list[str], vintage: int, batch: int = BATCH) -> np.ndarray:
    name = model_name(vintage)
    tok = AutoTokenizer.from_pretrained(name)
    net = AutoModel.from_pretrained(name).cuda().eval()
    ids = [np.asarray(tok(t, add_special_tokens=False)["input_ids"], dtype=np.int64) for t in texts]
    wins = [(d, a, b) for d, arr in enumerate(ids) for a, b in window_bounds(len(arr))]
    order = np.argsort([b - a for _, a, b in wins], kind="stable")
    pooled = np.zeros((len(wins), net.config.hidden_size), dtype=np.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for start in range(0, len(order), batch):
            part = order[start:start + batch]
            seqs = [np.concatenate([[tok.cls_token_id], ids[wins[k][0]][wins[k][1]:wins[k][2]], [tok.sep_token_id]])
                    for k in part]
            width = max(len(s) for s in seqs)
            input_ids = torch.full((len(seqs), width), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(seqs), width), dtype=torch.long)
            for r, s in enumerate(seqs):
                input_ids[r, :len(s)] = torch.from_numpy(s)
                mask[r, :len(s)] = 1
            mask = mask.cuda()
            hidden = net(input_ids=input_ids.cuda(), attention_mask=mask).last_hidden_state
            pooled[part] = mean_pool(hidden.float(), mask).cpu().numpy()
    owner = np.array([d for d, _, _ in wins])
    sums = np.zeros((len(texts), pooled.shape[1]), dtype=np.float64)
    np.add.at(sums, owner, pooled)
    out = (sums / np.bincount(owner, minlength=len(texts))[:, None]).astype(np.float32)
    del net, tok
    gc.collect()
    torch.cuda.empty_cache()
    return out


def needed_docs(docs: pd.DataFrame, vintage: int) -> pd.DataFrame:
    """Every document released up to the last year whose classifier uses this vintage."""
    years = docs["release"].dt.year
    using = [y for y in range(int(years.min()), int(years.max()) + 1) if vintage_year(y) == vintage]
    return docs[years <= max(using)] if using else docs.iloc[:0]


def fingerprint(docs: pd.DataFrame, vintage: int) -> str:
    h = hashlib.sha1(f"{model_name(vintage)}|{WINDOW}".encode())
    for d, t in sorted(zip(docs["doc_id"].astype(str), docs["text"].astype(str))):
        h.update(f"{d}\t{hashlib.sha1(t.encode()).hexdigest()}\n".encode())
    return h.hexdigest()


def load_or_embed(docs: pd.DataFrame, vintage: int, embed_fn=embed_texts, cache_dir=CACHE_DIR) -> dict:
    if not docs["doc_id"].is_unique:
        raise ValueError("duplicate doc_id in the document table")
    need = needed_docs(docs, vintage).sort_values("doc_id")
    path = cache_dir / f"{vintage}.npz"
    fp = fingerprint(need, vintage)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if str(z["fingerprint"]) == fp:
            return dict(zip(z["doc_id"].tolist(), z["emb"]))
    emb = embed_fn(need["text"].tolist(), vintage)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / f"{vintage}.tmp.npz"
    np.savez(tmp, doc_id=need["doc_id"].to_numpy(dtype=str), emb=emb, fingerprint=np.array(fp))
    tmp.replace(path)
    return dict(zip(need["doc_id"].tolist(), emb))


def cached_embedder(docs: pd.DataFrame, embed_fn=embed_texts, cache_dir=CACHE_DIR):
    store: dict[int, dict] = {}

    def embed(vintage: int, doc_ids: list[str]) -> np.ndarray:
        if vintage not in store:
            store.clear()
            store[vintage] = load_or_embed(docs, vintage, embed_fn, cache_dir)
        return np.stack([store[vintage][d] for d in doc_ids])

    return embed


def embed_all(docs: pd.DataFrame, embed_fn=embed_texts, cache_dir=CACHE_DIR) -> None:
    years = range(int(docs["release"].dt.year.min()), int(docs["release"].dt.year.max()) + 1)
    for v in sorted({vintage_year(y) for y in years}):
        table = load_or_embed(docs, v, embed_fn, cache_dir)
        print(f"vintage {v}: {len(table)} documents", flush=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src /d/gold_lm/venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_decision_embed.py`
Expected: 6 passed, 1 deselected.

Then the GPU smoke test: `PYTHONPATH=src /d/gold_lm/venv/Scripts/python.exe -m pytest -q -p no:cacheprovider -m gpu tests/test_decision_embed.py`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fed_news/decision_embed.py tests/test_decision_embed.py
git commit -m "Study 3: frozen ChronoBERT document embeddings with fingerprinted cache

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Study 3 text runner and command line

**Files:**
- Modify: `src/fed_news/decision_text.py` (append `run` and `main`)
- Test: `tests/test_decision_text.py` (append)

**Interfaces:**
- Consumes: Task 1 functions; Task 2 `cached_embedder`, `embed_all`; `fed_news.stance2.all_docs() -> DataFrame[doc_id, type, release, text, source]`.
- Produces: `run(docs=None, meetings=None, embed=None, out_path=OUT_PATH, shuffled_path=SHUFFLED_PATH, results=RESULTS, repo=config.REPO) -> None` writing `doc_decision.pkl`, `doc_decision_shuffled.pkl` (columns doc_id, type, release, source, label_date, label, year, stance), `results/study3/probe_quality.csv`, `probe_quality_shuffled.csv`; CLI `python -m fed_news.decision_text [--embed-only]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_decision_text.py`:

```python
import pytest


def test_run_refuses_without_registration(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "gate", lambda repo=None: "tag prereg-study3 missing")
    with pytest.raises(PermissionError, match="refused"):
        dt.run(out_path=tmp_path / "a.pkl", shuffled_path=tmp_path / "b.pkl", results=tmp_path)


def test_run_writes_real_and_shuffled_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "gate", lambda repo=None: None)
    docs, meet, emb = world()
    dt.run(docs=docs.assign(text="x"), meetings=meet, embed=emb, out_path=tmp_path / "real.pkl",
           shuffled_path=tmp_path / "shuf.pkl", results=tmp_path / "res")
    real, shuf = pd.read_pickle(tmp_path / "real.pkl"), pd.read_pickle(tmp_path / "shuf.pkl")
    assert {"doc_id", "type", "release", "source", "label", "label_date", "year", "stance"} <= set(real.columns)
    assert "text" not in real.columns
    assert (real["doc_id"] == shuf["doc_id"]).all()
    assert (tmp_path / "res" / "probe_quality.csv").exists()
    assert (tmp_path / "res" / "probe_quality_shuffled.csv").exists()


def test_run_refuses_duplicate_doc_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "gate", lambda repo=None: None)
    docs, meet, emb = world()
    docs = pd.concat([docs, docs.iloc[:1]], ignore_index=True).assign(text="x")
    with pytest.raises(ValueError, match="duplicate"):
        dt.run(docs=docs, meetings=meet, embed=emb, out_path=tmp_path / "a.pkl",
               shuffled_path=tmp_path / "b.pkl", results=tmp_path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src /d/gold_lm/venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_decision_text.py`
Expected: 3 failures with `AttributeError: module 'fed_news.decision_text' has no attribute 'run'`.

- [ ] **Step 3: Write the implementation**

Append to `src/fed_news/decision_text.py`:

```python
def run(docs: pd.DataFrame | None = None, meetings: pd.DataFrame | None = None, embed=None,
        out_path=OUT_PATH, shuffled_path=SHUFFLED_PATH, results=RESULTS, repo=config.REPO) -> None:
    reason = gate(repo)
    if reason:
        raise PermissionError(f"Study 3 real-label run refused: {reason}")
    if docs is None:
        from fed_news.stance2 import all_docs
        docs = all_docs()
    if not docs["doc_id"].is_unique:
        raise ValueError("duplicate doc_id in the document table")
    meetings = pd.read_pickle(config.DATA / "meetings.pkl") if meetings is None else meetings
    if embed is None:
        from fed_news.decision_embed import cached_embedder
        embed = cached_embedder(docs)
    labelled = next_decision(docs.drop(columns=["text"], errors="ignore"), meetings)
    results.mkdir(parents=True, exist_ok=True)
    for path, seed, name in ((out_path, None, "probe_quality.csv"),
                             (shuffled_path, 0, "probe_quality_shuffled.csv")):
        scored, quality = score_all(labelled, embed, shuffle_seed=seed)
        scored.to_pickle(path)
        quality.to_csv(results / name, index=False)
        print(f"{path.name}: {int(scored['stance'].notna().sum())} of {len(scored)} documents scored", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-only", action="store_true", help="compute ChronoBERT embeddings only (no outcome)")
    args = ap.parse_args(argv)
    if args.embed_only:
        from fed_news.decision_embed import embed_all
        from fed_news.stance2 import all_docs
        embed_all(all_docs())
    else:
        run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src /d/gold_lm/venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_decision_text.py`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fed_news/decision_text.py tests/test_decision_text.py
git commit -m "Study 3: text runner (real and shuffled-label scores) and command line

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Forecasting, registered tests and power statement

**Files:**
- Create: `src/fed_news/study3.py`
- Test: `tests/test_study3.py`

**Interfaces:**
- Consumes: `decision_text.OUT_PATH, SHUFFLED_PATH, RESULTS, gate`; `naive_stance.naive_panel(name, doc_stance)`; `walkforward.Spec, common_rows, run_spec`; `panel.S_COLS, M_COLS, N_COLS`; `clark_west.compare(small, large) -> {n, dm_hln_p, clark_west_t, clark_west_p}`; `stats_tests.dm_hln, mcnemar_greater`; `evaluate.metrics`.
- Produces: `same_rows(*frames)`, `panels(ds, ds_shuffled)`, `forecasts(dec, shu, orig) -> dict[str, DataFrame]` with keys `S, S+D, placebo, shuffled-text, S+M, S+M+D, S+N`; `hypothesis_tests(f) -> DataFrame` with rows named `H7 Clark-West`, `placebo Clark-West`, `shuffled-text Clark-West`, `S+M+D vs S+M Clark-West`, `decisions McNemar`, `DM S+D vs S`, `DM S+D vs S+N` and columns `test, role, base, model, n, rmse_base, rmse_model, accuracy_base, accuracy_model, stat, p`; `decide_h7(p_main, p_placebo, p_shuffled, alpha=config.ALPHA) -> str`; `power_statement(base, new) -> dict`; `power_sentence(p: dict) -> str`; CLI `python -m fed_news.study3 [--power]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_study3.py`:

```python
import numpy as np
import pandas as pd
import pytest

from fed_news import study3 as s3


def preds(y_pred, y_true, seed=0):
    dates = pd.date_range("2001-01-01", periods=len(y_true), freq="45D")
    cls = lambda v: np.where(v > 0.125, 2, np.where(v < -0.125, 0, 1))
    return pd.DataFrame({"date": dates, "y_true": y_true, "y_pred": y_pred, "lag": 0.0,
                         "class_true": cls(y_true), "class_pred": cls(y_pred)})


def fake_forecasts(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice([-0.25, 0.0, 0.25], n, p=[0.2, 0.6, 0.2])
    base = y + rng.normal(0, 0.2, n)
    good = y + rng.normal(0, 0.1, n)
    noise = lambda: base + rng.normal(0, 0.02, n)
    return {"S": preds(base, y), "S+D": preds(good, y), "placebo": preds(noise(), y),
            "shuffled-text": preds(noise(), y), "S+M": preds(base, y), "S+M+D": preds(good, y),
            "S+N": preds(noise(), y)}


def test_decide_h7():
    assert s3.decide_h7(0.01, 0.5, 0.5) == "supported"
    assert s3.decide_h7(0.20, 0.5, 0.5) == "not supported"
    assert s3.decide_h7(0.01, 0.04, 0.5) == "void"
    assert s3.decide_h7(0.01, 0.5, 0.04) == "void"
    with pytest.raises(ValueError):
        s3.decide_h7(float("nan"), 0.5, 0.5)


def test_hypothesis_tests_names_and_direction():
    table = s3.hypothesis_tests(fake_forecasts()).set_index("test")
    assert list(table.index) == ["H7 Clark-West", "placebo Clark-West", "shuffled-text Clark-West",
                                 "S+M+D vs S+M Clark-West", "decisions McNemar", "DM S+D vs S", "DM S+D vs S+N"]
    assert table.loc["H7 Clark-West", "p"] < 0.01             # S+D is much better by construction
    assert table["p"].between(0, 1).all()
    assert (table["n"] == 200).all()


def test_same_rows_intersects_indices():
    a = pd.DataFrame({"x": range(4)}, index=[1, 2, 3, 4])
    b = pd.DataFrame({"x": range(3)}, index=[2, 3, 5])
    ra, rb = s3.same_rows(a, b)
    assert list(ra.index) == [2, 3] and list(rb.index) == [2, 3]


def test_power_statement_formula():
    y = np.zeros(100)
    base, new = preds(np.r_[np.ones(50), -np.ones(50)], y), preds(np.zeros(100), y)
    p = s3.power_statement(base, new)
    d = np.ones(100)                                          # every loss differential equals 1
    assert p["mse_base"] == pytest.approx(1.0) and p["sd_loss_diff"] == pytest.approx(np.std(d, ddof=1))
    assert p["mde_mse"] == pytest.approx(s3.Z80 * p["sd_loss_diff"] / 10)
    assert "100 meetings" in s3.power_sentence(p)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src /d/gold_lm/venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_study3.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'fed_news.study3'`.

- [ ] **Step 3: Write the implementation**

`src/fed_news/study3.py`:

```python
"""Study 3 forecasting and tests (PREREG_STUDY3.md, hypothesis H7).

The decision-trained document scores (decision_text) become the meeting signal D: naive_stance.naive_panel
writes the mean of group means over [previous cutoff, cutoff), carried forward, into the `n_stance` column of a
copy of the T2 panel. Every model is the Plan 1 penalised ordered logit (walkforward.run_spec, unchanged) on
exactly the same rows. `--power` writes the pre-registration power statement from the committed Plan 1
forecasts and needs no Study 3 output.
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from fed_news import config
from fed_news.clark_west import compare
from fed_news.decision_text import OUT_PATH, RESULTS, SHUFFLED_PATH, gate
from fed_news.evaluate import metrics
from fed_news.naive_stance import naive_panel
from fed_news.panel import M_COLS, N_COLS, S_COLS
from fed_news.stats_tests import dm_hln, mcnemar_greater
from fed_news.walkforward import Spec, common_rows, run_spec

D = ["n_stance"]  # the decision signal lives in this column of the Study 3 panel
Z80 = float(stats.norm.ppf(0.95) + stats.norm.ppf(0.80))
CW_TESTS = (("H7 Clark-West", "primary", "S", "S+D"),
            ("placebo Clark-West", "control", "S", "placebo"),
            ("shuffled-text Clark-West", "control", "S", "shuffled-text"),
            ("S+M+D vs S+M Clark-West", "secondary", "S+M", "S+M+D"))


def same_rows(*frames: pd.DataFrame) -> list[pd.DataFrame]:
    rows = frames[0].index
    for f in frames[1:]:
        rows = rows.intersection(f.index)
    return [f.loc[rows] for f in frames]


def panels(ds: pd.DataFrame, ds_shuffled: pd.DataFrame) -> list[pd.DataFrame]:
    """Decision-signal panel, shuffled-label-signal panel and the original Plan 1 panel, on the same rows."""
    return same_rows(common_rows(naive_panel("t2", ds)), common_rows(naive_panel("t2", ds_shuffled)),
                     common_rows(pd.read_pickle(config.DATA / "t2_full.pkl")))


def forecasts(dec: pd.DataFrame, shu: pd.DataFrame, orig: pd.DataFrame) -> dict[str, pd.DataFrame]:
    def run(df, name, cols, **kw):
        return run_spec(df, Spec(name, "T2", "primary", cols, **kw), config.TEST_START)
    return {"S": run(dec, "S", S_COLS),
            "S+D": run(dec, "S+D", S_COLS + D),
            "placebo": run(dec, "placebo", S_COLS + D, placebo=True),
            "shuffled-text": run(shu, "shuffled-text", S_COLS + D),
            "S+M": run(dec, "S+M", S_COLS + M_COLS),
            "S+M+D": run(dec, "S+M+D", S_COLS + M_COLS + D),
            "S+N": run(orig, "S+N", S_COLS + N_COLS)}


def _correct(p: pd.DataFrame) -> np.ndarray:
    return (p["class_pred"] == p["class_true"]).to_numpy()


def _sq(p: pd.DataFrame) -> np.ndarray:
    return ((p["y_pred"] - p["y_true"]) ** 2).to_numpy(float)


def hypothesis_tests(f: dict[str, pd.DataFrame]) -> pd.DataFrame:
    def base_row(test, role, base, model):
        a, b = metrics(f[base]), metrics(f[model])
        return {"test": test, "role": role, "base": base, "model": model, "n": a["n"], "rmse_base": a["rmse"],
                "rmse_model": b["rmse"], "accuracy_base": a["accuracy"], "accuracy_model": b["accuracy"]}
    rows = []
    for test, role, base, model in CW_TESTS:
        c = compare(f[base], f[model])
        rows.append({**base_row(test, role, base, model), "stat": c["clark_west_t"], "p": c["clark_west_p"]})
    compare(f["S"], f["S+D"])                                   # raises if the forecast dates differ
    fixes, breaks, p = mcnemar_greater(_correct(f["S"]), _correct(f["S+D"]))
    rows.append({**base_row("decisions McNemar", "secondary", "S", "S+D"), "stat": fixes - breaks, "p": p})
    for test, base in (("DM S+D vs S", "S"), ("DM S+D vs S+N", "S+N")):
        compare(f[base], f["S+D"])
        stat, p = dm_hln(_sq(f[base]), _sq(f["S+D"]))
        rows.append({**base_row(test, "secondary", base, "S+D"), "stat": stat, "p": p})
    return pd.DataFrame(rows)


def decide_h7(p_main: float, p_placebo: float, p_shuffled: float, alpha: float = config.ALPHA) -> str:
    if any(np.isnan(p) for p in (p_main, p_placebo, p_shuffled)):
        raise ValueError("NaN p-value in the H7 decision")
    if p_placebo < alpha or p_shuffled < alpha:
        return "void"
    return "supported" if p_main < alpha else "not supported"


def power_statement(base: pd.DataFrame, new: pd.DataFrame) -> dict:
    """MSE improvement a one-sided 5% test detects with 80% power: Z80 * sd(loss differential) / sqrt(n)."""
    y = base["y_true"].to_numpy(float)
    e_base = (y - base["y_pred"].to_numpy(float)) ** 2
    d = e_base - (y - new["y_pred"].to_numpy(float)) ** 2
    mse, sd = float(e_base.mean()), float(np.std(d, ddof=1))
    mde = Z80 * sd / np.sqrt(len(d))
    return {"n": len(d), "mse_base": mse, "sd_loss_diff": sd, "z": Z80, "mde_mse": mde,
            "mde_pct_of_mse": 100 * mde / mse, "rmse_base": float(np.sqrt(mse)),
            "rmse_at_mde": float(np.sqrt(max(mse - mde, 0.0)))}


def power_sentence(p: dict) -> str:
    return (f"With {int(p['n'])} meetings, a one-sided 5% test has about 80% power only if the text model lowers "
            f"the mean squared error by at least {p['mde_mse']:.4f} ({p['mde_pct_of_mse']:.1f}% of the "
            f"statistics-only MSE {p['mse_base']:.4f}; RMSE from {p['rmse_base']:.3f} to {p['rmse_at_mde']:.3f}). "
            f"Computed from the spread of the squared-error differential of the committed Plan 1 per-meeting "
            f"forecasts (S vs S+N, sd {p['sd_loss_diff']:.4f}), normal approximation.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", action="store_true", help="write the registration power statement only")
    args = ap.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.power:
        load = lambda n: pd.read_csv(config.RESULTS / f"preds_T2_{n}.csv", parse_dates=["date"])
        p = power_statement(load("S"), load("S+N"))
        pd.DataFrame([p]).to_csv(RESULTS / "power.csv", index=False)
        print(power_sentence(p))
        return
    reason = gate()
    if reason:
        raise PermissionError(f"Study 3 forecasting refused: {reason}")
    dec, shu, orig = panels(pd.read_pickle(OUT_PATH), pd.read_pickle(SHUFFLED_PATH))
    print(f"{len(dec)} common rows, {dec.index.min().date()} .. {dec.index.max().date()}", flush=True)
    f = forecasts(dec, shu, orig)
    for name, p in f.items():
        p.to_csv(RESULTS / f"preds_T2_{name}.csv", index=False)
    table = hypothesis_tests(f)
    table.to_csv(RESULTS / "tests.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src /d/gold_lm/venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_study3.py`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fed_news/study3.py tests/test_study3.py
git commit -m "Study 3: same-row forecasts, registered tests and power statement

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Results report

**Files:**
- Create: `src/fed_news/evaluate_study3.py`
- Test: `tests/test_evaluate_study3.py`

**Interfaces:**
- Consumes: `decision_text.OSF_NAME, OUT_PATH, RESULTS, TAG`; `evaluate._git(repo, *args) -> str`, `evaluate.metrics`; `study3.decide_h7, power_sentence`; files `tests.csv`, `preds_T2_{model}.csv`, `probe_quality.csv`, `power.csv` in the results folder.
- Produces: `report(results=RESULTS, repo=config.REPO, data_path=OUT_PATH) -> str`; CLI writes `RESULTS_STUDY3.md`.

- [ ] **Step 1: Write the failing tests**

`tests/test_evaluate_study3.py`:

```python
import numpy as np
import pandas as pd

from fed_news import evaluate_study3 as ev
from fed_news import study3 as s3


def write_results(folder, p_main, p_placebo):
    folder.mkdir()
    n = 30
    y = np.zeros(n)
    for m in ev.MODELS:
        pd.DataFrame({"date": pd.date_range("2001-01-01", periods=n, freq="45D"), "y_true": y,
                      "y_pred": np.full(n, 0.1), "lag": 0.0, "class_true": 1, "class_pred": 1}
                     ).to_csv(folder / f"preds_T2_{m}.csv", index=False)
    names = ["H7 Clark-West", "placebo Clark-West", "shuffled-text Clark-West", "S+M+D vs S+M Clark-West",
             "decisions McNemar", "DM S+D vs S", "DM S+D vs S+N"]
    p = [p_main, p_placebo, 0.5, 0.5, 0.5, 0.5, 0.5]
    pd.DataFrame({"test": names, "p": p, "stat": 0.0}).to_csv(folder / "tests.csv", index=False)
    pd.DataFrame({"year": [2001], "accuracy": [0.6]}).to_csv(folder / "probe_quality.csv", index=False)
    pd.DataFrame([s3.power_statement(
        pd.read_csv(folder / "preds_T2_S.csv"), pd.read_csv(folder / "preds_T2_S+D.csv").assign(y_pred=0.0))]
    ).to_csv(folder / "power.csv", index=False)


def test_report_states_verdict_and_registration(tmp_path):
    write_results(tmp_path / "res", 0.2, 0.5)
    (tmp_path / "PREREG_STUDY3_OSF.txt").write_text("https://osf.io/abcd1/", encoding="utf-8")
    text = ev.report(results=tmp_path / "res", repo=tmp_path, data_path=tmp_path / "missing.pkl")
    assert "**H7 verdict: not supported.**" in text
    assert "https://osf.io/abcd1/" in text and "30 meetings" in text


def test_report_void_when_placebo_passes(tmp_path):
    write_results(tmp_path / "res", 0.01, 0.01)
    text = ev.report(results=tmp_path / "res", repo=tmp_path, data_path=tmp_path / "missing.pkl")
    assert "**H7 verdict: void.**" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src /d/gold_lm/venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_evaluate_study3.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'fed_news.evaluate_study3'`.

- [ ] **Step 3: Write the implementation**

`src/fed_news/evaluate_study3.py`:

```python
"""RESULTS_STUDY3.md from results/study3/ (Study 3, H7; PREREG_STUDY3.md)."""
import hashlib

import pandas as pd

from fed_news import config
from fed_news.decision_text import OSF_NAME, OUT_PATH, RESULTS, TAG
from fed_news.evaluate import _git, metrics
from fed_news.study3 import decide_h7, power_sentence

MODELS = ("S", "S+D", "placebo", "shuffled-text", "S+M", "S+M+D", "S+N")


def sha16(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else "missing"


def report(results=RESULTS, repo=config.REPO, data_path=OUT_PATH) -> str:
    tests = pd.read_csv(results / "tests.csv")
    p = tests.set_index("test")["p"]
    verdict = decide_h7(p["H7 Clark-West"], p["placebo Clark-West"], p["shuffled-text Clark-West"])
    table = pd.DataFrame([{"model": m, **metrics(pd.read_csv(results / f"preds_T2_{m}.csv", parse_dates=["date"]))}
                          for m in MODELS])
    quality = pd.read_csv(results / "probe_quality.csv")
    power = pd.read_csv(results / "power.csv").iloc[0].to_dict()
    osf_path = repo / OSF_NAME
    osf = osf_path.read_text(encoding="utf-8").strip() if osf_path.exists() else "missing"
    tag = _git(repo, "rev-parse", "--verify", "--quiet", f"refs/tags/{TAG}^{{commit}}") or "no tag"
    lines = [
        "# Results: Study 3 (ChronoBERT trained on Fed decisions, H7)", "",
        f"**H7 verdict: {verdict}.**", "",
        "H7 (PREREG_STUDY3.md): on the 2001-2026 FOMC meetings, the penalised ordered logit with the 16 real-time "
        "statistics plus the decision-trained text signal (S+D) forecasts the target rate better than the same "
        "model with the statistics only (S): one-sided Clark-West, p < 0.05; void if the placebo or the "
        "shuffled-label text model passes the same test.", "",
        "## Tests", "", tests.round(4).to_markdown(index=False), "",
        "## Forecast metrics (same rows for every model)", "", table.round(4).to_markdown(index=False), "",
        "## Text model quality (out of sample, per release year)", "", quality.round(4).to_markdown(index=False), "",
        "## Power (registered before the run)", "", power_sentence(power), "",
        "## Disclosures", "",
        "- The 2001-2026 meetings were already used to evaluate earlier models in this repository.",
        "- Documents released before 2000 are embedded with the 1999 ChronoBERT vintage (training rows only).",
        "- Document labels are noisy: most documents precede a hold, and the next decision depends mostly on data.",
        "- The code and analyses were produced with an AI coding agent (Claude Code) under the author's direction.", "",
        "## Provenance", "",
        f"- {TAG} tag commit: {tag}",
        f"- OSF registration: {osf}",
        f"- HEAD: {_git(repo, 'rev-parse', 'HEAD')}",
        f"- sha256 (first 16 hex) doc_decision.pkl: {sha16(data_path)}", ""]
    return "\n".join(lines)


def main() -> None:
    text = report()
    (config.REPO / "RESULTS_STUDY3.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src /d/gold_lm/venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_evaluate_study3.py`
Expected: 2 passed.

Then the full offline suite: `PYTHONPATH=src /d/gold_lm/venv/Scripts/python.exe -m pytest -q -p no:cacheprovider`
Expected: 250 passed (225 earlier + 25 new), 11 deselected.

- [ ] **Step 5: Commit**

```bash
git add src/fed_news/evaluate_study3.py tests/test_evaluate_study3.py
git commit -m "Study 3: results report

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Compute the embeddings (GPU, no outcome)

**Files:** none in git (cache in `D:/fed_news/decision_emb/`).

- [ ] **Step 1: Start the embedding run in the background**

Run (background; about 2-3 hours):
`cd /c/Users/User/fed_news && PYTHONPATH=src /d/gold_lm/venv/Scripts/python.exe -m fed_news.decision_text --embed-only > logs/study3_embed.log 2>&1`
If the harness kills background jobs for memory, restart it detached with `nohup ... &` and wait on the log instead.

- [ ] **Step 2: Verify the cache**

Run: `ls D:/fed_news/decision_emb/*.npz | wc -l` and `tail -3 logs/study3_embed.log`
Expected: `26` files; the last line reads `vintage 2024: 6140 documents`.

---

### Task 7: Power statement, pre-registration document and tag

**Files:**
- Create: `PREREG_STUDY3.md`
- Create: `results/study3/power.csv` (by command)

- [ ] **Step 1: Compute the power statement**

Run: `PYTHONPATH=src /d/gold_lm/venv/Scripts/python.exe -m fed_news.study3 --power`
Expected: one printed sentence beginning `With 212 meetings, a one-sided 5% test has about 80% power only if ...`; `results/study3/power.csv` written.

- [ ] **Step 2: Write `PREREG_STUDY3.md`**

Copy this text exactly, replacing the line `POWER SENTENCE` with the sentence printed in Step 1:

```markdown
# Pre-registration: Study 3 (ChronoBERT trained on Fed decisions)

Registered before any classifier is trained on real decision labels. Spec:
docs/superpowers/specs/2026-10-11-study3-decision-trained-text-design.md. Code: src/fed_news/decision_embed.py,
decision_text.py, study3.py, evaluate_study3.py at the commit tagged `prereg-study3`.

## Hypothesis H7 (the only confirmatory test)

On the FOMC meetings from 2001-01-03 to 2026-09-16 that have a text signal (expected: all 212), the penalised
ordered logit with the 16 real-time statistics plus the decision-trained text signal (S+D) forecasts the target
rate better than the same model with the statistics only (S), both on exactly the same rows: one-sided
Clark-West test on squared target-rate errors, p < 0.05.

Void (counts as not supported) if either control passes the same test against S at p < 0.05:
(a) placebo: S+D with the signal permuted across rows inside each training window; (b) shuffled-label text
model: the whole text pipeline re-run with document labels permuted inside each year's training set (seed 0).

## Method (fixed)

- Documents: 6,140 (Fed statements, minutes, press conferences, Board speeches; regional presidents' speeches
  from FRASER and BIS), `stance2.all_docs()`.
- Label: y_class (cut / hold / hike, +/-0.125 band) of the first meeting strictly after the release date.
- Year rule: the classifier of release year y is trained only on documents whose label meeting is before
  1 January of y; embeddings from ChronoBERT vintage min(max(y - 1, 1999), 2024); it scores the documents of
  year y. A year needs at least 100 training documents and at least 10 per class, otherwise its documents stay
  unscored.
- Embedding: frozen ChronoBERT; windows of at most 512 tokens, mean-pooled; document = mean of windows.
- Classifier: multinomial logistic regression, class_weight balanced, standardised embeddings; C from
  {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1} by 5-fold expanding-window cross-validation (log-likelihood) over
  training documents in label-date order, using only folds whose training part has all three classes;
  C = 0.01 if no fold qualifies.
- Document score: P(hike) - P(cut). Meeting signal: mean of group means (statements + press conferences,
  minutes, speeches) over [previous cutoff, cutoff), carried forward when empty.
- Forecasting model: Plan 1 penalised ordered logit and walk-forward engine, unchanged; test start 2001-01-01.

## Secondary analyses (reported, not decisive)

McNemar on correct decisions (S+D vs S); DM-HLN (S+D vs S); Clark-West S+M+D vs S+M; DM-HLN S+D vs S+N (Plan 1
news features); per-year out-of-sample document accuracy and macro-F1 against the majority class.

## Power

POWER SENTENCE

A non-significant result will be reported as inconclusive, not as evidence of no effect.

## Disclosures

- The 2001-2026 meetings were used to evaluate earlier models in this repository; their results (including
  the Plan 1 per-meeting text tests, DM p = 0.061, Clark-West p = 0.001) are known to the authors.
- The embeddings were computed before this registration; they use no outcome.
- Documents released before 2000 are embedded with the 1999 vintage (training rows only).
- The code was written with an AI coding agent (Claude Code) under the author's direction.

## Deviations

Any change after the tag is listed in DEVIATIONS.md under "Study 3".
```

- [ ] **Step 3: Commit and tag**

```bash
git add PREREG_STUDY3.md results/study3/power.csv
git commit -m "Pre-register Study 3 (H7)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git tag prereg-study3
git push origin master --tags
```

- [ ] **Step 4: Stop for the author's OSF registration**

The author registers `PREREG_STUDY3.md` on https://osf.io (Registrations, OSF Preregistration or Open-Ended
Registration; attach the file and the tag commit hash), then gives the URL. Write it to
`PREREG_STUDY3_OSF.txt` (one line, `https://osf.io/...`), commit with message `Record Study 3 OSF registration`
and push. Do not continue until this file exists.

---

### Task 8: Real run, report and documentation

**Files:**
- Create: `RESULTS_STUDY3.md` (by command), `results/study3/*`
- Modify: `README.md` (status table, a Study 3 section, layout list), `DEVIATIONS.md` (append "Study 3" notes)

- [ ] **Step 1: Train the decision classifiers and score documents**

Run: `PYTHONPATH=src /d/gold_lm/venv/Scripts/python.exe -m fed_news.decision_text`
This fits about 2,000 logistic regressions on CPU (up to 1-2 hours). Expected: two lines `doc_decision.pkl: N of 6140 documents scored` and `doc_decision_shuffled.pkl: N of 6140 documents scored` with the same N; `results/study3/probe_quality.csv` exists.

- [ ] **Step 2: Forecast and test**

Run: `PYTHONPATH=src /d/gold_lm/venv/Scripts/python.exe -m fed_news.study3`
Expected: a line `N common rows, YYYY-MM-DD .. 2026-09-16` (N includes training meetings), then a 7-row test table with the same `n` (expected 212) in every row; `results/study3/tests.csv` and 7 prediction files exist.

- [ ] **Step 3: Write the report**

Run: `PYTHONPATH=src /d/gold_lm/venv/Scripts/python.exe -m fed_news.evaluate_study3`
Expected: `RESULTS_STUDY3.md` starts with `# Results: Study 3` and states the H7 verdict and the OSF URL.

- [ ] **Step 4: Update README and DEVIATIONS**

In `README.md`: add a status-table row `| Study 3 (H7, tag prereg-study3, OSF) | Does a ChronoBERT signal trained on Fed decisions help the per-meeting model? | <verdict from RESULTS_STUDY3.md> | <Clark-West p, n> |`; add `decision_embed.py`, `decision_text.py`, `study3.py`, `evaluate_study3.py` to the layout list; add a short "Study 3" subsection under section 7 describing the architecture in the spec's section 3 diagram and linking `RESULTS_STUDY3.md`; add the three Study 3 commands to the run order. In `DEVIATIONS.md` append `### Study 3 notes` listing any change after the tag, or "None".

- [ ] **Step 5: Full test suite, commit, push**

```bash
PYTHONPATH=src /d/gold_lm/venv/Scripts/python.exe -m pytest -q -p no:cacheprovider
git add RESULTS_STUDY3.md results/study3 README.md DEVIATIONS.md
git commit -m "Record Study 3 results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin master
```
Expected: all offline tests pass; `git status --porcelain` empty; `git rev-parse HEAD` equals `git rev-parse origin/master`.
