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
from scipy import stats
from sklearn.metrics import f1_score

from fed_news import config
from fed_news.models import linear as models
from fed_news.features.stance import vintage_year

TAG = "prereg-study3"
OSF_NAME = "PREREG_STUDY3_OSF.txt"
OUT_PATH = config.DATA / "doc_decision.pkl"
SHUFFLED_PATH = config.DATA / "doc_decision_shuffled.pkl"
RESULTS = config.RESULTS / "study3"
CLASSES = (0, 1, 2)
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
FALLBACK_C = 0.01
MIN_DOCS, MIN_PER_CLASS = 100, 10
POC_LAST_YEAR = 2000
CENTRE_DAYS, CENTRE_MIN = 180, 5


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


def centred_embedder(docs: pd.DataFrame, embed, window_days: int = CENTRE_DAYS, min_docs: int = CENTRE_MIN):
    """The "change in language" view: a document's embedding minus the mean embedding (same vintage) of the documents
    released in the `window_days` strictly before its release date; NaN with fewer than `min_docs` of them."""
    order = docs.sort_values("release")
    release, ids = order["release"].to_numpy(dtype="datetime64[ns]"), order["doc_id"].to_numpy()
    when = dict(zip(ids, release))
    memo: dict[tuple[int, str], np.ndarray] = {}

    def centred(vintage: int, doc_ids: list[str]) -> np.ndarray:
        missing = [d for d in doc_ids if (vintage, d) not in memo]
        if missing:
            raw = embed(vintage, missing)
            for d, x in zip(missing, raw):
                r = when[d]
                lo, hi = np.searchsorted(release, r - np.timedelta64(window_days, "D")), np.searchsorted(release, r)
                prior = ids[lo:hi].tolist()
                memo[(vintage, d)] = (x - embed(vintage, prior).mean(axis=0) if len(prior) >= min_docs
                                      else np.full_like(x, np.nan, dtype=float))
        return np.stack([memo[(vintage, d)] for d in doc_ids])

    return centred


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
        X = embed(v, train["doc_id"].tolist())
        finite = np.isfinite(X).all(axis=1)              # documents that cannot be embedded are skipped
        train, X = train[finite], X[finite]
        if not enough(train):
            rows.append({**row, "classifier": False})
            continue
        y = train["label"].to_numpy(int)
        if shuffle_seed is not None:
            y = np.random.default_rng([shuffle_seed, year]).permutation(y)
        model, mu, sd, c = fit_probe(X, y)
        Xs = embed(v, docs.iloc[here]["doc_id"].tolist())
        ok = np.isfinite(Xs).all(axis=1)
        here, Xs = here[ok], Xs[ok]
        ydocs = docs.iloc[here]
        proba = model.predict_proba((Xs - mu) / sd)
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


def run(docs: pd.DataFrame | None = None, meetings: pd.DataFrame | None = None, embed=None,
        out_path=OUT_PATH, shuffled_path=SHUFFLED_PATH, results=RESULTS, repo=config.REPO) -> None:
    reason = gate(repo)
    if reason:
        raise PermissionError(f"Study 3 real-label run refused: {reason}")
    if docs is None:
        from fed_news.studies.study2_regional.stance2 import all_docs
        docs = all_docs()
    if not docs["doc_id"].is_unique:
        raise ValueError("duplicate doc_id in the document table")
    meetings = pd.read_pickle(config.DATA / "meetings.pkl") if meetings is None else meetings
    if embed is None:
        from fed_news.studies.study3_decision_text.decision_embed import cached_embedder
        embed = cached_embedder(docs)
    labelled = next_decision(docs.drop(columns=["text"], errors="ignore"), meetings)
    results.mkdir(parents=True, exist_ok=True)
    for path, seed, name in ((out_path, None, "probe_quality.csv"),
                             (shuffled_path, 0, "probe_quality_shuffled.csv")):
        scored, quality = score_all(labelled, embed, shuffle_seed=seed)
        scored.to_pickle(path)
        quality.to_csv(results / name, index=False)
        print(f"{path.name}: {int(scored['stance'].notna().sum())} of {len(scored)} documents scored", flush=True)


def poc_labels(labelled: pd.DataFrame, last_year: int = POC_LAST_YEAR) -> pd.DataFrame:
    """Documents released up to `last_year`; labels from meetings on or after 1 January of the next year are
    removed, so the proof of concept never reads an outcome from the 2001-2026 test window."""
    cut = pd.Timestamp(f"{last_year + 1}-01-01")
    docs = labelled[labelled["release"] < cut]
    late = (docs["label_date"] >= cut).to_numpy()
    return docs.assign(label=np.where(late, -1, docs["label"].to_numpy()),
                       label_date=docs["label_date"].where(~late))


def poc_summary(scored: pd.DataFrame, quality: pd.DataFrame) -> dict:
    """Pass mark fixed before running: pooled out-of-sample accuracy above the majority class and a positive
    rank correlation between the score and the next decision (cut < hold < hike)."""
    q = quality[(quality["classifier"] == True) & quality["accuracy"].notna()]  # noqa: E712
    n = q["n_scored_labelled"].astype(float)
    acc = float((q["accuracy"] * n).sum() / n.sum())
    maj = float((q["majority_accuracy"] * n).sum() / n.sum())
    known = scored[(scored["label"] >= 0) & scored["stance"].notna()]
    rho = float(stats.spearmanr(known["stance"], known["label"]).statistic)
    return {"years": f"{int(q['year'].min())}-{int(q['year'].max())}", "n_docs": int(n.sum()), "accuracy": acc,
            "majority_accuracy": maj, "spearman": rho, "passed": bool(acc > maj and rho > 0)}


def poc(docs: pd.DataFrame | None = None, meetings: pd.DataFrame | None = None, embed=None,
        results=RESULTS, centre: bool = False, old_scores: pd.DataFrame | None = None, prefix: str = "poc") -> dict:
    """Feasibility check before registration, on release years up to POC_LAST_YEAR only (no gate needed)."""
    if docs is None:
        from fed_news.studies.study2_regional.stance2 import all_docs
        docs = all_docs()
    if not docs["doc_id"].is_unique:
        raise ValueError("duplicate doc_id in the document table")
    meetings = pd.read_pickle(config.DATA / "meetings.pkl") if meetings is None else meetings
    if embed is None:
        from fed_news.studies.study3_decision_text.decision_embed import cached_embedder
        embed = cached_embedder(docs)
    if centre:
        embed = centred_embedder(docs, embed)
    labelled = poc_labels(next_decision(docs.drop(columns=["text"], errors="ignore"), meetings))
    scored, quality = score_all(labelled, embed)
    summary = poc_summary(scored, quality)
    if old_scores is not None:  # attempt-2 pass mark: beat the old hawk/dove score on the same documents
        both = scored[(scored["label"] >= 0) & scored["stance"].notna()].merge(
            old_scores[["doc_id", "stance"]].dropna(), on="doc_id", suffixes=("", "_old"))
        rho = float(stats.spearmanr(both["stance"], both["label"]).statistic)
        rho_old = float(stats.spearmanr(both["stance_old"], both["label"]).statistic)
        summary.update({"n_common": len(both), "spearman_common": rho, "spearman_old": rho_old,
                        "passed_v2": bool(rho > 0 and rho > rho_old)})
    if results is not None:
        results.mkdir(parents=True, exist_ok=True)
        quality.to_csv(results / f"{prefix}_quality.csv", index=False)
        pd.DataFrame([summary]).to_csv(results / f"{prefix}_summary.csv", index=False)
    print(quality.round(3).to_string(index=False), flush=True)
    print(summary, flush=True)
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-only", action="store_true", help="compute ChronoBERT embeddings only (no outcome)")
    ap.add_argument("--poc", action="store_true", help="proof of concept on release years up to 2000 only")
    ap.add_argument("--poc-centred", action="store_true", help="proof of concept, attempt 2: change in language")
    args = ap.parse_args(argv)
    if args.poc:
        poc()
    elif args.poc_centred:
        poc(centre=True, old_scores=pd.read_pickle(config.DATA / "doc_stance2.pkl"), prefix="poc2")
    elif args.embed_only:
        from fed_news.studies.study3_decision_text.decision_embed import embed_all
        from fed_news.studies.study2_regional.stance2 import all_docs
        embed_all(all_docs())
    else:
        run()


if __name__ == "__main__":
    main()
