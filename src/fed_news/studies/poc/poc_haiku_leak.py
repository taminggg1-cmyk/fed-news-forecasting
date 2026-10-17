"""Haiku as a direct sentence scorer, step 1 (exploratory): leak mitigation and leak tests
(docs/superpowers/specs/2026-10-12-poc-haiku-leak-design.md).

Test A: can Haiku date masked sentences (more than 50% within 2 years means the masking fails; step 2 is off).
Test B: do Haiku's labels track the next FOMC decision beyond the human label and leak-free ChronoBERT (flag).
"""
import glob
import json
import re

import numpy as np
import pandas as pd

from fed_news import config
from fed_news.studies.poc import poc_llm_labels as pl
from fed_news.studies.poc.stmt_change import normalise

OUT = config.RESULTS / "poc_haiku"
N_DATE, SEED, BATCH, CAP = 300, 0, 25, 400_000
WITHIN, FAIL_SHARE = 2, 0.50
KNOWN_NAMES = {"greenspan", "bernanke", "yellen", "powell"}
NOT_NAMES = {"affairs"}  # last word of a job-title speaker field, not a surname

_EVENTS = re.compile(r"\b(lehman(?: brothers)?|bear stearns|aig|fannie mae|freddie mac|katrina|rita|september 11|9/11"
                     r"|enron|worldcom|y2k|covid(?:-19)?|coronavirus|pandemic|omicron|brexit|ukraine"
                     r"|silicon valley bank|svb)\b")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_MONTH = re.compile(r"\b(?:january|february|april|june|july|august|september|october|november|december)\b")
_AMBIG_BEFORE = re.compile(r"\b(in|since|during|through|until|by|of|early|late|mid|from|last|next|between)\s+"
                           r"(?:may|march)\b")
_AMBIG_AFTER = re.compile(r"\b(?:may|march)(?=\s+(?:\[|\d))")
_NUM = re.compile(r"\d[\d\-/.,]*\d|\d")

DATE_SYSTEM = (
    "You estimate when sentences from US Federal Reserve communications were written. Placeholders such as "
    "[YEAR], [MONTH], [NUM], [NAME] and [EVENT] replace removed details. For each numbered sentence, give your "
    "best guess of the calendar year it was written. Reply with one JSON object mapping each sentence number to a "
    "four-digit year, and nothing else."
)


def official_names(corpus: pd.DataFrame) -> set[str]:
    names = set(KNOWN_NAMES)
    for s in corpus["speaker"].dropna().unique():
        parts = [p for p in re.split(r"[\s,]+", s.strip()) if p and p.rstrip(".") not in ("Jr", "Sr")]
        if parts:
            names.add(parts[-1].lower().rstrip("."))
    return names - NOT_NAMES


def mask(sentence: str, names: set[str]) -> str:
    s = _EVENTS.sub("[EVENT]", sentence.lower())
    if names:
        s = re.sub(r"\b(?:" + "|".join(sorted(map(re.escape, names))) + r")\b", "[NAME]", s)
    s = _YEAR.sub("[YEAR]", s)
    s = _MONTH.sub("[MONTH]", s)
    s = _AMBIG_BEFORE.sub(lambda m: f"{m.group(1)} [MONTH]", s)
    s = _AMBIG_AFTER.sub("[MONTH]", s)
    return _NUM.sub("[NUM]", s)


def parse_years(text: str, n: int) -> list[int]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    obj = {str(k).strip(): v for k, v in json.loads(m.group(0)).items()}
    out = []
    for i in range(1, n + 1):
        try:
            out.append(int(str(obj[str(i)]).strip()[:4]))
        except (KeyError, ValueError):
            raise ValueError(f"sentence {i}: no year") from None
    return out


def next_decision(releases, meetings: pd.DataFrame) -> np.ndarray:
    """Direction of the first decision strictly after each release: cut -1, hold 0, hike +1 (+/-0.125 band)."""
    dates = meetings.index.sort_values()
    pos = np.searchsorted(dates.values, pd.DatetimeIndex(releases).values, side="right")
    ch = np.array([meetings["y_change"].loc[dates[p]] if p < len(dates) else np.nan for p in pos], dtype=float)
    return np.where(np.isnan(ch), np.nan, np.where(ch > config.CLASS_BAND, 1, np.where(ch < -config.CLASS_BAND, -1, 0)))


def next_meeting(releases, meetings: pd.DataFrame) -> np.ndarray:
    dates = meetings.index.sort_values()
    pos = np.searchsorted(dates.values, pd.DatetimeIndex(releases).values, side="right")
    return np.array([dates[p] if p < len(dates) else pd.NaT for p in pos])


def hindsight(df: pd.DataFrame) -> dict:
    import statsmodels.api as sm
    d = df.dropna(subset=["next", "human", "haiku", "chrono"]).copy()
    for c in ("haiku", "chrono"):
        d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std()
    X = sm.add_constant(d[["human", "haiku_z", "chrono_z"]].astype(float))
    fit = sm.OLS(d["next"].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["meeting"])[0]})
    diff = fit.t_test("haiku_z - chrono_z = 0")
    p_diff, est = float(np.asarray(diff.pvalue)), float(np.asarray(diff.effect).ravel()[0])
    return {"n": int(len(d)), "n_meetings": int(d["meeting"].nunique()),
            "coef_human": float(fit.params["human"]), "coef_haiku": float(fit.params["haiku_z"]),
            "coef_chrono": float(fit.params["chrono_z"]), "p_haiku": float(fit.pvalues["haiku_z"]),
            "p_chrono": float(fit.pvalues["chrono_z"]), "haiku_minus_chrono": est, "p_difference": p_diff,
            "flag": bool(fit.params["haiku_z"] > 0 and est > 0 and p_diff < 0.05)}


def _ask_years(client, sentences, usage) -> list[int] | None:
    for _ in range(2):
        if usage["input"] + usage["output"] >= CAP:
            return None
        r = client.messages.create(model=pl.MODEL, system=DATE_SYSTEM, **pl.ATTEMPTS[1],
                                   messages=[{"role": "user", "content": pl.batch_prompt(sentences)}])
        usage["input"] += r.usage.input_tokens
        usage["output"] += r.usage.output_tokens
        if r.stop_reason in ("refusal", "max_tokens"):
            continue
        try:
            return parse_years("".join(b.text for b in r.content if b.type == "text"), len(sentences))
        except (ValueError, json.JSONDecodeError):
            continue
    return None


def _batched(fn, sentences, usage, label):
    out = []
    for s in range(0, len(sentences), BATCH):
        part = sentences[s:s + BATCH]
        got = fn(part)
        out += got if got is not None else [None] * len(part)
        if (s // BATCH) % 10 == 0:
            print(f"{label}: {s + len(part)} of {len(sentences)}; tokens {usage}", flush=True)
    return out


def chrono_scores(labels: pd.DataFrame) -> np.ndarray:
    """Saved ChronoBERT score of each labelled sentence from the classifier of its own year (never trained on it)."""
    parts = {}
    for f in glob.glob(str(config.DATA / "stance" / "*.pkl")):
        frame = pd.read_pickle(f)[0]
        parts[int(f[-8:-4])] = dict(zip(frame["sentence"].map(normalise), frame["stance"]))
    return np.array([parts.get(max(int(y), config.FIRST_CLASSIFIER_YEAR), {}).get(normalise(s), np.nan)
                     for s, y in zip(labels["sentence"], labels["year"])])


def run() -> dict:
    from fed_news.collect.labels import load_labels
    labels, corpus = load_labels(), pd.read_pickle(config.DATA / "corpus.pkl")
    meetings = pd.read_pickle(config.DATA / "meetings.pkl")
    names, client, usage = official_names(corpus), pl._client(), {"input": 0, "output": 0}
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"model": pl.MODEL, "n_names_masked": len(names)}

    # Test A
    a = labels.sample(N_DATE, random_state=SEED).reset_index(drop=True)
    guess = _batched(lambda b: _ask_years(client, b, usage), [mask(s, names) for s in a["sentence"]], usage, "dating")
    g = np.array([np.nan if x is None else x for x in guess], dtype=float)
    err = np.abs(g - a["year"].to_numpy())
    base = np.abs(np.median(labels["year"]) - a["year"].to_numpy())
    res["test_a"] = {"n": N_DATE, "n_unanswered": int(np.isnan(g).sum()),
                     "share_within_2y": float(np.nanmean(err <= WITHIN)) if (~np.isnan(g)).any() else None,
                     "median_abs_error": float(np.nanmedian(err)),
                     "baseline_share_within_2y": float((base <= WITHIN).mean()),
                     "baseline_median_abs_error": float(np.median(base))}
    res["test_a"]["masking_fails"] = bool(res["test_a"]["share_within_2y"] is None
                                          or res["test_a"]["share_within_2y"] > FAIL_SHARE)
    a.assign(masked=[mask(s, names) for s in a["sentence"]], guess=g)[["sentence", "masked", "year", "guess"]].to_csv(
        OUT / "test_a.csv", index=False)
    print(json.dumps(res["test_a"]), flush=True)

    # Test B: all labels, masked, in random order
    order = np.random.default_rng(SEED).permutation(len(labels))
    masked = [mask(s, names) for s in labels["sentence"].iloc[order]]
    got = _batched(lambda b: pl._ask(client, b, usage, pl.ATTEMPTS[1], cap=CAP), masked, usage, "labelling")
    haiku = np.full(len(labels), np.nan)
    haiku[order] = [np.nan if x is None else {0: -1, 1: 1, 2: 0}[x] for x in got]
    df = pd.DataFrame({"next": next_decision(labels["release"], meetings),
                       "human": labels["label"].map({0: -1, 1: 1, 2: 0}).to_numpy(float),
                       "haiku": haiku, "chrono": chrono_scores(labels),
                       "meeting": next_meeting(labels["release"], meetings)})
    df.assign(sentence=labels["sentence"].to_numpy(), year=labels["year"].to_numpy()).to_csv(OUT / "test_b.csv",
                                                                                           index=False)
    res["test_b"] = {"n_labels": int(len(labels)), "n_haiku_missing": int(np.isnan(haiku).sum()),
                     "n_chrono_missing": int(df["chrono"].isna().sum()), **hindsight(df)}
    res["tokens"] = usage
    (OUT / "step1.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str), flush=True)
    return res


if __name__ == "__main__":
    run()
