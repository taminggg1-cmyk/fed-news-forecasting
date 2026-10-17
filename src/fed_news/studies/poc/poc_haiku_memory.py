"""Does Claude Haiku 5.5 remember what the Fed did next? (docs/superpowers/specs/2026-10-12-poc-haiku-memory-design.md)

Exam 0 (descriptive): recall decisions from dates alone. Exam 1: next decision from masked statements vs the
persistence rule (McNemar). Exam 2: is Exam 1 accuracy higher on statements Haiku dates correctly (Fisher)?
Passes ("no detectable memory") only if Exam 1 is not significant and Exam 2 raises no flag.
"""
import json
import re

import numpy as np
import pandas as pd
from scipy import stats

from fed_news import config
from fed_news.studies.poc import poc_llm_labels as pl
from fed_news.studies.poc.poc_haiku_leak import mask, official_names, parse_years
from fed_news.evaluation.stats_tests import mcnemar_greater

OUT = config.RESULTS / "poc_haiku"
BATCH_TEXT, BATCH_DATES, SEED, CAP = 5, 25, 0, 400_000
DATED_WITHIN, FLAG_GAP = 1, 0.10
LETTERS = {"C": -1, "H": 0, "R": 1}

NEXT_SYSTEM = (
    "Each numbered item is a statement issued by the US Federal Open Market Committee after a policy meeting. "
    "Dates, numbers, names and named events are replaced by placeholders. For each item, predict the Committee's "
    "decision at its next meeting: C (cut rates), H (hold) or R (raise rates). Reply with one JSON object "
    "mapping each item number to C, H or R, and nothing else."
)
YEAR_SYSTEM = (
    "Each numbered item is a statement issued by the US Federal Open Market Committee, with dates, numbers, names "
    "and named events replaced by placeholders. For each item, give your best guess of the calendar year it was "
    "issued. Reply with one JSON object mapping each item number to a four-digit year, and nothing else."
)
RECALL_SYSTEM = (
    "Each numbered item is the date of a US Federal Open Market Committee policy decision. For each, state what "
    "the Committee decided on that date: C (cut rates), H (hold) or R (raise rates). Reply with one JSON object "
    "mapping each item number to C, H or R, and nothing else."
)


def prepare(text: str, names: set[str]) -> str:
    cut = re.search(r"\bvoting\b", text, re.I)
    return mask(text[: cut.start()] if cut else text, names).strip()


def parse_next(text: str, n: int) -> list[int]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    obj = {str(k).strip(): str(v).strip().upper()[:1] for k, v in json.loads(m.group(0)).items()}
    out = []
    for i in range(1, n + 1):
        if obj.get(str(i)) not in LETTERS:
            raise ValueError(f"item {i}: {obj.get(str(i))!r}")
        out.append(LETTERS[obj[str(i)]])
    return out


def direction(change) -> np.ndarray:
    change = np.asarray(change, float)
    return np.where(change > config.CLASS_BAND, 1, np.where(change < -config.CLASS_BAND, -1, 0))


def verdict(truth, haiku, persistence, dated) -> dict:
    truth, persistence, dated = np.asarray(truth), np.asarray(persistence), np.asarray(dated, bool)
    ok_h = np.asarray(haiku, float) == truth
    ok_p = persistence == truth
    fixes, breaks, p_mc = mcnemar_greater(ok_p, ok_h)
    acc_d, acc_n = float(ok_h[dated].mean()), float(ok_h[~dated].mean())
    table = [[int(ok_h[dated].sum()), int((~ok_h[dated]).sum())], [int(ok_h[~dated].sum()), int((~ok_h[~dated]).sum())]]
    p_f = float(stats.fisher_exact(table, alternative="greater").pvalue)
    flag = bool(acc_d - acc_n >= FLAG_GAP and p_f < 0.05)
    return {"accuracy_haiku": float(ok_h.mean()), "accuracy_persistence": float(ok_p.mean()),
            "haiku_right_persistence_wrong": fixes, "persistence_right_haiku_wrong": breaks, "p_mcnemar": p_mc,
            "n_dated": int(dated.sum()), "accuracy_dated": acc_d, "accuracy_not_dated": acc_n, "p_fisher": p_f,
            "date_dependence_flag": flag, "passed": bool(p_mc >= 0.05 and not flag)}


def _ask(client, system, items, parser, usage, render):
    for _ in range(2):
        if usage["input"] + usage["output"] >= CAP:
            return None
        r = client.messages.create(model=pl.MODEL, system=system, **pl.ATTEMPTS[1],
                                   messages=[{"role": "user", "content": render(items)}])
        usage["input"] += r.usage.input_tokens
        usage["output"] += r.usage.output_tokens
        if r.stop_reason in ("refusal", "max_tokens"):
            continue
        try:
            return parser("".join(b.text for b in r.content if b.type == "text"), len(items))
        except (ValueError, json.JSONDecodeError):
            continue
    return None


def _run(client, system, items, parser, usage, batch, render, label):
    order = np.random.default_rng(SEED).permutation(len(items))
    got = np.full(len(items), np.nan)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        ans = _ask(client, system, [items[i] for i in idx], parser, usage, render)
        if ans is not None:
            got[idx] = ans
        print(f"{label}: {min(s + batch, len(order))} of {len(order)}; tokens {usage}", flush=True)
    return got


def run() -> dict:
    corpus, meetings = pd.read_pickle(config.DATA / "corpus.pkl"), pd.read_pickle(config.DATA / "meetings.pkl")
    dates = meetings.index.sort_values()
    rows = []
    for _, doc in corpus[corpus["type"] == "statement"].sort_values("release").iterrows():
        gap = np.abs((dates - doc["release"]).days)
        if gap.min() <= 1 and int(gap.argmin()) + 1 < len(dates):
            k = int(gap.argmin())
            rows.append({"date": dates[k], "text": doc["text"], "current": meetings["y_change"].loc[dates[k]],
                         "next": meetings["y_change"].loc[dates[k + 1]]})
    items = pd.DataFrame(rows).drop_duplicates("date").reset_index(drop=True)
    names, client, usage = official_names(corpus), pl._client(), {"input": 0, "output": 0}
    masked = [prepare(t, names) for t in items["text"]]
    render_text = lambda xs: "\n\n".join(f"### {i}\n{x}" for i, x in enumerate(xs, 1))
    render_dates = lambda xs: "\n".join(f"{i}. {x}" for i, x in enumerate(xs, 1))
    current, truth = direction(items["current"]), direction(items["next"])

    recall = _run(client, RECALL_SYSTEM, [d.strftime("%Y-%m-%d") for d in items["date"]], parse_next, usage,
                  BATCH_DATES, render_dates, "exam 0")
    nxt = _run(client, NEXT_SYSTEM, masked, parse_next, usage, BATCH_TEXT, render_text, "exam 1")
    years = _run(client, YEAR_SYSTEM, masked, parse_years, usage, BATCH_TEXT, render_text, "exam 2")
    dated = np.abs(years - items["date"].dt.year.to_numpy()) <= DATED_WITHIN

    base0 = float(pd.Series(current).value_counts(normalize=True).max())
    res = {"model": pl.MODEL, "n_statements": int(len(items)),
           "exam0_recall_accuracy": float((recall == current).mean()), "exam0_base_rate_most_common": base0,
           "exam0_p_vs_base": float(stats.binomtest(int((recall == current).sum()), len(items), base0,
                                                    alternative="greater").pvalue),
           "exam2_share_dated_within_1y": float(dated.mean()),
           **verdict(truth, nxt, current, dated), "tokens": usage}
    OUT.mkdir(parents=True, exist_ok=True)
    items.drop(columns="text").assign(masked=masked, recall=recall, haiku_next=nxt, year_guess=years,
                                      truth_next=truth, persistence=current).to_csv(OUT / "memory_exam.csv", index=False)
    (OUT / "memory_exam.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    return res


if __name__ == "__main__":
    run()
