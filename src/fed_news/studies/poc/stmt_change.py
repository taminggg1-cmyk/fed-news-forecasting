"""Proof of concept (exploratory): what changed in FOMC statements
(docs/superpowers/specs/2026-10-12-poc-statement-change-design.md, sections 2 and 6).

x_shift:       sum of stance of sentences added to a statement minus sum of stance of sentences removed,
               against the previous decision-day statement; vote lists, footers, the policy-action sentence and
               discount-rate sentences are excluded. Stance = the saved Plan 1 sentence scores (no retraining).
x_dissent_n:   members voting against, from the statement text.
x_dissent_dir: dissenters preferring tighter minus those preferring easier policy, relative to the action.
Stage 1 checks the measurement only: no forecast outcome is used.
"""
import argparse
import glob
import json
import re

import numpy as np
import pandas as pd

from fed_news import config

OUT = config.RESULTS / "poc_stmt"
WORK = config.DATA / "poc_stmt"
N_LABEL, N_DISSENT, SEED = 40, 20, 0
FIRST_WINDOW_DAYS = 45

_MONTHS = r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b"
_STOP = re.compile(r"\bvot(ing|ed)\b|implementation note|for media inquiries|monetary policy home|last update:")
_DROP = re.compile(r"\bdiscount rate\b|\brelated action\b|\bboard\b.*\bapproved\b"
                   r"|\bdecided\b.*\b(target|federal funds rate)\b")
_HEADER = re.compile(r"^.{0,120}?for immediate release\s*(share\s+)?|^e[ds]t\s+share\s+")
_DISSENT_END = re.compile(r"Implementation Note|For media|In a related action|In taking|\d{4} Monetary policy"
                          r"|Monetary Policy Home|Related Information|Statement Regarding|Return to text", re.I)
_DISSENT_HEAD = re.compile(r"^Voting against[\w ]*?(?:were|was):?\s*")
_NAMES_END = re.compile(r" who |\.\s+(?=(?:Governor|President|Mr\.|Ms\.|Chairman|Chair)\b)")
_NAME_SPLIT = re.compile(r",\s*(?:and\s+)?|\s+and\s+")
_SUFFIX = re.compile(r"^(Jr|Sr)\.?$|^I{2,3}$")
_LESS_ACCOMMODATION = re.compile(r"\bnot support (?:additional|further) (?:policy )?accommodation"
                                 r"|\b(reduc\w*|less|remov\w*|withdraw\w*) (?:in |of )?(?:the )?(?:monetary |policy )?"
                                 r"accommodat")
_EASIER = re.compile(r"\b(lower|reduc\w*|cut|decreas\w*|eas(e|ing)|more accommodat\w*"
                     r"|additional (?:policy )?accommodation|further accommodation)\b")
_TIGHTER = re.compile(r"\b(rais(e|ing)|increas\w*|tighten\w*|firm(ing|er)?|higher)\b")
_SIZE_PP = re.compile(r"\bby (\d+(?:\.\d+)?|1/4|1/2|3/4) percentage point")
_SIZE_BP = re.compile(r"\b(\d+)[ -]basis[ -]points?\b")
_FRACTIONS = {"1/4": 0.25, "1/2": 0.5, "3/4": 0.75}
_HOLD = re.compile(r"\b(no change|maintain\w*|keep|unchanged|leave)\b")


def normalise(sentence: str) -> str:
    s = _HEADER.sub("", sentence.lower().strip())
    s = re.sub(_MONTHS, "#m", s)
    s = re.sub(r"\d[\d\-/.,:]*", "#", s)
    return re.sub(r"\s+", " ", s).strip()


def _kept_pairs(pairs):
    out = []
    for sentence, stance in pairs:
        low = sentence.lower()
        if _STOP.search(low):
            break
        if not _DROP.search(low):
            out.append((sentence, stance))
    return out


def keep_sentences(sentences: list[str]) -> list[str]:
    return [s for s, _ in _kept_pairs((s, 0.0) for s in sentences)]


def _diff(prev, cur) -> tuple[float, list[str], list[str]]:
    p = {normalise(s): (s, st) for s, st in _kept_pairs(prev)}
    c = {normalise(s): (s, st) for s, st in _kept_pairs(cur)}
    added = [c[k] for k in c if k not in p]
    removed = [p[k] for k in p if k not in c]
    shift = sum(st for _, st in added) - sum(st for _, st in removed)
    return float(shift), [s for s, _ in added], [s for s, _ in removed]


def statement_shift(prev, cur) -> float:
    return _diff(prev, cur)[0]


def _size(reason: str) -> float | None:
    m = _SIZE_PP.search(reason)
    if m:
        return _FRACTIONS.get(m.group(1)) or float(m.group(1))
    m = _SIZE_BP.search(reason)
    return float(m.group(1)) / 100 if m else None


def _direction(reason: str, action: float) -> int:
    """+1 tighter, -1 easier than the committee's action, 0 when the text gives no rate direction."""
    r = reason.lower()
    if _LESS_ACCOMMODATION.search(r):
        return 1
    d = -1 if _EASIER.search(r) else (1 if _TIGHTER.search(r) else 0)
    if d:
        size = _size(r)
        if size is not None and action * d > 0:  # same direction as the action: a smaller move is the opposite
            return d if size > abs(action) + 1e-9 else (-d if size < abs(action) - 1e-9 else 0)
        return d
    if _HOLD.search(r):
        return 1 if action < 0 else (-1 if action > 0 else 0)
    return 0


def parse_dissent(text: str, action: float) -> tuple[int, int]:
    start = text.find("Voting against")
    if start < 0:
        return 0, 0
    seg = _DISSENT_END.split(text[start:])[0]
    seg = _DISSENT_HEAD.sub("", seg)
    n_total, direction = 0, 0
    for clause in seg.split(";"):
        cut = _NAMES_END.search(clause)
        names_part, reason = (clause[:cut.start()], clause[cut.end():]) if cut else (clause, "")
        names = [t.strip(" .") for t in _NAME_SPLIT.split(names_part.strip().removeprefix("and ").strip(" ,."))]
        names = [t for t in names if t and t[0].isupper() and not _SUFFIX.match(t)]
        n_total += len(names)
        direction += len(names) * _direction(reason, action)
    return n_total, direction


def meeting_features(stm: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Sum of shifts over statements released in [previous cutoff, cutoff); dissent from the latest one."""
    out, prev = [], None
    for cutoff in rows["cutoff"]:
        start = prev if prev is not None else cutoff - pd.Timedelta(days=FIRST_WINDOW_DAYS)
        win = stm[(stm.index >= start) & (stm.index < cutoff)]
        last = win.iloc[-1] if len(win) else None
        out.append({"x_shift": float(win["shift"].sum()) if len(win) else 0.0,
                    "x_dissent_n": int(last["dissent_n"]) if last is not None else 0,
                    "x_dissent_dir": int(last["dissent_dir"]) if last is not None else 0})
        prev = cutoff
    return pd.DataFrame(out, index=rows.index)


def statement_table(corpus: pd.DataFrame, scores: pd.DataFrame, meetings: pd.DataFrame) -> pd.DataFrame:
    """Decision-day statements in release order, each compared with the previous one."""
    pairs = {d: list(zip(g["sentence"], g["stance"])) for d, g in scores.groupby("doc_id", sort=False)}
    stm = corpus[corpus["type"] == "statement"].sort_values("release")
    decisions = meetings.index
    rows, prev = {}, None
    for _, doc in stm.iterrows():
        gap = np.abs((decisions - doc["release"]).days)
        if gap.min() > 1:
            continue
        action = float(meetings["y_change"].iloc[int(gap.argmin())])
        cur = pairs.get(doc["doc_id"], [])
        shift, added, removed = _diff(prev, cur) if prev is not None else (0.0, [], [])
        n, direction = parse_dissent(doc["text"], action)
        rows[doc["release"]] = {"doc_id": doc["doc_id"], "action": action, "shift": shift, "added": added,
                                "removed": removed, "dissent_n": n, "dissent_dir": direction}
        prev = cur
    return pd.DataFrame.from_dict(rows, orient="index")


def label_sheet(table: pd.DataFrame, n: int = N_LABEL, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A sample of changed pairs without dates or scores (the sheet) and the hidden key."""
    changed = table[(table["added"].str.len() > 0) | (table["removed"].str.len() > 0)]
    pick = changed.sample(n=min(n, len(changed)), random_state=seed)
    ids = np.arange(1, len(pick) + 1)
    sheet = pd.DataFrame({"pair_id": ids, "removed": ["\n".join(r) for r in pick["removed"]],
                          "added": ["\n".join(a) for a in pick["added"]], "label": ""})
    key = pd.DataFrame({"pair_id": ids, "doc_id": pick["doc_id"].to_numpy(), "release": pick.index,
                        "x_shift": pick["shift"].to_numpy()})
    return sheet, key


def stage1_measurement(labels: pd.DataFrame, key: pd.DataFrame) -> dict:
    m = labels.merge(key, on="pair_id")
    m["label"] = m["label"].astype(str).str.strip().str.upper()
    d = m[m["label"].isin(["H", "D"])]
    correct = ((d["label"] == "H") & (d["x_shift"] > 0)) | ((d["label"] == "D") & (d["x_shift"] < 0))
    agreement = float(correct.mean()) if len(d) else 0.0
    majority = float(d["label"].value_counts(normalize=True).max()) if len(d) else 1.0
    return {"n_labelled": int(len(m)), "n_directional": int(len(d)), "agreement": agreement,
            "majority_share": majority, "passed": bool(agreement >= 0.7 and agreement > majority)}


def stage1_human_labels(table: pd.DataFrame, scores: pd.DataFrame,
                        labels: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Design section 10: model vs human shift on changed sentences whose label the scoring classifier never saw."""
    sign = {0: -1.0, 1: 1.0, 2: 0.0}  # labels.LABELS: dovish 0, hawkish 1, neutral 2
    lab = labels.assign(key=labels["sentence"].map(normalise))
    consistent = lab.groupby("key")["label"].nunique() == 1
    ref = lab[lab["key"].map(consistent)].groupby("key").agg(label=("label", "first"), year=("year", "min"))
    stance = {d: {normalise(s): st for s, st in zip(g["sentence"], g["stance"])}
              for d, g in scores.groupby("doc_id", sort=False)}
    clf_year = lambda day: max(day.year, config.FIRST_CLASSIFIER_YEAR)  # as stance.classifier_year
    rows = []
    for k in range(1, len(table)):
        cur, prev = table.iloc[k], table.iloc[k - 1]
        human = model = 0.0
        n = 0
        for sentences, doc, year, sgn in ((cur["added"], cur["doc_id"], clf_year(table.index[k]), 1.0),
                                          (cur["removed"], prev["doc_id"], clf_year(table.index[k - 1]), -1.0)):
            for s in sentences:
                key = normalise(s)
                if key in ref.index and ref.at[key, "year"] >= year and key in stance.get(doc, {}):
                    human += sgn * sign[int(ref.at[key, "label"])]
                    model += sgn * stance[doc][key]
                    n += 1
        if n:
            rows.append({"release": table.index[k], "doc_id": cur["doc_id"], "n_labelled": n,
                         "human_shift": human, "model_shift": model})
    pairs = pd.DataFrame(rows, columns=["release", "doc_id", "n_labelled", "human_shift", "model_shift"])
    d = pairs[pairs["human_shift"] != 0]
    agreement = float((np.sign(d["model_shift"]) == np.sign(d["human_shift"])).mean()) if len(d) else 0.0
    majority = float(np.sign(d["human_shift"]).value_counts(normalize=True).max()) if len(d) else 1.0
    return pairs, {"label_source": "TDW/WCB human sentence labels (design section 10)",
                   "n_pairs_with_labels": int(len(pairs)), "n_directional": int(len(d)),
                   "agreement": agreement, "majority_share": majority,
                   "passed": bool(len(d) >= 20 and agreement >= 0.7 and agreement > majority)}


def score_stage1_human() -> dict:
    from fed_news.labels import load_labels
    _, scores, _ = load_inputs()
    pairs, res = stage1_human_labels(pd.read_pickle(WORK / "statements.pkl"), scores, load_labels())
    pairs.to_csv(OUT / "stage1_human_pairs.csv", index=False)
    check = pd.read_csv(OUT / "dissent_check.csv", encoding="utf-8-sig")
    res["dissent_checked"] = int(check["correct"].notna().sum())
    res["dissent_all_correct"] = bool((check["correct"].astype(str).str.upper() == "Y").all())
    res["stage1_passed"] = bool(res["passed"] and res["dissent_all_correct"] and res["dissent_checked"] == len(check))
    (OUT / "stage1.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    return res


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    corpus = pd.read_pickle(config.DATA / "corpus.pkl")
    scores = pd.concat([pd.read_pickle(f)[0] for f in sorted(glob.glob(str(config.DATA / "stance" / "*.pkl")))])
    return corpus, scores, pd.read_pickle(config.DATA / "meetings.pkl")


def prepare_stage1() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    corpus, scores, meetings = load_inputs()
    table = statement_table(corpus, scores, meetings)
    table.to_pickle(WORK / "statements.pkl")
    sheet, key = label_sheet(table)
    sheet.to_csv(OUT / "label_sheet.csv", index=False, encoding="utf-8-sig")
    key.to_csv(WORK / "label_key.csv", index=False)  # hidden from the labeller: kept outside the repository
    texts = corpus.set_index("doc_id")["text"]
    dis = table[table["doc_id"].map(lambda d: "Voting against" in texts[d])]
    pick = dis.sample(n=min(N_DISSENT, len(dis)), random_state=SEED).sort_index()
    check = pd.DataFrame({"release": pick.index.date, "action": pick["action"].to_numpy(),
                          "passage": [_DISSENT_END.split(texts[d][texts[d].find("Voting against"):])[0].strip()
                                      for d in pick["doc_id"]],
                          "parsed_n": pick["dissent_n"].to_numpy(), "parsed_dir": pick["dissent_dir"].to_numpy(),
                          "correct": ""})
    check.to_csv(OUT / "dissent_check.csv", index=False, encoding="utf-8-sig")
    print(f"statements {len(table)}, changed pairs {(table['shift'] != 0).sum()}, with dissent {len(dis)}; "
          f"sheet {len(sheet)} pairs -> {OUT / 'label_sheet.csv'}; dissent check {len(check)} -> "
          f"{OUT / 'dissent_check.csv'}", flush=True)


def score_stage1() -> dict:
    labels = pd.read_csv(OUT / "labels.csv", encoding="utf-8-sig")
    res = stage1_measurement(labels, pd.read_csv(WORK / "label_key.csv"))
    check = pd.read_csv(OUT / "dissent_check.csv", encoding="utf-8-sig")
    res["dissent_checked"] = int(check["correct"].notna().sum())
    res["dissent_all_correct"] = bool((check["correct"].astype(str).str.upper() == "Y").all())
    res["stage1_passed"] = bool(res["passed"] and res["dissent_all_correct"] and res["dissent_checked"] == len(check))
    (OUT / "stage1.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "score", "score-human"])
    args = ap.parse_args(argv)
    {"prepare": prepare_stage1, "score": score_stage1, "score-human": score_stage1_human}[args.step]()


if __name__ == "__main__":
    main()
