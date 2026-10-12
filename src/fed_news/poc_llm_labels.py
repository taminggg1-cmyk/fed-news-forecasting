"""Proof of concept (exploratory): does Claude Haiku 5.5 label Fed sentences like the human annotators?
(docs/superpowers/specs/2026-10-12-poc-llm-labels-design.md)

300 human-labelled sentences dated before 2012, text only, 25 per request, zero-shot. Pass: three-class
agreement with the human labels of at least 0.70.
"""
import json
import os
import re

import numpy as np

from fed_news import config

MODEL = "claude-haiku-5-5"
N_PILOT, SEED, BATCH, SPLIT_YEAR = 300, 0, 25, 2012
PASS_AGREEMENT = 0.70
CODES = {"D": 0, "H": 1, "N": 2}  # labels.LABELS: dovish 0, hawkish 1, neutral 2
OUT = config.RESULTS / "poc_llm"

SYSTEM = (
    "You classify sentences from US Federal Reserve communications by their monetary policy stance.\n"
    "H (hawkish): the sentence points toward tighter monetary policy - higher or rising interest rates, less "
    "accommodation, concern that inflation is too high or rising, or an economy or labour market that is strong "
    "or overheating.\n"
    "D (dovish): the sentence points toward looser monetary policy - lower interest rates or more accommodation, "
    "inflation that is low or falling, or an economy or labour market that is weak, slowing or under stress.\n"
    "N (neutral): neither - descriptive or procedural, balanced, or not about the policy outlook.\n"
    "Judge only the text of each sentence. Reply with one JSON object mapping each sentence number to H, D or N, "
    "and nothing else."
)


def batch_prompt(sentences: list[str]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))


def parse_labels(text: str, n: int) -> list[int]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    obj = {str(k).strip(): str(v).strip().upper() for k, v in json.loads(m.group(0)).items()}
    out = []
    for i in range(1, n + 1):
        v = obj.get(str(i))
        if v not in CODES:
            raise ValueError(f"sentence {i}: label {v!r}")
        out.append(CODES[v])
    return out


def _client():
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    path = config.SECRETS / "anthropic_api_key.txt"
    if not key and path.exists():
        key = path.read_text().strip()
    return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()


ATTEMPTS = {  # design doc: attempt 1 (no thinking, effort low); attempt 2 (adaptive thinking, effort medium)
    1: {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}, "max_tokens": 1024},
    2: {"output_config": {"effort": "medium"}, "max_tokens": 8000},
}
TOKEN_CAP = 95_000


def pilot_sample(labels, attempt: int):
    """Attempt 1: 300 pre-2012 labels (seed 0). Attempt 2: 300 others (seed 1), disjoint from attempt 1."""
    pool = labels[labels["year"] < SPLIT_YEAR]
    first = pool.sample(N_PILOT, random_state=SEED)
    pick = first if attempt == 1 else pool.drop(first.index).sample(N_PILOT, random_state=attempt - 1)
    return pick.reset_index(drop=True)


def _ask(client, sentences: list[str], usage: dict, settings: dict) -> list[int] | None:
    for _ in range(2):  # one retry on a malformed, cut-off or refused answer
        if usage["input"] + usage["output"] >= TOKEN_CAP:
            return None
        r = client.messages.create(model=MODEL, system=SYSTEM, **settings,
                                   messages=[{"role": "user", "content": batch_prompt(sentences)}])
        usage["input"] += r.usage.input_tokens
        usage["output"] += r.usage.output_tokens
        if r.stop_reason in ("refusal", "max_tokens"):
            continue
        try:
            return parse_labels("".join(b.text for b in r.content if b.type == "text"), len(sentences))
        except (ValueError, json.JSONDecodeError):
            continue
    return None


def run_pilot(attempt: int = 1) -> dict:
    import pandas as pd
    from sklearn.metrics import confusion_matrix, recall_score
    from fed_news.labels import load_labels
    labels = load_labels()
    pilot = pilot_sample(labels, attempt)
    client, usage, llm = _client(), {"input": 0, "output": 0}, []
    for s in range(0, len(pilot), BATCH):
        sents = pilot["sentence"].iloc[s:s + BATCH].tolist()
        got = _ask(client, sents, usage, ATTEMPTS[attempt])
        llm += got if got is not None else [-1] * len(sents)
        print(f"labelled {s + len(sents)} of {len(pilot)}; tokens so far {usage}", flush=True)
    human, llm = pilot["label"].to_numpy(int), np.array(llm)
    hd = (human < 2) & (llm >= 0) & (llm < 2)
    res = {"model": MODEL, "attempt": attempt, "settings": ATTEMPTS[attempt], "n": int(len(human)), "n_unlabelled": int((llm < 0).sum()),
           "agreement": float((llm == human).mean()),
           "recall_dovish_hawkish_neutral": recall_score(human, llm, labels=[0, 1, 2], average=None,
                                                         zero_division=0).round(3).tolist(),
           "confusion_rows_human_cols_llm_DHN": confusion_matrix(human, llm, labels=[0, 1, 2]).tolist(),
           "hawk_dove_sign_agreement": float((llm[hd] == human[hd]).mean()) if hd.any() else None,
           "tokens": usage}
    res["passed"] = bool(res["agreement"] >= PASS_AGREEMENT)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "" if attempt == 1 else str(attempt)
    pilot.assign(llm=llm)[["sentence", "label", "llm", "year", "source"]].to_csv(OUT / f"pilot{tag}_labels.csv",
                                                                                  index=False)
    (OUT / f"pilot{tag}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    return res


if __name__ == "__main__":
    import sys
    run_pilot(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
