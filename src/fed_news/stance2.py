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
