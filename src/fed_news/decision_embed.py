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
        with np.load(path, allow_pickle=False) as z:  # closed before a recompute replaces the file (Windows)
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
