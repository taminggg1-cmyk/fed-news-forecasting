"""Proof of concept (exploratory): fine-tuned ChronoGPT vs fine-tuned ChronoBERT as hawk/dove scorers
(docs/superpowers/specs/2026-10-12-poc-finetune-design.md).

ChronoGPT: the 26 encoder blocks with LoRA adapters on every linear map, base weights frozen in bfloat16, a linear
head on the mean-pooled normalised layer-26 states. ChronoBERT: stance.train_and_score, unchanged. Same split as
the frozen test (labels before 2012 train, from 2012 test); three seeds each, probabilities averaged.
"""
import gc
import json
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from fed_news import config
from fed_news.chronogpt import norm
from fed_news.poc_scorer import MAX_TOKENS, SPLIT_YEAR, VINTAGE, evaluate, load_chronogpt, passed, split
from fed_news.stats_tests import mcnemar_greater

SEEDS = (0, 1, 2)
RANK, ALPHA, DROPOUT = 16, 32, 0.05
LR, WEIGHT_DECAY, EPOCHS, BATCH, CLIP = 2e-4, 0.01, 3, 16, 1.0
WORK = config.DATA / "poc_finetune"
OUT = config.RESULTS / "poc_finetune"


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale, self.drop = alpha / rank, nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + ((self.drop(x).float() @ self.lora_A.T @ self.lora_B.T) * self.scale).to(x.dtype)


class Encoder(nn.Module):
    """The encoder half of ChronoGPT plus a classification head; states() equals the full model's layer output."""

    def __init__(self, model, checkpointing: bool = False):
        super().__init__()
        self.n = model.num_encoder_layers
        self.embed, self.value_embeds = model.embed, model.value_embeds
        self.blocks = nn.ModuleList(list(model.blocks)[: self.n])
        self.head = nn.Linear(model.embed.embedding_dim, 3, dtype=torch.float32)
        self.checkpointing = checkpointing

    def states(self, ids):
        B = ids.size(0)
        x0 = norm(self.embed(ids).bfloat16())
        x = x0
        ve = [self.value_embeds(ids[i].view(-1)) for i in range(B)]
        ve = [torch.stack([ve[b][i] for b in range(B)]) if ve[0][i] is not None else None for i in range(self.n)]
        for block, v in zip(self.blocks, ve):
            if self.checkpointing and self.training:
                x = checkpoint(block, x, v, x0, use_reentrant=False)
            else:
                x = block(x, v, x0)
        return norm(x)

    def forward(self, ids, mask):
        h = self.states(ids).float()
        m = mask.unsqueeze(-1).float()
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1))


def add_lora(enc: Encoder, rank: int = RANK, alpha: float = ALPHA, dropout: float = DROPOUT) -> None:
    for p in enc.parameters():
        p.requires_grad_(False)
    for block in enc.blocks:
        for parent, names in ((block.attn, ("c_q", "c_k", "c_v", "c_proj")), (block.mlp, ("c_fc", "c_proj"))):
            for name in names:
                setattr(parent, name, LoRALinear(getattr(parent, name), rank, alpha, dropout))
    for p in enc.head.parameters():
        p.requires_grad_(True)


def _pad(token_lists, pad: int, device: str):
    width = max(len(t) for t in token_lists)
    ids = torch.tensor([t + [pad] * (width - len(t)) for t in token_lists], device=device)
    mask = torch.tensor([[1] * len(t) + [0] * (width - len(t)) for t in token_lists], device=device)
    return ids, mask


def train_and_score_gpt(texts, y, test_texts, seed: int, build, tokenize, device: str = "cuda",
                        epochs: int = EPOCHS, batch: int = BATCH, pad: int = 0) -> np.ndarray:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc = build()
    add_lora(enc)
    enc.to(device)
    params = [p for p in enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    y_t = torch.tensor(np.asarray(y), dtype=torch.long, device=device)
    toks = tokenize(texts)
    for epoch in range(epochs):
        enc.train()
        order = rng.permutation(len(texts))
        for s in range(0, len(order), batch):
            j = order[s:s + batch]
            ids, mask = _pad([toks[k] for k in j], pad, device)
            loss = nn.functional.cross_entropy(enc(ids, mask), y_t[j])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, CLIP)
            opt.step()
        print(f"chronogpt seed {seed} epoch {epoch + 1} last loss {loss.item():.3f}", flush=True)
    enc.eval()
    test_toks, out = tokenize(test_texts), []
    with torch.inference_mode():
        for s in range(0, len(test_toks), 32):
            ids, mask = _pad(test_toks[s:s + 32], pad, device)
            out.append(torch.softmax(enc(ids, mask), dim=-1).cpu().numpy())
    del enc, opt
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(out)


def _gpt_parts():
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    tokenize = lambda batch: [tok.encode(t)[:MAX_TOKENS] for t in batch]

    def build():
        model = load_chronogpt(device="cpu")
        return Encoder(model, checkpointing=True)
    return build, tokenize, tok.eos_token_id


def _cached(path, make) -> np.ndarray:
    if path.exists():
        return np.load(path)
    p = make()
    WORK.mkdir(parents=True, exist_ok=True)
    np.save(path, p)
    return p


def run() -> dict:
    from fed_news.labels import load_labels
    from fed_news.stance import train_and_score
    labels = load_labels()
    train, test = split(labels)
    tr, te = labels[train].reset_index(drop=True), labels[test]
    y_test = te["label"].to_numpy(int)
    build, tokenize, pad = _gpt_parts()
    res, probs = {"split_year": SPLIT_YEAR, "vintage": VINTAGE, "n_train": int(train.sum())}, {}
    runners = {"chronobert": lambda seed: train_and_score(tr, te["sentence"].tolist(), VINTAGE, seed),
               "chronogpt": lambda seed: train_and_score_gpt(tr["sentence"].tolist(), tr["label"].to_numpy(int),
                                                             te["sentence"].tolist(), seed, build, tokenize, pad=pad)}
    for name, runner in runners.items():
        seeds = [_cached(WORK / f"{name}_seed{s}.npy", lambda s=s: runner(s)) for s in SEEDS]
        probs[name] = np.mean(seeds, axis=0)
        res[name] = {**evaluate(probs[name], y_test),
                     "seed_accuracy": [float((p.argmax(1) == y_test).mean()) for p in seeds]}
        print(name, json.dumps(res[name]), flush=True)
    ok = {k: v.argmax(1) == y_test for k, v in probs.items()}
    fixes, breaks, p = mcnemar_greater(ok["chronobert"], ok["chronogpt"])
    res["mcnemar_gpt_vs_bert"] = {"gpt_right_bert_wrong": fixes, "bert_right_gpt_wrong": breaks, "p_one_sided": p}
    res["passed"] = passed(res["chronogpt"]["accuracy"], res["chronobert"]["accuracy"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    return res


if __name__ == "__main__":
    run()
