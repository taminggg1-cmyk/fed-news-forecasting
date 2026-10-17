"""Proof of concept (exploratory): is ChronoGPT a better hawk/dove sentence scorer than ChronoBERT?
(docs/superpowers/specs/2026-10-12-poc-scorer-design.md)

Both 2011 vintages, frozen; mean-pooled hidden states of one layer -> standardised multinomial logistic regression.
Layer and C are chosen by stratified 5-fold log loss on labels dated before 2012; labels from 2012 on are the test.
"""
import gc
import json

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fed_news import config
from fed_news.stats_tests import mcnemar_greater

SPLIT_YEAR, VINTAGE = 2012, 2011
GRID = tuple(float(c) for c in np.logspace(-3, 2, 9))
BERT_LAYERS, GPT_LAYERS = (6, 11, 17, 22), (13, 26, 39, 52)
MAX_TOKENS, EPS = 256, 1e-6
PASS_ACC, PASS_MARGIN = 0.75, 0.05
WORK = config.DATA / "poc_scorer"
OUT = config.RESULTS / "poc_scorer"


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return ((hidden * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy()


def split(labels: pd.DataFrame, year: int = SPLIT_YEAR):
    train = labels["year"].to_numpy() < year
    return train, ~train


def _pipe(C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def _log_loss(proba: np.ndarray, y) -> float:
    return float(-np.log(np.clip(proba[np.arange(len(y)), np.asarray(y, int)], EPS, None)).mean())


def select_layer_C(feats: dict, y) -> tuple[int, float]:
    """Layer and C with the lowest mean log loss over stratified 5-fold CV of the training labels."""
    y = np.asarray(y, int)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y))
    best, best_score = None, np.inf
    for layer, X in feats.items():
        for C in GRID:
            s = np.mean([_log_loss(_pipe(C).fit(X[tr], y[tr]).predict_proba(X[va]), y[va]) for tr, va in folds])
            if s < best_score:
                best, best_score = (layer, C), s
    return best


def evaluate(proba: np.ndarray, y) -> dict:
    """Classes: dovish 0, hawkish 1, neutral 2 (labels.LABELS)."""
    y = np.asarray(y, int)
    yhat = proba.argmax(1)
    rec = recall_score(y, yhat, labels=[0, 1, 2], average=None, zero_division=0)
    hd = y < 2
    score = proba[hd, 1] - proba[hd, 0]
    sign_ok = ((y[hd] == 1) & (score > 0)) | ((y[hd] == 0) & (score < 0))
    return {"n_test": int(len(y)), "accuracy": float((yhat == y).mean()),
            "macro_f1": float(f1_score(y, yhat, labels=[0, 1, 2], average="macro", zero_division=0)),
            "recall_dovish": float(rec[0]), "recall_hawkish": float(rec[1]), "recall_neutral": float(rec[2]),
            "log_loss": _log_loss(proba, y), "sign_agreement_hd": float(sign_ok.mean())}


def passed(acc_gpt: float, acc_bert: float) -> bool:
    return bool(acc_gpt >= PASS_ACC and acc_gpt - acc_bert >= PASS_MARGIN - 1e-9)


def embed_chronobert(sentences: list[str], layers=BERT_LAYERS, batch: int = 32) -> dict:
    from transformers import AutoModel, AutoTokenizer
    name = f"manelalab/chrono-bert-v1-{VINTAGE}1231"
    tok, model = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).cuda().eval()
    out = {layer: [] for layer in layers}
    with torch.no_grad():
        for s in range(0, len(sentences), batch):
            enc = tok(sentences[s:s + batch], padding=True, truncation=True, max_length=MAX_TOKENS,
                      return_tensors="pt").to("cuda")
            hs = model(**enc, output_hidden_states=True).hidden_states
            for layer in layers:
                out[layer].append(mean_pool(hs[layer], enc["attention_mask"]))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {layer: np.concatenate(v) for layer, v in out.items()}


def trim_rotary(model, n: int) -> None:
    """The authors' rotary tables cover 65,536 positions per layer (about 1.7 GB in float32 on the GPU);
    keep the first n, which leaves every sequence of at most n tokens unchanged."""
    for block in model.blocks:
        rot = block.attn.rotary
        rot.cos, rot.sin = rot.cos[:n].contiguous(), rot.sin[:n].contiguous()


def load_chronogpt(device: str = "cuda"):
    from huggingface_hub import hf_hub_download
    from fed_news.chronogpt import ChronoGPT
    repo = f"manelalab/chrono-gpt-v1-{VINTAGE}1231"
    cfg = torch.load(hf_hub_download(repo, "config.pt"), weights_only=True)
    state = torch.load(hf_hub_download(repo, "pytorch_model.bin"), weights_only=True, map_location="cpu")
    state = {k: v.bfloat16() if v.is_floating_point() else v for k, v in state.items()}
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = ChronoGPT(**cfg)
    finally:
        torch.set_default_dtype(previous)
    model.load_state_dict(state)
    trim_rotary(model, MAX_TOKENS)
    return model.to(device).eval()


def embed_chronogpt(sentences: list[str], layers=GPT_LAYERS, batch: int = 8) -> dict:
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    model = load_chronogpt()
    pad = tok.eos_token_id
    out = {layer: [] for layer in layers}
    with torch.inference_mode():
        for s in range(0, len(sentences), batch):
            ids = [tok.encode(t)[:MAX_TOKENS] for t in sentences[s:s + batch]]
            width = max(len(i) for i in ids)
            x = torch.tensor([i + [pad] * (width - len(i)) for i in ids], device="cuda")
            mask = torch.tensor([[1] * len(i) + [0] * (width - len(i)) for i in ids], device="cuda")
            _, layer_outputs = model(x)  # causal: right padding does not change the real tokens' states
            for layer in layers:
                out[layer].append(mean_pool(layer_outputs[layer - 1], mask))
            if (s // batch) % 50 == 0:
                print(f"chronogpt embedded {s + len(ids)} of {len(sentences)}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {layer: np.concatenate(v) for layer, v in out.items()}


def _features(name: str, embed, sentences: list[str]) -> dict:
    path = WORK / f"{name}.npz"
    if path.exists():
        z = np.load(path)
        return {int(k): z[k] for k in z.files}
    feats = embed(sentences)
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{str(k): v for k, v in feats.items()})
    return feats


def run() -> dict:
    from fed_news.labels import load_labels
    labels = load_labels()
    train, test = split(labels)
    sentences, y = labels["sentence"].tolist(), labels["label"].to_numpy(int)
    res, correct = {"split_year": SPLIT_YEAR, "vintage": VINTAGE, "n_train": int(train.sum())}, {}
    for name, embed in (("chronobert", embed_chronobert), ("chronogpt", embed_chronogpt)):
        feats = _features(name, embed, sentences)
        layer, C = select_layer_C({k: v[train] for k, v in feats.items()}, y[train])
        proba = _pipe(C).fit(feats[layer][train], y[train]).predict_proba(feats[layer][test])
        res[name] = {"layer": layer, "C": C, **evaluate(proba, y[test])}
        correct[name] = proba.argmax(1) == y[test]
        print(name, json.dumps(res[name]), flush=True)
    fixes, breaks, p = mcnemar_greater(correct["chronobert"], correct["chronogpt"])
    res["mcnemar_gpt_vs_bert"] = {"gpt_right_bert_wrong": fixes, "bert_right_gpt_wrong": breaks, "p_one_sided": p}
    res["passed"] = passed(res["chronogpt"]["accuracy"], res["chronobert"]["accuracy"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    return res


if __name__ == "__main__":
    run()
