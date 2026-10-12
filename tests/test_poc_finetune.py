import numpy as np
import pytest
import torch

from fed_news import poc_finetune as pf
from fed_news.chronogpt import ChronoGPT


def tiny(seed=0):
    torch.manual_seed(seed)
    model = ChronoGPT(vocab_size=40, num_layers=4, num_heads=2, model_dim=16)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.2)
    return model


def test_encoder_states_equal_the_full_models_last_encoder_layer():
    model = tiny()
    ids = torch.randint(0, 40, (3, 6))
    with torch.no_grad():
        full = model(ids)[1][1]  # 4 blocks: encoder = first 2, layer outputs index 1
        enc = pf.Encoder(model).eval()
        assert torch.equal(enc.states(ids), full)


def test_lora_starts_as_identity_and_only_adapters_and_head_train():
    model = tiny()
    ids = torch.randint(0, 40, (2, 5))
    with torch.no_grad():
        before = pf.Encoder(tiny()).eval().states(ids)
    enc = pf.Encoder(model)
    pf.add_lora(enc, rank=2, alpha=4, dropout=0.0)
    enc.eval()
    with torch.no_grad():
        assert torch.equal(enc.states(ids), before)
    trainable = {n for n, p in enc.named_parameters() if p.requires_grad}
    assert trainable and all(".lora_" in n or n.startswith("head.") for n in trainable)
    enc.train()
    loss = enc(ids, torch.ones_like(ids)).sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in enc.named_parameters() if "lora_B" in n)


def test_train_and_score_returns_probabilities_for_every_test_text():
    texts = ["hawk " * 3, "dove " * 2, "neutral line", "hawk dove"] * 4
    y = np.array([1, 0, 2, 2] * 4)
    tokenize = lambda batch: [[ord(c) % 40 for c in t][:20] for t in batch]
    p = pf.train_and_score_gpt(texts, y, texts[:5], seed=0, build=lambda: pf.Encoder(tiny()), tokenize=tokenize,
                               device="cpu", epochs=1, batch=4)
    assert p.shape == (5, 3)
    assert np.allclose(p.sum(1), 1.0, atol=1e-5)
