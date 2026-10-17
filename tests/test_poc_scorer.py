import numpy as np
import pandas as pd
import pytest
import torch

from fed_news.studies.poc import poc_scorer as pc


def test_mean_pool_ignores_padding():
    h = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert pc.mean_pool(h, mask).tolist() == [[2.0, 3.0]]


def test_split_is_by_label_year():
    labels = pd.DataFrame({"year": [2010, 2011, 2012, 2020]})
    train, test = pc.split(labels, 2012)
    assert train.tolist() == [True, True, False, False]
    assert test.tolist() == [False, False, True, True]


def test_selection_picks_the_informative_layer_using_training_rows_only():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    informative = np.eye(3)[y] * 2 + rng.normal(size=(300, 3))
    feats = {6: rng.normal(size=(300, 3)), 11: informative}
    layer, C = pc.select_layer_C(feats, y)
    assert layer == 11 and C in pc.GRID


def test_evaluate_reports_accuracy_and_hawk_dove_sign_agreement():
    # classes: dovish 0, hawkish 1, neutral 2
    proba = np.array([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7], [0.5, 0.4, 0.1]])
    y = np.array([0, 1, 2, 1])
    res = pc.evaluate(proba, y)
    assert res["accuracy"] == pytest.approx(0.75)
    assert res["sign_agreement_hd"] == pytest.approx(2 / 3)
    assert res["n_test"] == 4


def test_pass_rule():
    assert pc.passed(0.76, 0.70) is True
    assert pc.passed(0.76, 0.72) is False
    assert pc.passed(0.74, 0.60) is False


def test_trim_rotary_keeps_outputs_for_short_inputs_and_shrinks_tables():
    from fed_news.studies.poc.chronogpt import ChronoGPT
    torch.manual_seed(0)
    model = ChronoGPT(vocab_size=50, num_layers=4, num_heads=2, model_dim=16)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.2)
    x = torch.randint(0, 50, (2, 7))
    before = [h.float() for h in model(x)[1]]
    pc.trim_rotary(model, 10)
    assert all(b.attn.rotary.cos.shape[0] == 10 and b.attn.rotary.sin.shape[0] == 10 for b in model.blocks)
    after = [h.float() for h in model(x)[1]]
    assert all(torch.equal(a, b) for a, b in zip(before, after))
