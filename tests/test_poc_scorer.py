import numpy as np
import pandas as pd
import pytest
import torch

from fed_news import poc_scorer as pc


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
