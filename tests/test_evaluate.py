import pandas as pd
import pytest

from fed_news import evaluate


def test_metrics_on_toy_predictions():
    preds = pd.DataFrame({"y_true": [1.0, 2.0], "y_pred": [1.5, 2.0],
                          "class_true": [2, 1], "class_pred": [2, 0]})
    m = evaluate.metrics(preds)
    assert m["mse"] == pytest.approx(0.125) and m["mae"] == pytest.approx(0.25)
    assert m["mbe"] == pytest.approx(0.25) and m["accuracy"] == pytest.approx(0.5)


def test_decide_requires_all_three():
    assert evaluate.decide([0.01, 0.02, 0.03], [0.5, 0.6, 0.7]) == "supported"
    assert evaluate.decide([0.01, 0.20, 0.03], [0.5, 0.6, 0.7]) == "not supported"


def test_decide_void_when_placebo_passes_any():
    assert evaluate.decide([0.01, 0.02, 0.03], [0.5, 0.04, 0.7]) == "void"
    assert evaluate.decide([0.30, 0.02, 0.03], [0.5, 0.04, 0.7]) == "void"
