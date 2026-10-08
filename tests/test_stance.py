import pandas as pd
import pytest

from fed_news import stance


def test_vintage_rule():
    assert stance.vintage_year(1994) == 1999
    assert stance.vintage_year(2000) == 1999
    assert stance.vintage_year(2016) == 2015
    assert stance.vintage_year(2026) == 2024


def test_classifier_year_maps_early_rows_to_2000():
    assert stance.classifier_year(1996) == 2000
    assert stance.classifier_year(2013) == 2013


def test_model_name():
    assert stance.model_name(2015) == "manelalab/chrono-bert-v1-20151231"


def test_eligible_is_strictly_before_year():
    lab = pd.DataFrame({"sentence": [f"s{i}" for i in range(120)], "label": 0,
                        "year": [1998] * 60 + [2000] * 60})
    assert set(stance.eligible(lab, 2000)["year"]) == {1998}


def test_eligible_raises_when_too_few():
    lab = pd.DataFrame({"sentence": ["a", "b"], "label": [0, 1], "year": [1998, 1999]})
    with pytest.raises(ValueError, match="labels"):
        stance.eligible(lab, 2000)


@pytest.mark.gpu
def test_train_and_score_smoke():
    train = pd.DataFrame({"sentence": ["the committee decided to raise rates"] * 16
                          + ["the committee decided to lower rates"] * 16,
                          "label": [1] * 16 + [0] * 16})
    probs = stance.train_and_score(train, ["inflation pressures warrant tightening"], 1999, 0)
    assert probs.shape == (1, 3)
    assert abs(probs.sum() - 1.0) < 1e-4
