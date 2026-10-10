import numpy as np
import pandas as pd

from fed_news import event_study as es


def synthetic(n=400, slope=5.0, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("1998-01-01", periods=n)
    s = rng.normal(size=n)
    return pd.DataFrame({"surprise": s, "dy2": slope * s + rng.normal(size=n),
                         "dy2_before": rng.normal(size=n), "category": "speech",
                         "stance": s, "n_docs": 1}, index=pd.Index(days, name="day"))


def test_walk_forward_uses_only_earlier_days():
    e = synthetic()
    k = 250
    base = es.walk_forward(e, "dy2", test_start=str(e.index[200].date()))
    pert = e.copy()
    pert.iloc[k:, pert.columns.get_loc("dy2")] = 100.0
    again = es.walk_forward(pert, "dy2", test_start=str(e.index[200].date()))
    cut = e.index[k]
    cols = ["pred_small", "pred_large"]
    assert np.allclose(base.loc[:cut, cols], again.loc[:cut, cols])


def test_decide_rule():
    assert es.decide(0.01, 0.20) == "supported"
    assert es.decide(0.20, 0.20) == "not supported"
    assert es.decide(0.01, 0.01) == "void"


def test_run_detects_signal_and_void_is_null():
    summary, wf = es.run(synthetic(), test_start=str(synthetic().index[200].date()))
    assert summary["verdict"] == "supported"
    assert summary["cw_p"] < 0.01 and summary["void_cw_p"] > 0.05
    assert summary["n_test"] == len(wf) == 200


def test_walk_forward_refuses_tiny_training_set():
    e = synthetic(n=60)
    try:
        es.walk_forward(e, "dy2", test_start=str(e.index[10].date()))
    except ValueError as err:
        assert "training events" in str(err)
    else:
        raise AssertionError("must refuse fewer than MIN_TRAIN training events")
