import numpy as np
import pandas as pd

from fed_news.studies.study2_regional import study2_frozen as sf


def synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("1998-01-01", periods=n)
    s = rng.normal(size=n)
    return pd.DataFrame({"surprise": s, "dy2": 4.0 * s + rng.normal(size=n), "dy2_before": rng.normal(size=n),
                         "category": "speech"}, index=pd.Index(days, name="day"))


def test_frozen_event_fits_once_on_pre_cutoff_days():
    ev = synthetic()
    cut = str(ev.index[400].date())
    row = sf.frozen_event(ev, cut, test_end=str(ev.index[-1].date()))
    assert row["n_train"] == 400 and row["n_test"] == 200
    assert row["frozen_cw_p"] < 0.01 and row["walkforward_cw_p"] < 0.01
    pert = ev.copy()
    pert.iloc[400:, pert.columns.get_loc("dy2")] = 50.0          # later outcomes cannot change the frozen fit
    again = sf.frozen_event(pert, cut, test_end=str(ev.index[-1].date()))
    assert np.isclose(row["frozen_slope"], again["frozen_slope"])


def test_frozen_event_needs_training_days():
    ev = synthetic(n=60)
    try:
        sf.frozen_event(ev, str(ev.index[10].date()))
    except ValueError as err:
        assert "training events" in str(err)
    else:
        raise AssertionError("must refuse fewer than MIN_TRAIN training events")
