import numpy as np
import pandas as pd
import pytest

from fed_news import panel, text_features as tf

T = pd.Timestamp


def stance_docs():
    return pd.DataFrame({
        "release": [T("2020-01-05"), T("2020-01-06"), T("2020-01-07"), T("2020-01-10")],
        "type": ["statement", "minutes", "speech", "speech"],
        "stance": [0.6, 0.0, -0.3, 0.9],
    })


def test_fed_stance_is_mean_of_group_means_and_excludes_cutoff_day():
    # window [Jan 5, Jan 10): policy 0.6, minutes 0.0, speech -0.3 -> mean 0.1
    assert tf.fed_stance(stance_docs(), T("2020-01-05"), T("2020-01-10")) == pytest.approx(0.1)


def test_fed_stance_skips_empty_groups():
    assert tf.fed_stance(stance_docs(), T("2020-01-07"), T("2020-01-11")) == pytest.approx(0.3)


def test_empty_window_carries_forward():
    rows = pd.DataFrame({"win_start": [T("2020-01-05"), T("2020-01-11")],
                         "cutoff": [T("2020-01-10"), T("2020-01-20")]},
                        index=[T("2020-01-11"), T("2020-01-21")])
    idx = pd.DataFrame({"news": 0.1, "gpr": 100.0, "epu": 50.0},
                       index=pd.date_range("2019-11-01", "2020-01-31"))
    beige = pd.Series([0.2], index=[T("2019-12-01")])
    out = tf.build_text_features(rows, stance_docs(), beige, idx)
    assert list(out.columns) == panel.N_COLS
    assert out["n_stance"].tolist() == pytest.approx([0.1, 0.1])
    assert out["n_stance_chg"].iloc[1] == pytest.approx(0.0)
    assert out["n_beige"].tolist() == pytest.approx([0.2, 0.2])
    assert out["n_gpr"].iloc[0] == pytest.approx(np.log(100.0))


def test_beige_is_latest_strictly_before_cutoff():
    rows = pd.DataFrame({"win_start": [T("2020-01-01")], "cutoff": [T("2020-01-15")]}, index=[T("2020-01-16")])
    idx = pd.DataFrame({"news": 0.0, "gpr": 1.0, "epu": 1.0}, index=pd.date_range("2019-11-01", "2020-01-31"))
    beige = pd.Series([0.2, 0.9], index=[T("2019-12-01"), T("2020-01-15")])
    out = tf.build_text_features(rows, stance_docs(), beige, idx)
    assert out["n_beige"].iloc[0] == pytest.approx(0.2)


def test_lm_tone_sign():
    assert tf.lm_tone("Growth was strong and conditions improved, a positive gain.") > 0
    assert tf.lm_tone("Losses worsened and conditions were weak, with severe declines.") < 0
