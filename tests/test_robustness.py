import pandas as pd

from fed_news import robustness as rb


def test_sign_test_matches_binomial_tail():
    assert abs(rb.sign_test(6, 9) - 130 / 512) < 1e-12
    assert rb.sign_test(0, 0) == 1.0


def test_above_zlb_keeps_rows_strictly_above_bound():
    p = pd.DataFrame({"lag": [0.05, 0.25, 0.26, 5.0], "y_true": 0.0, "y_pred": 0.0})
    assert rb.above_zlb(p)["lag"].tolist() == [0.26, 5.0]


def test_text_pairs_cover_eleven_model_task_pairs():
    pairs = rb.text_pairs("T1") + rb.text_pairs("T2")
    assert len(pairs) == 11
    assert ("linear+market", "S+M", "S+M+N") in pairs
    assert all(text == base.replace("-S", "-S+N") or text == base + "+N" for _, base, text in pairs)
