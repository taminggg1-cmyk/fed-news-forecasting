from fed_news.studies.plan5_event_study import ablation as ab
from fed_news.features.panel import N_COLS, S_COLS


def test_specs_cover_each_group_alone_and_left_out():
    specs = {s.name: s for s in ab.specs("T1")}
    assert set(specs) == {"S+stance", "S+beige", "S+newspapers", "S+N-stance", "S+N-beige", "S+N-newspapers"}
    assert specs["S+stance"].cols == S_COLS + ["n_stance", "n_stance_chg"]
    assert set(specs["S+N-beige"].cols) == set(S_COLS + [c for c in N_COLS if c != "n_beige"])
    assert all(s.kind == "primary" and s.task == "T1" for s in specs.values())
