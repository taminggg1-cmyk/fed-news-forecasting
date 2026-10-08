from fed_news import run_models, walkforward as wf


def test_no_text_selection_excludes_every_news_spec():
    names = [s.name for s in run_models.select_specs("T2", no_text=True)]
    assert names == ["rw", "taylor", "S", "S+M"]


def test_full_selection_is_the_whole_ladder():
    assert [s.name for s in run_models.select_specs("T1", no_text=False)] == [s.name for s in wf.ladder("T1")]
