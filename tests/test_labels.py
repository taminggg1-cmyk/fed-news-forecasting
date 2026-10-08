import pytest

from fed_news import labels


@pytest.mark.network
def test_labels_load_and_tdw_mapping_is_hawkish_for_tightening():
    lab = labels.load_labels(refresh=True)
    assert set(lab["label"]) == {0, 1, 2}
    assert lab["sentence"].is_unique
    assert (lab["sentence"] == lab["sentence"].str.lower()).all()
    tdw = lab[lab["source"] == "tdw"]
    assert len(tdw) > 2000
    tight = tdw[tdw["sentence"].str.contains(r"\btighten", regex=True)]
    assert tight["label"].mode().iloc[0] == labels.LABELS["hawkish"]
    assert lab["year"].min() <= 1997
