import pytest

from fed_news import poc_llm_labels as pl


def test_prompt_numbers_sentences_and_sends_text_only():
    prompt = pl.batch_prompt(["inflation is elevated.", "growth has slowed."])
    assert "1. inflation is elevated." in prompt and "2. growth has slowed." in prompt


def test_parse_labels_maps_to_dataset_codes():
    # labels.LABELS: dovish 0, hawkish 1, neutral 2
    assert pl.parse_labels('{"1": "H", "2": "D", "3": "N"}', 3) == [1, 0, 2]


def test_parse_labels_accepts_text_around_the_json():
    assert pl.parse_labels('Here you go:\n```json\n{"1": "d", "2": "H"}\n```', 2) == [0, 1]


def test_parse_labels_rejects_missing_or_unknown_labels():
    with pytest.raises(ValueError):
        pl.parse_labels('{"1": "H"}', 2)
    with pytest.raises(ValueError):
        pl.parse_labels('{"1": "X"}', 1)
