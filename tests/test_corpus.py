import pandas as pd
import pytest

from fed_news import corpus

T = pd.Timestamp


def test_html_to_text_prefers_article_and_drops_scripts():
    html = ("<html><body><nav>menu</nav><div id='article'><p>The Committee decided.</p>"
            "<script>x=1</script></div><footer>foot</footer></body></html>")
    assert corpus.html_to_text(html) == "The Committee decided."


def test_split_sentences_keeps_reasonable_lengths():
    text = ("Inflation has moved up over the past year and remains elevated. Ok. "
            "The Committee decided to raise the target range for the federal funds rate.")
    assert corpus.split_sentences(text) == [
        "Inflation has moved up over the past year and remains elevated.",
        "The Committee decided to raise the target range for the federal funds rate."]


def test_beige_release_matches_url_month():
    text = "Last Update: December 3, 2018 ... Beige Book January 16, 2019 Summary"
    assert corpus.beige_release(text, T("2019-01-01")) == T("2019-01-16")
    assert pd.isna(corpus.beige_release("no dates here", T("2019-01-01")))


def test_minutes_without_release_are_dropped(monkeypatch):
    cal = pd.DataFrame([
        {"kind": "minutes", "date": T("2005-02-02"), "release": pd.NaT, "url": "u1"},
        {"kind": "minutes", "date": T("2005-03-22"), "release": T("2005-04-12"), "url": "u2"},
    ])
    monkeypatch.setattr(corpus, "get", lambda url, **k: "<div id='article'>" + "word " * 200 + "</div>")
    docs, dropped = corpus.docs_from_calendar(cal, "minutes")
    assert dropped == 1
    assert docs["release"].tolist() == [T("2005-04-12")]
    assert list(docs.columns) == corpus.DOC_COLS


def test_parse_speech_json():
    raw = ('﻿[{"d":"10/8/2026 4:30:00 AM","t":"Title","s":"Governor A &amp; B",'
           '"l":"/newsevents/speech/waller20261008a.htm"}]')
    frame = corpus.parse_speech_json(raw)
    assert frame.loc[0, "release"] == T("2026-10-08")
    assert frame.loc[0, "speaker"] == "Governor A & B"
    assert frame.loc[0, "url"].endswith("/newsevents/speech/waller20261008a.htm")


def test_parse_old_speech_list():
    page = ('<a href="/boarddocs/speeches/2002/20021121/default.htm">x</a>'
            '<a href="http://www.federalreserve.gov/boarddocs/speeches/2002/200211212/default.htm">y</a>')
    frame = corpus.parse_old_speech_list(page)
    assert frame["release"].tolist() == [T("2002-11-21"), T("2002-11-21")]
    assert all(u.startswith("https://www.federalreserve.gov/boarddocs/") for u in frame["url"])


def test_check_coverage_raises_on_missing_year():
    docs = pd.DataFrame({"type": ["statement", "minutes", "statement"],
                         "release": [T("2001-03-20"), T("2001-04-12"), T("2002-01-30")]})
    cov = corpus.coverage(docs)
    corpus.check_coverage(cov, [2001], ("statement", "minutes"))
    with pytest.raises(corpus.CoverageError, match="2002"):
        corpus.check_coverage(cov, [2001, 2002], ("statement", "minutes"))


@pytest.mark.network
def test_live_speech_json_is_parseable():
    frame = corpus.parse_speech_json(corpus.get(corpus.SPEECH_JSON, cache=False))
    assert len(frame) > 500 and frame["release"].min() < T("2008-01-01")
