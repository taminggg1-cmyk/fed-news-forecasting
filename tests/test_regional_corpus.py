import pandas as pd

from fed_news import regional_corpus as rc


def test_in_tenure_and_surname():
    assert rc.in_tenure("Yellen", 2006) and not rc.in_tenure("Yellen", 2012)
    assert rc.in_tenure("Williams", 2015, "San Francisco") and rc.in_tenure("Williams", 2020, "New York")
    assert not rc.in_tenure("Williams", 2015, "New York")
    assert rc.surname("Governor Lael Brainard") == "Brainard" and rc.surname("Robert D. McTeer Jr.") == "McTeer"
    assert rc.surname("") == ""


def test_fraser_selection_and_docs(tmp_path):
    items = pd.DataFrame({"item_id": ["1", "2", "3", "4"], "title_id": ["930", "930", "7161", "9999"],
                          "date": pd.to_datetime(["2006-05-01", "2012-05-01", "2015-01-16", "2015-01-16"]),
                          "title": ["a", "b", "c", "d"], "text_urls": [["u"]] * 4})
    sel = rc.fraser_selection(items)
    assert list(sel["item_id"]) == ["1", "3"]                       # Yellen 2012 and unknown title dropped
    (tmp_path / "1.txt").write_text("Yellen text", encoding="utf-8")
    (tmp_path / "3.txt").write_text("", encoding="utf-8")            # empty text dropped
    docs = rc.fraser_docs(sel, text_dir=tmp_path)
    assert list(docs["speaker"]) == ["Yellen"] and docs["source"].iloc[0] == "fraser" and docs["type"].iloc[0] == "speech"


def test_bis_docs_president_bank_and_tenure(tmp_path):
    rows = pd.DataFrame({
        "date": ["2019-01-18", "2019-02-22", "2019-03-01", "2012-03-01", "2019-04-01"],
        "description": ["Remarks by Mr John C Williams, President and Chief Executive Officer of the Federal Reserve Bank of New York, at ...",
                        "Remarks by Mr Simon M Potter, Executive Vice President of the Federal Reserve Bank of New York, at ...",
                        "Remarks by Mr Richard Fisher, President of the Federal Reserve Bank of Boston, at ...",
                        "Remarks by Ms Mary C Daly, President of the Federal Reserve Bank of San Francisco, at ...",
                        "Remarks by Ms Lael Brainard, Member of the Board of Governors of the Federal Reserve System, at ..."],
        "author": ["John C Williams", "Simon M Potter", "Richard Fisher", "Mary C Daly", "Lael Brainard"],
        "text": ["t1", "t2", "t3", "t4", "t5"]})
    rows.to_csv(tmp_path / "speeches_2019.csv", index=False)
    out = rc.bis_docs(tmp_path)
    assert list(out["speaker"]) == ["Williams"]                      # VP, wrong bank, out of tenure, Board: all excluded
    assert out["source"].iloc[0] == "bis" and out["release"].iloc[0] == pd.Timestamp("2019-01-18")


def test_build_regional_dedup_and_corpus_overlap():
    def doc(i, sp, day, src):
        return {"doc_id": i, "type": "speech", "speaker": sp, "release": pd.Timestamp(day), "text": i, "source": src}
    fr = pd.DataFrame([doc("f1", "Bullard", "2015-01-16", "fraser"), doc("f2", "Yellen", "2008-03-03", "fraser")])
    bis = pd.DataFrame([doc("b1", "Bullard", "2015-01-16", "bis"), doc("b2", "Daly", "2019-05-05", "bis")])
    corpus = pd.DataFrame({"doc_id": ["c1"], "type": ["speech"], "speaker": ["Vice Chair Janet L. Yellen"],
                           "release": [pd.Timestamp("2008-03-03")], "text": ["x"]})
    out = rc.build_regional(fr, bis, corpus)
    assert list(out["doc_id"]) == ["f1", "b2"]
