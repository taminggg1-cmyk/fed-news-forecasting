import pandas as pd
import pytest

from fed_news import stmt_change as sc

S2008 = ("Voting for the FOMC monetary policy action were: Ben S. Bernanke, Chairman; Timothy F. Geithner, Vice "
         "Chairman; Elizabeth A. Duke; Richard W. Fisher; Donald L. Kohn; and Kevin M. Warsh. In a related action, "
         "the Board of Governors unanimously approved a 50-basis-point decrease in the discount rate.")
S2011 = ("Voting for the FOMC monetary policy action were: Ben S. Bernanke, Chairman; Janet L. Yellen. Voting "
         "against the action were: Richard W. Fisher, Narayana Kocherlakota, and Charles I. Plosser, who would "
         "have preferred to continue to describe economic conditions as likely to warrant exceptionally low "
         "levels for the federal funds rate for an extended period.")
S2019 = ("Voting for the monetary policy action were Jerome H. Powell, Chair, John C. Williams, Vice Chair; "
         "Randal K. Quarles. Voting against the action were James Bullard, who preferred at this meeting to lower "
         "the target range for the federal funds rate to 1-1/2 to 1-3/4 percent; and Esther L. George and Eric S. "
         "Rosengren, who preferred to maintain the target range at 2 percent to 2-1/4 percent. Implementation "
         "Note issued September 18, 2019")
S2025 = ("Voting for the monetary policy action were Jerome H. Powell, Chair. Voting against this action were "
         "Stephen I. Miran, who preferred to lower the target range for the federal funds rate by 1/2 percentage "
         "point at this meeting; and Austan D. Goolsbee and Jeffrey R. Schmid, who preferred no change to the "
         "target range for the federal funds rate at this meeting. For media inquiries, please email.")


def test_normalise_ignores_month_digits_and_release_header():
    a = sc.normalise("edt share information received since the committee met in july indicates 2 percent.")
    b = sc.normalise("information received since the committee met in june indicates 3-1/2 percent.")
    assert a == b


def test_keep_sentences_truncates_at_vote_and_drops_action_and_discount_rate():
    sents = ["release date: august 12, 2003 for immediate release the federal open market committee decided "
             "today to keep its target for the federal funds rate at 1 percent.",
             "business pricing power and increases in core consumer prices remain muted.",
             "in a related action, the board of governors approved a reduction in the discount rate to 2 percent.",
             "policy accommodation can be maintained for a considerable period.",
             "voting for the fomc monetary policy action were: alan greenspan, chairman; ben s.",
             "alfred broaddus, jr.; roger w."]
    assert sc.keep_sentences(sents) == [sents[1], sents[3]]


def test_shift_is_added_minus_removed_and_ignores_unchanged_sentences():
    prev = [("inflation risks are tilted to the upside.", 0.5), ("labor markets remain solid.", 0.1)]
    cur = [("labor markets remain solid.", 0.3), ("inflation pressures have eased.", -0.4)]
    assert sc.statement_shift(prev, cur) == pytest.approx(-0.4 - 0.5)


def test_shift_ignores_a_changed_action_sentence():
    prev = [("the committee decided today to lower its target for the federal funds rate to 1 percent.", -0.3),
            ("growth is moderate.", 0.0)]
    cur = [("the committee decided today to raise its target for the federal funds rate to 1-1/4 percent.", 0.6),
           ("growth is moderate.", 0.0)]
    assert sc.statement_shift(prev, cur) == 0.0


@pytest.mark.parametrize("text, action, expected", [
    (S2008, -0.5, (0, 0)),
    (S2011, 0.0, (3, 0)),
    (S2019, -0.25, (3, 1)),
    (S2025, -0.25, (3, 1)),
    ("Voting against the action was Jane Doe, who preferred no change in the target.", 0.25, (1, -1)),
    ("Voting against the action were Richard W. Fisher and Charles I. Plosser, who did not support additional "
     "policy accommodation at this time.", 0.0, (2, 2)),
    ("Voting against the action were Thomas M. Hoenig, who preferred to raise the target.", 0.0, (1, 1)),
])
def test_parse_dissent_counts_members_and_direction_relative_to_action(text, action, expected):
    assert sc.parse_dissent(text, action) == expected


def test_meeting_features_sum_window_and_take_latest_dissent():
    stm = pd.DataFrame({"shift": [1.0, -0.25, 2.0], "dissent_n": [1, 2, 0], "dissent_dir": [1, -2, 0]},
                       index=pd.to_datetime(["2005-01-10", "2005-01-20", "2005-02-01"]))
    rows = pd.DataFrame({"cutoff": pd.to_datetime(["2005-01-05", "2005-01-25", "2005-03-01"])},
                        index=pd.to_datetime(["2005-01-12", "2005-02-01", "2005-03-08"]))
    out = sc.meeting_features(stm, rows)
    assert out["x_shift"].tolist() == [0.0, 0.75, 2.0]
    assert out["x_dissent_n"].tolist() == [0, 2, 0]
    assert out["x_dissent_dir"].tolist() == [0, -2, 0]


def test_statement_on_the_cutoff_day_is_not_used():
    stm = pd.DataFrame({"shift": [5.0], "dissent_n": [1], "dissent_dir": [1]}, index=pd.to_datetime(["2005-01-25"]))
    rows = pd.DataFrame({"cutoff": pd.to_datetime(["2005-01-25"])}, index=pd.to_datetime(["2005-02-01"]))
    assert sc.meeting_features(stm, rows)["x_shift"].tolist() == [0.0]


def test_stage1_agreement_counts_zero_shift_as_wrong_and_skips_n_labels():
    key = pd.DataFrame({"pair_id": [1, 2, 3, 4, 5], "x_shift": [0.4, -0.2, 0.0, 0.3, -0.1]})
    labels = pd.DataFrame({"pair_id": [1, 2, 3, 4, 5], "label": ["H", "D", "H", "N", "H"]})
    res = sc.stage1_measurement(labels, key)
    assert res["n_directional"] == 4
    assert res["agreement"] == pytest.approx(0.5)
    assert res["majority_share"] == pytest.approx(0.75)
    assert res["passed"] is False


def _toy_inputs():
    corpus = pd.DataFrame({
        "doc_id": ["statement:a", "statement:liq", "statement:b"],
        "type": ["statement"] * 3, "speaker": [""] * 3,
        "release": pd.to_datetime(["2005-01-10", "2005-01-15", "2005-02-01"]),
        "text": ["x", "x", "Voting against the action was Jane Doe, who preferred to raise the target."]})
    scores = pd.DataFrame({
        "doc_id": ["statement:a", "statement:a", "statement:liq", "statement:b", "statement:b"],
        "sentence": ["growth is moderate.", "inflation risks are elevated.", "a new lending facility opens.",
                     "growth is moderate.", "inflation pressures have eased."],
        "stance": [0.0, 0.6, 0.9, 0.0, -0.3]})
    meetings = pd.DataFrame({"y_change": [0.25, 0.0]}, index=pd.to_datetime(["2005-01-10", "2005-02-01"]))
    return corpus, scores, meetings


def test_statement_table_compares_decision_day_statements_only():
    table = sc.statement_table(*_toy_inputs())
    assert list(table.index) == list(pd.to_datetime(["2005-01-10", "2005-02-01"]))
    assert table["shift"].tolist() == pytest.approx([0.0, -0.3 - 0.6])
    assert table.loc["2005-02-01", "added"] == ["inflation pressures have eased."]
    assert table.loc["2005-02-01", "removed"] == ["inflation risks are elevated."]
    assert (table["dissent_n"].tolist(), table["dissent_dir"].tolist()) == ([0, 1], [0, 1])


def test_label_sheet_hides_dates_and_scores_and_samples_changed_pairs_only():
    table = sc.statement_table(*_toy_inputs())
    sheet, key = sc.label_sheet(table, n=5, seed=0)
    assert list(sheet.columns) == ["pair_id", "removed", "added", "label"]
    assert len(sheet) == 1 and sheet["label"].tolist() == [""]
    assert "inflation pressures have eased." in sheet["added"].iloc[0]
    assert key["x_shift"].tolist() == pytest.approx([-0.9])


@pytest.mark.parametrize("text, action, expected", [
    ("Voting against the action were: Edward M. Gramlich and Robert D. McTeer, Jr. Governor Gramlich and President "
     "McTeer preferred a reduction in the target for the federal funds rate. 2002 Monetary policy Home", 0.0, (2, -2)),
    ("Voting against the action was Robert T. Parry. President Parry preferred a 50 basis point reduction in the "
     "target for the federal funds rate. In taking the discount rate action, the Federal Reserve Board approved the "
     "requests submitted by the Boards of Directors of the Federal Reserve Banks of Boston, New York.", -0.25, (1, -1)),
    ("Voting against the action were Richard W. Fisher and Charles I. Plosser. President Fisher believed that "
     "conditions will likely warrant an earlier reduction in monetary accommodation than is suggested.", 0.0, (2, 2)),
    ("Voting against this action was Esther L. George, who preferred at this meeting to raise the target range for "
     "the federal funds rate by 0.5 percentage point to 1-1/4 percent to 1-1/2 percent.", 0.75, (1, -1)),
    ("Voting against this action was Michelle W. Bowman, who preferred to lower the target range for the federal "
     "funds rate by 1/4 percentage point at this meeting.", -0.5, (1, 1)),
    ("Voting against the policy was Thomas M. Hoenig, who preferred to raise the target.", 0.0, (1, 1)),
])
def test_parse_dissent_older_formats_and_move_size(text, action, expected):
    assert sc.parse_dissent(text, action) == expected


def test_stage1_human_labels_uses_only_labels_unseen_by_the_scoring_classifier():
    table = pd.DataFrame({"doc_id": ["a", "b", "c"],
                          "added": [[], ["inflation pressures have eased."], ["growth is strong.", "a conflicted line."]],
                          "removed": [[], ["inflation risks are elevated."], ["inflation pressures have eased."]]},
                         index=pd.to_datetime(["2005-01-10", "2005-02-01", "2005-03-01"]))
    scores = pd.DataFrame({"doc_id": ["a", "b", "c", "c"],
                           "sentence": ["inflation risks are elevated.", "inflation pressures have eased.",
                                        "growth is strong.", "a conflicted line."],
                           "stance": [0.6, -0.3, -0.8, -5.0]})
    labels = pd.DataFrame({"sentence": ["inflation pressures have eased.", "inflation risks are elevated.",
                                        "growth is strong.", "a conflicted line.", "a conflicted line."],
                           "label": [0, 1, 1, 1, 0], "year": [2005, 2005, 2004, 2005, 2005]})
    pairs, res = sc.stage1_human_labels(table, scores, labels)
    assert pairs["human_shift"].tolist() == [-2.0, 1.0]
    assert pairs["model_shift"].tolist() == pytest.approx([-0.9, 0.3])
    assert res["n_directional"] == 2 and res["agreement"] == 1.0
    assert res["passed"] is False  # fewer than 20 directional pairs
