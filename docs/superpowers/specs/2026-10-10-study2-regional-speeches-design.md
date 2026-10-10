# Study 2 design: does Fed communication move the 2-year yield? A larger, pre-registered event study

Date: 2026-10-10. Status: new confirmatory study after Plan 5 (H5 not supported: Clark-West p = 0.129 on
1,778 events; the full-sample slope was positive, HAC p = 0.056). Binding settings go into
`PREREG_STUDY2.md`, tagged `prereg-study2` after the code is written and before any regional speech text is
scored or any new event-day yield change is computed. Nothing here changes H1, H2a or H5.

## 1. Question and hypothesis

**H6 (confirmatory).** On an event set expanded with the speeches of regional Federal Reserve Bank
presidents, the stance surprise of Fed communication predicts the same-day change in the 2-year Treasury
yield out of sample. Supported if the one-sided Clark-West p < 0.05; void if the same test on the
previous trading day's change gives p < 0.05 (the Plan 5 rules, unchanged).

Why: Plan 5's effect pointed the right way but lacked power (t = 1.13 at n = 1,778). A power calculation at
the same effect size needs about 3,800 events. The expanded set gives about 3,400, and a better-trained
stance classifier (section 4) should reduce measurement noise. Every choice below is fixed before testing;
nothing is selected by looking at yields.

## 2. Data sources (fixed)

1. **Plan 1 corpus**: statements, minutes, press-conference transcripts and Board speeches (2,578
   documents, `D:/fed_news/corpus.pkl`).
2. **FRASER** (St. Louis Fed), series 3762 "Statements and Speeches of Federal Reserve Officials", the
   titles of the 35 regional presidents listed in section 3 (3,969 items, all with dates and full-text
   links; metadata already saved under `D:/fed_news/fraser/meta/`). Texts are downloaded from each item's
   `textUrl` with the FRASER API key (`D:/fed_news/secrets/fraser_api_key.txt`, header `X-API-Key`, at most
   30 requests per minute).
3. **BIS central bankers' speeches** (`D:/fed_news/bis/raw/speeches_*.csv`, downloaded 2026-10-10, through
   2026-06-22): speeches whose description names a Federal Reserve Bank president in section 3, used for
   presidents FRASER lacks (Williams, Daly, Kashkari, Hoenig, Schmid) and for any (speaker, date) FRASER
   lacks.

**Inclusion.** A regional speech is included if its speaker is a president in section 3 and its date year is
within that president's tenure years (inclusive). Release day = the FRASER `sortDate` or the BIS `date`.

**De-duplication.** One document per (speaker surname, date): FRASER text preferred, else BIS. A regional
speech on the same date and by the same surname as a Plan 1 corpus speech is dropped (Yellen as Board member).

**Type.** Regional speeches get type `speech` (so the Plan 5 event code treats them as speeches) and a
`source` column (`board`, `fraser`, `bis`).

## 3. Regional presidents and tenure years (fixed)

Boston: Minehan 1994-2007, Rosengren 2007-2021, Collins 2022-.
New York: McDonough 1993-2003, Geithner 2003-2009, Dudley 2009-2018, Williams 2018-.
Philadelphia: Santomero 2000-2006, Plosser 2006-2015, Harker 2015-2025, Paulson 2025-.
Cleveland: Jordan 1992-2003, Pianalto 2003-2014, Mester 2014-2024, Hammack 2024-.
Richmond: Broaddus 1993-2004, Lacker 2004-2017, Barkin 2018-.
Atlanta: Guynn 1996-2006, Lockhart 2007-2017, Bostic 2017-2025.
Chicago: Moskow 1994-2007, Evans 2007-2023, Goolsbee 2023-.
St. Louis: Poole 1998-2008, Bullard 2008-2023, Musalem 2024-.
Minneapolis: Stern 1985-2009, Kocherlakota 2009-2015, Kashkari 2016-.
Kansas City: Hoenig 1991-2011, George 2011-2023, Schmid 2023-.
Dallas: McTeer 1991-2004, Fisher 2005-2015, Kaplan 2015-2021, Logan 2022-.
San Francisco: Parry 1986-2004, Yellen 2004-2010, Williams 2011-2018, Daly 2018-.

## 4. Stance classifier (fixed)

Same as Plan 1 (per-year ChronoBERT vintage trained on text up to the end of year y-1, three classes,
Trillion Dollar Words + World Central Banks labels dated by their source documents, learning rate 2e-5,
3 epochs, batch 16, length 256, three seeds averaged), with one change: the classifier that scores
documents is trained on **all** eligible labels for its year, instead of 80% (Plan 1 held out 20% for
accuracy reporting). Held-out accuracy is still reported from an 80/20 split, as in Plan 1. Every
document, old and new, is scored again with these classifiers. Sentence stance = P(hawkish) - P(dovish);
document stance = mean over sentences (30-1000 characters).

## 5. Event study (unchanged rules)

The Plan 5 code (`events.build_events`, `event_study.run`) on the new document stance table: events keyed by
outcome trading day; surprise = event stance minus the document-mean stance of earlier events' documents in
the 30 calendar days before; outcome = 2-year yield change in basis points; walk-forward OLS on all earlier
events from 1994; one-sided Clark-West; void rule on the previous trading day. Test window: outcome days
2001-01-01 to 2026-09-30.

**Secondary (reported, not decisive):** events with regional speeches only; the Plan 5 event set (no
regional speeches) re-scored with the new classifier (separates the classifier effect from the data
effect); the full-sample OLS slope with Newey-West errors; the monthly and per-meeting forecasting tests
(Plan 1 primary model, S + new N against S, one-sided Clark-West and DM), with the Fed-stance features
rebuilt from the new document table.

## 6. Code (new modules only; Plans 1, 2 and 5 files unchanged)

| file | responsibility |
|---|---|
| `src/fed_news/fraser.py` | FRASER text download (rate-limited, cached, key never in errors), metadata parsing |
| `src/fed_news/regional_corpus.py` | tenure table, BIS parsing, inclusion, de-duplication, regional document table |
| `src/fed_news/stance2.py` | per-year classifiers on all eligible labels, scoring of all documents, `doc_stance2.pkl` |
| `src/fed_news/study2.py` | H6 event study, secondary tests, forecasting with rebuilt Fed-stance features |
| `src/fed_news/evaluate_study2.py` | `RESULTS_STUDY2.md` with the H6 verdict |

## 7. Order

1. Code and offline tests. 2. `PREREG_STUDY2.md`, tag `prereg-study2`. 3. Download FRASER texts (about 4,000
files, background, rate-limited) and build the regional corpus. 4. Re-train and score (GPU, background).
5. H6 and secondary tests, report. Data hashes of the new inputs are recorded in the report.

## 8. Risks stated in advance

Regional presidents rotate as FOMC voters, so their speeches may carry less information than Board
speeches and could dilute the effect. FRASER coverage thins after 2023. Daily data cannot place
releases within the day, so the test stays conservative for late releases. No significance is guaranteed;
the result is reported whatever it is.
