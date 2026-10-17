"""Plan 5 report: RESULTS_TEXT.md (event study verdict, look-ahead comparison, market benchmark, ablation)."""
import hashlib
import subprocess

import pandas as pd

from fed_news import config

TAG = "prereg-text"


def _git(*args) -> str:
    out = subprocess.run(["git", "-C", str(config.REPO), *args], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else ""


def _table(name: str) -> str:
    path = config.RESULTS / "text" / name
    if not path.exists():
        return "not run"
    return pd.read_csv(path).round(4).to_markdown(index=False)


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else ""


def overlap_share(events_csv, y2_pickle) -> str:
    """Share of test events whose previous trading day is also an event day, as 'NN%' (empty if unavailable)."""
    if not (events_csv.exists() and y2_pickle.exists()):
        return ""
    from fed_news.studies.plan5_event_study.events import TEST_END, TEST_START
    days = pd.to_datetime(pd.read_csv(events_csv)["day"])
    trading = pd.read_pickle(y2_pickle)["y2"].dropna().index
    test = days[(days >= pd.Timestamp(TEST_START)) & (days <= pd.Timestamp(TEST_END))]
    pos = trading.searchsorted(test.to_numpy())
    prev = trading[pos - 1]
    return f"{100 * prev.isin(days).mean():.0f}%"


def stance_corr(a_pkl, b_pkl) -> str:
    if not (a_pkl.exists() and b_pkl.exists()):
        return ""
    a, b = pd.read_pickle(a_pkl), pd.read_pickle(b_pkl)
    m = a[["doc_id", "stance"]].merge(b[["doc_id", "stance"]], on="doc_id").dropna()
    return f"{m['stance_x'].corr(m['stance_y']):.2f}" if len(m) > 2 else ""


def main() -> None:
    text_dir = config.RESULTS / "text"
    es = text_dir / "event_study.csv"
    verdict = pd.read_csv(es)["verdict"].iloc[0] if es.exists() else "not run"
    model = (text_dir / "naive_model.txt").read_text(encoding="utf-8").strip() if (text_dir / "naive_model.txt").exists() else "not run"
    share = overlap_share(text_dir / "events.csv", config.DATA / "market_daily.pkl")
    overlap = (f"About {share} of test events follow an event on the previous trading day, so the void-rule "
               "outcome there is the previous event's own outcome." if share else
               "Some test events follow an event on the previous trading day, so the void-rule outcome there is "
               "the previous event's own outcome.")
    corr = stance_corr(config.DATA / "doc_stance.pkl", config.DATA / "doc_stance_naive.pkl")
    corr_txt = f" Its stance correlates {corr} with ChronoBERT's." if corr else ""
    hashes = [f"- sha256 (first 16 hex) {n}: {_sha(config.DATA / n)}" for n in
              ("doc_stance.pkl", "doc_stance_naive.pkl", "market_bill.pkl") if (config.DATA / n).exists()]
    lines = [
        "# Results: Plan 5 (text as a forecasting method)", "",
        f"**H5 verdict: {verdict}.**", "",
        "H5 (pre-registered in PREREG_TEXT.md, tag prereg-text): the Fed stance surprise predicts the same-day "
        "2-year Treasury yield change out of sample (one-sided Clark-West, p < 0.05); void if the same test on the "
        "day-before change gives p < 0.05. It does not change the H1 or H2a verdicts.", "",
        "## Event study (confirmatory)", "", _table("event_study.csv"), "",
        "The daily test is conservative for releases after the 2-year yield's afternoon quote (late speeches, the "
        "end of press conferences): those are matched to a yield fixed before they happened, which biases the test "
        "toward no effect.", "", overlap, "",
        "Method details: press conferences are grouped with statements; MIN_TRAIN never binds; the first events "
        "(no earlier trading day pair) are dropped; zero-change days are excluded from sign accuracy.", "",
        "## Look-ahead comparison (exploratory, added after results)", "",
        "Direct tests of the naive pipeline against ChronoBERT on the same dates (column dm_p_naive_vs_chrono: "
        "DM-HLN p that the naive forecast is better; Clark-West is not meaningful because the models are not nested):",
        "", _table("naive_comparison.csv"), "",
        "Event study, both pipelines on their common test days (exploratory, added after results; MSE ratio = large "
        "model over small model):", "", _table("lookahead_event_compare.csv"), "",
        "The naive pipeline differs in at least four ways: the language model (one ModernBERT instead of 27 "
        "ChronoBERT vintages), label dating (all labels at once), label count (3,074 against 418-2,438 after the "
        f"20% hold-out) and overlap (its training sentences come from the documents it scores).{corr_txt} The "
        "comparison illustrates a typical pipeline; it does not isolate look-ahead, and the gap is not "
        "statistically significant.", "",
        f"Naive model: {model}, fine-tuned on all labels at once (deliberate look-ahead).", "",
        "Event study with the naive stance (exploratory, added after results; no H5 verdict, which applies only to "
        "the pre-registered ChronoBERT pipeline):", "", _table("naive_event_study.csv"), "",
        "## Market benchmark (descriptive)", "",
        "Descriptive. Headline row: the 3-month bill on the full sample. Its DM p-values are not tests. The "
        "expanding-mean spread drifts over time (the bill-to-rate gap shrank from the 1990s to the 2020s), so the "
        "bill forecast is biased upward and is a weak benchmark; do not read this table as text beating the market.",
        "", _table("market_benchmark.csv"), "",
        "## Ablation (exploratory, added after results)", "",
        "Unadjusted p-values from 12 exploratory comparisons; descriptive only.", "", _table("ablation.csv"), "",
        "## Provenance", "",
        f"- {TAG} tag commit: {_git('rev-list', '-n', '1', TAG) or 'no tag'}",
        f"- HEAD: {_git('rev-parse', '--short', 'HEAD')}", *hashes,
        f"- naive_model.txt: {model}", "",
    ]
    config.report_path("RESULTS_TEXT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:4]))


if __name__ == "__main__":
    main()
