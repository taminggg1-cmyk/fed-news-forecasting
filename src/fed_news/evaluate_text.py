"""Plan 5 report: RESULTS_TEXT.md (event study verdict, look-ahead comparison, market benchmark, ablation)."""
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


def main() -> None:
    text_dir = config.RESULTS / "text"
    es = text_dir / "event_study.csv"
    verdict = pd.read_csv(es)["verdict"].iloc[0] if es.exists() else "not run"
    model = (text_dir / "naive_model.txt").read_text(encoding="utf-8").strip() if (text_dir / "naive_model.txt").exists() else "not run"
    lines = [
        "# Results: Plan 5 (text as a forecasting method)", "",
        f"**H5 verdict: {verdict}.**", "",
        "H5 (pre-registered in PREREG_TEXT.md, tag prereg-text): the Fed stance surprise predicts the same-day "
        "2-year Treasury yield change out of sample (one-sided Clark-West, p < 0.05); void if the same test on the "
        "day-before change gives p < 0.05. It does not change the H1 or H2a verdicts.", "",
        "## Event study (confirmatory)", "", _table("event_study.csv"), "",
        "## Look-ahead comparison (exploratory, added after results)", "",
        f"Naive model: {model}, fine-tuned on all labels at once (deliberate look-ahead).", "",
        _table("naive_comparison.csv"), "", "Event study with the naive stance:", "", _table("naive_event_study.csv"), "",
        "## Market benchmark (descriptive)", "", _table("market_benchmark.csv"), "",
        "## Ablation (exploratory, added after results)", "", _table("ablation.csv"), "",
        "## Provenance", "",
        f"- {TAG} tag commit: {_git('rev-list', '-n', '1', TAG) or 'no tag'}",
        f"- HEAD: {_git('rev-parse', '--short', 'HEAD')}", "",
    ]
    (config.REPO / "RESULTS_TEXT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:4]))


if __name__ == "__main__":
    main()
