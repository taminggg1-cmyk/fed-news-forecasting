"""Study 2 report: RESULTS_STUDY2.md (H6 verdict, secondary event sets, forecasting, provenance)."""
import hashlib
import subprocess

import pandas as pd

from fed_news import config

TAG = "prereg-study2"


def _git(*args) -> str:
    out = subprocess.run(["git", "-C", str(config.REPO), *args], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else ""


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else "missing"


def _table(name: str) -> str:
    path = config.RESULTS / "study2" / name
    return pd.read_csv(path).round(4).to_markdown(index=False) if path.exists() else "not run"


def main() -> None:
    h6 = config.RESULTS / "study2" / "h6.csv"
    verdict = pd.read_csv(h6)["verdict"].iloc[0] if h6.exists() else "not run"
    lines = [
        "# Results: Study 2 (regional Fed speeches, H6)", "",
        f"**H6 verdict: {verdict}.**", "",
        "H6 (PREREG_STUDY2.md, tag prereg-study2): on the event set expanded with regional Federal Reserve Bank "
        "presidents' speeches, the stance surprise predicts the same-day 2-year Treasury yield change out of sample "
        "(one-sided Clark-West, p < 0.05); void if the same test on the previous trading day's change gives p < 0.05.", "",
        "## H6 (confirmatory)", "", _table("h6.csv"), "",
        "The daily test is conservative for releases after the 2-year yield's afternoon quote: those are matched to a "
        "yield fixed before they happened, which biases the test toward no effect.", "",
        "## Secondary event sets (reported, not decisive)", "", _table("secondary_events.csv"), "",
        "## Forecasting with rebuilt Fed-stance features (secondary)", "", _table("forecasting.csv"), "",
        "## Provenance", "",
        f"- {TAG} tag commit: {_git('rev-list', '-n', '1', TAG) or 'no tag'}",
        f"- HEAD: {_git('rev-parse', '--short', 'HEAD')}",
        f"- sha256 (first 16 hex) regional_corpus.pkl: {_sha(config.DATA / 'regional_corpus.pkl')}",
        f"- sha256 (first 16 hex) doc_stance2.pkl: {_sha(config.DATA / 'doc_stance2.pkl')}", "",
    ]
    (config.REPO / "RESULTS_STUDY2.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:4]))


if __name__ == "__main__":
    main()
