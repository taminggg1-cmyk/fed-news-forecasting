"""RESULTS_STUDY3.md from results/study3/ (Study 3, H7; PREREG_STUDY3.md)."""
import hashlib

import pandas as pd

from fed_news import config
from fed_news.decision_text import OSF_NAME, OUT_PATH, RESULTS, TAG
from fed_news.evaluate import _git, metrics
from fed_news.study3 import decide_h7, power_sentence

MODELS = ("S", "S+D", "placebo", "shuffled-text", "S+M", "S+M+D", "S+N")


def sha16(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else "missing"


def report(results=RESULTS, repo=config.REPO, data_path=OUT_PATH) -> str:
    tests = pd.read_csv(results / "tests.csv")
    p = tests.set_index("test")["p"]
    verdict = decide_h7(p["H7 Clark-West"], p["placebo Clark-West"], p["shuffled-text Clark-West"])
    table = pd.DataFrame([{"model": m, **metrics(pd.read_csv(results / f"preds_T2_{m}.csv", parse_dates=["date"]))}
                          for m in MODELS])
    quality = pd.read_csv(results / "probe_quality.csv")
    power = pd.read_csv(results / "power.csv").iloc[0].to_dict()
    osf_path = repo / OSF_NAME
    osf = osf_path.read_text(encoding="utf-8").strip() if osf_path.exists() else "missing"
    tag = _git(repo, "rev-parse", "--verify", "--quiet", f"refs/tags/{TAG}^{{commit}}") or "no tag"
    lines = [
        "# Results: Study 3 (ChronoBERT trained on Fed decisions, H7)", "",
        f"**H7 verdict: {verdict}.**", "",
        "H7 (PREREG_STUDY3.md): on the 2001-2026 FOMC meetings, the penalised ordered logit with the 16 real-time "
        "statistics plus the decision-trained text signal (S+D) forecasts the target rate better than the same "
        "model with the statistics only (S): one-sided Clark-West, p < 0.05; void if the placebo or the "
        "shuffled-label text model passes the same test.", "",
        "## Tests", "", tests.round(4).to_markdown(index=False), "",
        "## Forecast metrics (same rows for every model)", "", table.round(4).to_markdown(index=False), "",
        "## Text model quality (out of sample, per release year)", "", quality.round(4).to_markdown(index=False), "",
        "## Power (registered before the run)", "", power_sentence(power), "",
        "## Disclosures", "",
        "- The 2001-2026 meetings were already used to evaluate earlier models in this repository.",
        "- Documents released before 2000 are embedded with the 1999 ChronoBERT vintage (training rows only).",
        "- Document labels are noisy: most documents precede a hold, and the next decision depends mostly on data.",
        "- The code and analyses were produced with an AI coding agent (Claude Code) under the author's direction.", "",
        "## Provenance", "",
        f"- {TAG} tag commit: {tag}",
        f"- OSF registration: {osf}",
        f"- HEAD: {_git(repo, 'rev-parse', 'HEAD')}",
        f"- sha256 (first 16 hex) doc_decision.pkl: {sha16(data_path)}", ""]
    return "\n".join(lines)


def main() -> None:
    text = report()
    (config.REPO / "RESULTS_STUDY3.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
