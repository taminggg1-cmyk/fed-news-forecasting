"""Re-run the linear walk-forward ladders from the saved data panels and compare with the saved predictions.

This is a read-only reproduction check for reviewers. It recomputes, in memory, every forecast of:

- Plan 1, monthly task, original cutoff:   D:/fed_news/t1_full.pkl  -> results/preds_T1_<spec>.csv
- Plan 1, meeting task:                    D:/fed_news/t2_full.pkl  -> results/preds_T2_<spec>.csv
- Post-audit monthly rerun, corrected cutoff: D:/fed_news/t1b_full.pkl -> results/post_audit_t1b/preds_T1_<spec>.csv

and reports, per file, whether the CSV text it would write is byte-identical to the saved file, and the largest
absolute difference in the forecast. Nothing is written to results/. It needs the cached panels under
FED_NEWS_DATA (default D:/fed_news) but no network and no GPU; it takes about 8 minutes on a CPU.

Run from the repository root:  python audit/verify_saved_outputs.py [--quick]
(--quick checks only the monthly ladders, about 2 minutes.)
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fed_news import config  # noqa: E402
from fed_news.models.walkforward import common_rows, ladder, run_spec  # noqa: E402

CHECKS = [
    ("Plan 1 T1, original cutoff", "T1", "t1_full.pkl", config.RESULTS),
    ("Post-audit T1, corrected cutoff", "T1", "t1b_full.pkl", config.RESULTS / "post_audit_t1b"),
    ("Plan 1 T2, meetings", "T2", "t2_full.pkl", config.RESULTS),
]


def compare(saved_path: Path, fresh: pd.DataFrame) -> tuple[bool, float]:
    """Byte comparison of the CSV text (line endings normalised), and the largest forecast difference with the
    saved file parsed exactly (float_precision='round_trip'; pandas' default parser can be off by one ulp)."""
    saved_text = saved_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    identical = saved_text == fresh.to_csv(index=False, lineterminator="\n")
    saved = pd.read_csv(saved_path, parse_dates=["date"], float_precision="round_trip")
    same_rows = saved["date"].equals(fresh["date"])
    diff = float((saved["y_pred"] - fresh["y_pred"]).abs().max()) if same_rows else float("inf")
    return identical, diff


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="monthly ladders only")
    args = ap.parse_args(argv)
    failures = 0
    for label, task, panel, out_dir in CHECKS:
        if args.quick and task == "T2":
            continue
        df = common_rows(pd.read_pickle(config.DATA / panel))
        print(f"{label}: {panel}, {len(df)} rows", flush=True)
        for spec in ladder(task):
            t0 = time.time()
            fresh = run_spec(df, spec, config.TEST_START)
            identical, diff = compare(out_dir / f"preds_{task}_{spec.name}.csv", fresh)
            ok = identical or diff == 0.0
            failures += not ok
            print(f"  {spec.name:9s} n={len(fresh):3d}  byte-identical={identical!s:5s}  max|diff|={diff:.3g}  "
                  f"{'OK' if ok else 'MISMATCH'}  ({time.time() - t0:.0f}s)", flush=True)
    print("all saved linear predictions reproduced" if not failures else f"{failures} file(s) differ")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
