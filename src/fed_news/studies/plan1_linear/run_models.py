"""Run the Plan 1 ladder on both tasks and write one prediction file per row."""
import argparse
import time

import pandas as pd

from fed_news import config
from fed_news.features.panel import N_COLS
from fed_news.models.walkforward import common_rows, ladder, run_spec


def load_task(task: str) -> pd.DataFrame:
    return common_rows(pd.read_pickle(config.DATA / f"{task.lower()}_full.pkl"))


def select_specs(task: str, no_text: bool):
    specs = ladder(task)
    return [s for s in specs if not any(c in N_COLS for c in s.cols)] if no_text else specs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-text", action="store_true")
    args = ap.parse_args(argv)
    config.RESULTS.mkdir(exist_ok=True)
    for task in ("T1", "T2"):
        df = load_task(task)
        print(f"{task}: {len(df)} common rows, {df.index.min().date()} .. {df.index.max().date()}", flush=True)
        for spec in select_specs(task, args.no_text):
            t0 = time.time()
            preds = run_spec(df, spec, config.TEST_START)
            preds.to_csv(config.RESULTS / f"preds_{task}_{spec.name}.csv", index=False)
            print(f"  {spec.name:8s} {len(preds)} forecasts ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
