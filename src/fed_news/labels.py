"""Hawkish / dovish / neutral sentence labels (spec section 5).

Trillion Dollar Words (gtfintechlab/fomc_communication, 2,480 sentences, label 0 dovish,
1 hawkish, 2 neutral, with a year) plus the 1,000 Federal Reserve sentences of the World
Central Banks set (gtfintechlab/federal_reserve_system; the configs 5768/78516/944601 are three
splits of the same sentences, so one config's train+val+test is the full set).
"""
import io

import pandas as pd

from fed_news import config
from fed_news.http import get

LABELS = {"dovish": 0, "hawkish": 1, "neutral": 2}
TDW = "https://huggingface.co/datasets/gtfintechlab/fomc_communication/resolve/main/{}.csv"
WCB = ("https://huggingface.co/datasets/gtfintechlab/federal_reserve_system/resolve/main/"
       "5768/{}-00000-of-00001.parquet")


def load_labels(refresh: bool = False) -> pd.DataFrame:
    path = config.DATA / "labels.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    tdw = pd.concat([pd.read_csv(io.StringIO(get(TDW.format(s)))) for s in ("train", "test")])
    tdw = pd.DataFrame({"sentence": tdw["sentence"], "label": tdw["label"].astype(int),
                        "year": tdw["year"].astype(int), "source": "tdw"})
    wcb = pd.concat([pd.read_parquet(io.BytesIO(get(WCB.format(s), binary=True)))
                     for s in ("train", "val", "test")])
    wcb = wcb[wcb["stance_label"].isin(LABELS)]
    wcb = pd.DataFrame({"sentence": wcb["sentences"], "label": wcb["stance_label"].map(LABELS),
                        "year": wcb["year"].astype(int), "source": "wcb"})
    out = pd.concat([tdw, wcb], ignore_index=True)
    out["sentence"] = out["sentence"].astype(str).str.lower().str.strip()
    out = out.drop_duplicates("sentence").reset_index(drop=True)
    out.to_pickle(path)
    return out
