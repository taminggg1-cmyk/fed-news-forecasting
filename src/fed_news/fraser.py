"""FRASER (St. Louis Fed) speech texts for Study 2 (spec 2026-10-10-study2-regional-speeches-design.md).

Item metadata were saved by the count step under D:/fed_news/fraser/meta/items_<titleId>.json. This module
reads them and downloads every textUrl of an item, at most one request every 2.1 seconds (FRASER allows 30 a
minute), caching texts under D:/fed_news/fraser/text/<item_id>.txt. The API key is read from the secrets folder
and never printed or included in errors.
"""
import glob
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

from fed_news import config

META = config.DATA / "fraser" / "meta"
TEXT = config.DATA / "fraser" / "text"
MIN_INTERVAL = 2.1


def api_key() -> str:
    return (config.SECRETS / "fraser_api_key.txt").read_text().strip()


def load_items(meta_dir=None) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(Path(meta_dir or META) / "items_*.json"))):
        tid = re.search(r"items_(\d+)\.json$", f).group(1)
        for r in json.loads(Path(f).read_text(encoding="utf-8")):
            rid = (r.get("recordInfo") or {}).get("recordIdentifier")
            urls = (r.get("location") or {}).get("textUrl") or []
            urls = urls if isinstance(urls, list) else [urls]
            rows.append({"item_id": str(rid[0] if isinstance(rid, list) else rid), "title_id": tid,
                         "date": (r.get("originInfo") or {}).get("sortDate"),
                         "title": ((r.get("titleInfo") or [{}])[0]).get("title", ""),
                         "text_urls": [u for u in urls if isinstance(u, str) and u.startswith("http")]})
    out = pd.DataFrame(rows, columns=["item_id", "title_id", "date", "title", "text_urls"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


class Throttle:
    def __init__(self, interval: float = MIN_INTERVAL):
        self.interval, self.last = interval, 0.0

    def wait(self) -> None:
        gap = self.interval - (time.time() - self.last)
        if gap > 0:
            time.sleep(gap)
        self.last = time.time()


def fetch_text(url: str, session, throttle: Throttle, retries: int = 5) -> str:
    for attempt in range(retries):
        throttle.wait()
        try:
            r = session.get(url, headers={"X-API-Key": api_key()}, timeout=90)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r.content.decode("utf-8", errors="replace")
        if r.status_code == 404:
            return ""
        time.sleep(30 if r.status_code == 429 else 5 * (attempt + 1))
    raise RuntimeError(f"FRASER text download failed after {retries} attempts")


def download_texts(items: pd.DataFrame, session=None, throttle=None, text_dir=None) -> int:
    out_dir = Path(text_dir or TEXT)
    out_dir.mkdir(parents=True, exist_ok=True)
    session, throttle = session or requests.Session(), throttle or Throttle()
    n = 0
    for row in items.itertuples():
        path = out_dir / f"{row.item_id}.txt"
        if path.exists():
            continue
        text = "\n".join(fetch_text(u, session, throttle) for u in row.text_urls)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
        n += 1
        if n % 100 == 0:
            print(f"{n} texts downloaded", flush=True)
    return n
