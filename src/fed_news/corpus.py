"""The Fed's own documents, each with the date it became public (spec section 5, N).

Statements and press conferences are public on the decision day; minutes on their stated
release date; Beige Books on their publication date; speeches on the day delivered. A document
without a usable release date, or with under MIN_DOC_CHARS characters of text, is dropped and
counted. Every text is cached by URL, so a rebuild is offline.
"""
import hashlib
import html as htmllib
import io
import json
import re

import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader

from fed_news import config
from fed_news.fomc_calendar import BASE, load_calendar
from fed_news.http import get

DOC_COLS = ["doc_id", "type", "speaker", "release", "text"]
TYPE_OF_KIND = {"statement": "statement", "minutes": "minutes", "presconf": "presconf", "beige": "beige"}
PRESCONF_PDF = BASE + "/mediacenter/files/FOMCpresconf{:%Y%m%d}.pdf"
SPEECH_JSON = BASE + "/json/ne-speeches.json"
OLD_SPEECH_LIST = BASE + "/newsevents/speech/{}speech.htm"
OLD_SPEECH_LINK = re.compile(
    r'href="((?:https?://www\.federalreserve\.gov)?/boarddocs/speeches/\d{4}/(\d{8})\d*/default\.htm)"', re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
MONTH_DATE = re.compile(r"(January|February|March|April|May|June|July|August|September|October|"
                        r"November|December)\s+(\d{1,2}),\s+(\d{4})")


class CoverageError(RuntimeError):
    pass


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    node = (soup.find(id="article") or soup.find("div", class_="col-xs-12 col-sm-8 col-md-8")
            or soup.body or soup)
    return re.sub(r"\s+", " ", node.get_text(" ")).strip()


def pdf_to_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    return re.sub(r"\s+", " ", " ".join(page.extract_text() or "" for page in reader.pages)).strip()


def split_sentences(text: str) -> list[str]:
    parts = (s.strip() for s in SENT_SPLIT.split(text))
    return [s for s in parts if 30 <= len(s) <= 1000]


def beige_release(text: str, month: pd.Timestamp) -> pd.Timestamp:
    for m in MONTH_DATE.finditer(text):
        day = pd.Timestamp(f"{m.group(1)} {m.group(2)} {m.group(3)}")
        if day.year == month.year and day.month == month.month:
            return day
    return pd.NaT


def _doc_id(doc_type: str, release: pd.Timestamp, url: str) -> str:
    return f"{doc_type}:{release:%Y%m%d}:{hashlib.sha1(url.encode()).hexdigest()[:8]}"


def _text(url: str) -> str:
    return pdf_to_text(get(url, binary=True)) if url.lower().endswith(".pdf") else html_to_text(get(url))


def docs_from_calendar(cal: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, int]:
    rows, dropped = [], 0
    for r in cal[cal["kind"] == kind].itertuples():
        try:
            if kind == "presconf":
                try:
                    text = pdf_to_text(get(PRESCONF_PDF.format(r.date), binary=True))
                except FileNotFoundError:
                    text = _text(r.url)
            else:
                text = _text(r.url)
        except FileNotFoundError:
            dropped += 1
            continue
        release = r.release
        if kind == "beige" and pd.isna(release):
            release = beige_release(text, r.date)
        if pd.isna(release) or len(text) < config.MIN_DOC_CHARS:
            dropped += 1
            continue
        doc_type = TYPE_OF_KIND[kind]
        rows.append({"doc_id": _doc_id(doc_type, release, r.url), "type": doc_type,
                     "speaker": "", "release": pd.Timestamp(release), "text": text})
    return pd.DataFrame(rows, columns=DOC_COLS), dropped


def parse_speech_json(raw: str) -> pd.DataFrame:
    rows = []
    for item in json.loads(raw.lstrip("﻿")):
        link = item.get("l", "")
        if not link or not item.get("d"):
            continue
        rows.append({"url": BASE + link if link.startswith("/") else link,
                     "release": pd.to_datetime(item["d"].split()[0], format="%m/%d/%Y"),
                     "speaker": htmllib.unescape(item.get("s", ""))})
    return pd.DataFrame(rows, columns=["url", "release", "speaker"])


def parse_old_speech_list(page: str) -> pd.DataFrame:
    rows = []
    for link, day in OLD_SPEECH_LINK.findall(page):
        url = link.replace("http://", "https://")
        rows.append({"url": url if url.startswith("https") else BASE + url,
                     "release": pd.to_datetime(day, format="%Y%m%d"), "speaker": ""})
    return pd.DataFrame(rows, columns=["url", "release", "speaker"]).drop_duplicates("url")


def fetch_speeches() -> tuple[pd.DataFrame, int]:
    listing = [parse_speech_json(get(SPEECH_JSON, cache=False))]
    for year in range(1996, 2006):
        try:
            listing.append(parse_old_speech_list(get(OLD_SPEECH_LIST.format(year))))
        except FileNotFoundError:
            continue
    listing = pd.concat(listing, ignore_index=True).drop_duplicates("url")
    rows, dropped = [], 0
    for r in listing.itertuples():
        try:
            text = _text(r.url)
        except FileNotFoundError:
            dropped += 1
            continue
        if len(text) < config.MIN_DOC_CHARS:
            dropped += 1
            continue
        rows.append({"doc_id": _doc_id("speech", r.release, r.url), "type": "speech",
                     "speaker": r.speaker, "release": r.release, "text": text})
    return pd.DataFrame(rows, columns=DOC_COLS), dropped


def coverage(docs: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(docs["release"].dt.year, docs["type"])


def check_coverage(cov: pd.DataFrame, years, required) -> None:
    missing = [f"{year}:{t}" for year in years for t in required
               if t not in cov.columns or year not in cov.index or cov.loc[year, t] == 0]
    if missing:
        raise CoverageError("no documents for " + ", ".join(missing))


def build_corpus(refresh: bool = False) -> pd.DataFrame:
    path = config.DATA / "corpus.pkl"
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    cal = load_calendar()
    parts, dropped = [], {}
    for kind in ("statement", "minutes", "presconf", "beige"):
        docs, n = docs_from_calendar(cal, kind)
        parts.append(docs)
        dropped[kind] = n
        print(f"{kind:9s} kept {len(docs):5d} dropped {n}", flush=True)
    speeches, n = fetch_speeches()
    parts.append(speeches)
    dropped["speech"] = n
    print(f"speech    kept {len(speeches):5d} dropped {n}", flush=True)
    docs = pd.concat(parts, ignore_index=True).drop_duplicates("doc_id")
    docs = docs[docs["release"] >= pd.Timestamp("1993-06-01")].sort_values("release").reset_index(drop=True)
    docs.to_pickle(path)
    config.RESULTS.mkdir(exist_ok=True)
    coverage(docs).to_csv(config.RESULTS / "coverage.csv")
    (config.RESULTS / "dropped_docs.json").write_text(json.dumps(dropped, indent=1))
    return docs


if __name__ == "__main__":
    corpus = build_corpus(refresh=True)
    cov = coverage(corpus)
    print(cov.to_string())
    check_coverage(cov, range(2001, corpus["release"].dt.year.max() + 1), ("statement", "minutes"))
    print("coverage check passed")
