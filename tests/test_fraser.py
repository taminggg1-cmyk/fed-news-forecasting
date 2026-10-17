import json

import pandas as pd

from fed_news.collect import fraser


def item(item_id, date, urls, title="Speech"):
    return {"titleInfo": [{"title": title}], "originInfo": {"sortDate": date},
            "recordInfo": {"recordIdentifier": [str(item_id)]}, "location": {"textUrl": urls}}


def test_load_items(tmp_path):
    (tmp_path / "items_7161.json").write_text(json.dumps([item(1, "2015-01-16", ["https://x/a.txt", "https://x/b.txt"]),
                                                         item(2, "bad", [])]), encoding="utf-8")
    out = fraser.load_items(tmp_path)
    assert list(out["item_id"]) == ["1", "2"] and list(out["title_id"]) == ["7161", "7161"]
    assert out["date"].iloc[0] == pd.Timestamp("2015-01-16") and pd.isna(out["date"].iloc[1])
    assert out["text_urls"].iloc[0] == ["https://x/a.txt", "https://x/b.txt"]


class FakeResp:
    def __init__(self, code, body=b""):
        self.status_code, self.content = code, body


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, headers=None, timeout=None):
        self.calls.append((url, headers))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_download_joins_urls_and_skips_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(fraser, "api_key", lambda: "SECRET")
    items = pd.DataFrame({"item_id": ["1", "2"], "text_urls": [["u1", "u2"], ["u3"]]})
    (tmp_path / "2.txt").write_text("cached", encoding="utf-8")
    s = FakeSession([FakeResp(200, b"part one"), FakeResp(200, b"part two")])
    n = fraser.download_texts(items, session=s, throttle=fraser.Throttle(0), text_dir=tmp_path)
    assert n == 1 and (tmp_path / "1.txt").read_text(encoding="utf-8") == "part one\npart two"
    assert [c[0] for c in s.calls] == ["u1", "u2"] and s.calls[0][1] == {"X-API-Key": "SECRET"}


def test_fetch_text_404_is_empty_and_failure_hides_key(monkeypatch):
    monkeypatch.setattr(fraser, "api_key", lambda: "SECRET")
    assert fraser.fetch_text("u", FakeSession([FakeResp(404)]), fraser.Throttle(0)) == ""
    import requests
    s = FakeSession([requests.ConnectionError("boom SECRET")] * 2)
    monkeypatch.setattr(fraser.time, "sleep", lambda *_: None)
    try:
        fraser.fetch_text("u", s, fraser.Throttle(0), retries=2)
    except RuntimeError as err:
        assert "SECRET" not in str(err)
    else:
        raise AssertionError("must raise")
