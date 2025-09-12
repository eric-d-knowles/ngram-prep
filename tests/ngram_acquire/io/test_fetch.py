# tests/io/test_fetch.py
from __future__ import annotations

import re
from pathlib import PurePosixPath

import pytest

from ngram_acquire.io.fetch import fetch_file_urls


class FakeResp:
    def __init__(self, text: str, status: int = 200, url: str = ""):
        self.text = text
        self.status_code = status
        self.url = url
        # Simulate requests.Response encoding handling
        self.encoding = None
        self.apparent_encoding = "utf-8"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, pages):
        # pages: list[tuple[text, status_or_exc]]
        self.pages = pages
        self.calls = 0
        self.last_timeout = None
        self.last_url = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, timeout=(10, 30)):
        self.calls += 1
        self.last_timeout = timeout
        self.last_url = url
        text, status = self.pages.pop(0)
        if isinstance(status, Exception):
            raise status
        return FakeResp(text, status=status, url=url)


def test_fetch_file_urls_happy_path(monkeypatch, tmp_path):
    html = """
    <a href="2-00001-of-00024.gz">1</a>
    <a href="2-00002-of-00024.gz">2</a>
    <a href="other.gz">x</a>
    """
    # Match filenames of the form "<n>-00000-of-00024.gz" (no corpus prefix)
    rx = re.compile(r"2-\d{5}-of-\d{5}\.gz$")

    # Patch the Session used inside fetch module
    import ngram_acquire.io.fetch as fetch_mod
    pages = [(html, 200)]
    monkeypatch.setattr(fetch_mod.requests, "Session", lambda: FakeSession(pages))

    urls = fetch_file_urls(
        "https://example.com/20200217/eng/2-ngrams_exports.html",
        rx,
    )

    assert [PurePosixPath(u).name for u in urls] == [
        "2-00001-of-00024.gz",
        "2-00002-of-00024.gz",
    ]




def test_retries_then_succeeds(monkeypatch):
    html_ok = '<a href="1-00000-of-00001.gz">ok</a>'
    rx = re.compile(r"1-\d{5}-of-\d{5}\.gz$")

    class NetErr(Exception):
        pass

    pages = [
        ("", NetErr("boom")),  # first call raises
        (html_ok, 200),        # second call ok
    ]

    # Patch sleep to keep tests fast and observable
    import ngram_acquire.io.fetch as fetch_mod
    sleeps = []
    monkeypatch.setattr(fetch_mod.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(fetch_mod.requests, "Session", lambda: FakeSession(pages))

    urls = fetch_file_urls("https://ex/foo.html", rx, max_retries=2, delay=0.1, backoff=3.0)

    assert [PurePosixPath(u).name for u in urls] == ["1-00000-of-00001.gz"]
    assert sleeps == [0.1]


def test_exhausts_retries_and_raises(monkeypatch):
    rx = re.compile(r"1-\d{5}-of-\d{5}\.gz$")

    class NetErr(Exception):
        pass

    pages = [("", NetErr("down")), ("", NetErr("still down"))]

    import ngram_acquire.io.fetch as fetch_mod
    monkeypatch.setattr(fetch_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(fetch_mod.requests, "Session", lambda: FakeSession(pages))

    with pytest.raises(RuntimeError):
        fetch_file_urls("https://ex/foo.html", rx, max_retries=2)
