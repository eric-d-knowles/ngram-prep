# tests/io/test_fetch.py
from __future__ import annotations

import re
from pathlib import PurePosixPath

import pytest

from ngram_prep.io.fetch import fetch_file_urls


class FakeResp:
    def __init__(self, text: str, status: int = 200):
        self.text = text
        self.status_code = status

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

    def get(self, url, timeout=30.0):
        self.calls += 1
        self.last_timeout = timeout
        self.last_url = url
        text, status = self.pages.pop(0)
        if isinstance(status, Exception):
            raise status
        return FakeResp(text, status=status)


def test_fetch_file_urls_happy_path(tmp_path):
    html = """
    <a href="eng-us-2-00001-of-00024.gz">1</a>
    <a href="eng-us-2-00002-of-00024.gz">2</a>
    <a href="other.gz">x</a>
    """
    rx = re.compile(r"^eng-us-2-\d{5}-of-\d{5}\.gz$")
    sess = FakeSession([(html, 200)])

    urls = fetch_file_urls(
        "https://example.com/20200217/eng-us/eng-us-2-ngrams_exports.html",
        rx,
        session=sess,
    )

    assert [PurePosixPath(u).name for u in urls] == [
        "eng-us-2-00001-of-00024.gz",
        "eng-us-2-00002-of-00024.gz",
    ]
    assert sess.calls == 1
    assert sess.last_timeout == 30.0


def test_dedup_and_sort():
    html = """
    <a href="eng-us-2-00002-of-00024.gz">2</a>
    <a href="eng-us-2-00001-of-00024.gz">1</a>
    <a href="eng-us-2-00001-of-00024.gz">1-dup</a>
    """
    rx = re.compile(r"^eng-us-2-\d{5}-of-\d{5}\.gz$")
    sess = FakeSession([(html, 200)])

    urls = fetch_file_urls("https://ex/foo.html", rx, session=sess)
    names = [PurePosixPath(u).name for u in urls]
    assert names == ["eng-us-2-00001-of-00024.gz", "eng-us-2-00002-of-00024.gz"]


def test_retries_then_succeeds(monkeypatch):
    html_ok = '<a href="eng-us-1-00000-of-00001.gz">ok</a>'
    rx = re.compile(r"^eng-us-1-\d{5}-of-\d{5}\.gz$")

    class NetErr(Exception):
        pass

    sess = FakeSession([
        ("", NetErr("boom")),  # first call raises
        (html_ok, 200),        # second call ok
    ])

    # Stub sleep to keep tests fast
    import ngram_prep.io.fetch as fetch_mod
    sleeps = []
    monkeypatch.setattr(fetch_mod.time, "sleep", lambda s: sleeps.append(s))

    urls = fetch_file_urls("https://ex/foo.html", rx, session=sess,
                           max_retries=2, delay_seconds=0.1, backoff=3.0)

    assert [PurePosixPath(u).name for u in urls] == ["eng-us-1-00000-of-00001.gz"]
    assert sleeps == [0.1]
    assert sess.calls == 2


def test_exhausts_retries_and_raises(monkeypatch):
    rx = re.compile(r"^eng-us-1-\d{5}-of-\d{5}\.gz$")
    class NetErr(Exception):
        pass
    sess = FakeSession([("", NetErr("down")), ("", NetErr("still down"))])

    import ngram_prep.io.fetch as fetch_mod
    monkeypatch.setattr(fetch_mod.time, "sleep", lambda s: None)

    with pytest.raises(RuntimeError):
        fetch_file_urls("https://ex/foo.html", rx, session=sess, max_retries=2)
