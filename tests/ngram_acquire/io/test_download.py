# tests/io/test_download.py
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import List

import pytest

from ngram_acquire.io.download import stream_download_with_retries


class _GoodResp:
    status_code = 200
    def raise_for_status(self):  # no-op
        return None


class _BadResp:
    status_code = 500
    def __init__(self, exc):
        self._exc = exc
    def raise_for_status(self):
        raise self._exc


def test_success_on_first_try(monkeypatch):
    calls: List[dict] = []

    class _Sess:
        def get(self, url, *, stream, timeout):
            calls.append({"url": url, "stream": stream, "timeout": timeout})
            return _GoodResp()

    sess = _Sess()
    resp = stream_download_with_retries(
        "https://example.com/file.gz", session=sess, timeout=12.5
    )
    assert isinstance(resp, _GoodResp)
    assert len(calls) == 1
    assert calls[0]["url"] == "https://example.com/file.gz"
    assert calls[0]["stream"] is True
    assert calls[0]["timeout"] == 12.5


def test_retries_then_success(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)

    # Simulate: HTTP 500 (raises) -> ConnectionError -> success
    import requests

    attempts = {"n": 0}
    def _get(url, *, stream, timeout):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return _BadResp(requests.HTTPError("500"))
        if attempts["n"] == 2:
            raise requests.ConnectionError("boom")
        return _GoodResp()

    class _Sess:
        get = staticmethod(_get)

    # Stub time.sleep to avoid waiting and capture the backoff sequence
    sleeps: List[float] = []
    monkeypatch.setattr("ngram_acquire.io.download.time.sleep", lambda s: sleeps.append(s))

    resp = stream_download_with_retries(
        "https://host/file.gz",
        session=_Sess(),
        max_retries=3,
        delay_seconds=1.0,
        backoff=2.0,
        timeout=4.2,
    )
    assert isinstance(resp, _GoodResp)

    # Two sleeps (between 3 attempts): 1.0, then 2.0
    assert sleeps == [1.0, 2.0]

    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("retrying in 1.0s" in m for m in warns)
    assert any("retrying in 2.0s" in m for m in warns)


def test_exhausts_retries_and_raises(monkeypatch, caplog):
    caplog.set_level(logging.INFO)

    import requests

    class _Sess:
        def get(self, url, *, stream, timeout):
            raise requests.Timeout("slow")

    sleeps: List[float] = []
    monkeypatch.setattr("ngram_acquire.io.download.time.sleep", lambda s: sleeps.append(s))

    with pytest.raises(Exception) as ei:
        stream_download_with_retries(
            "https://host/bad.gz",
            session=_Sess(),
            max_retries=3,
            delay_seconds=0.5,
            backoff=3.0,
            timeout=1.0,
        )
    # Make sure we bubbled a Timeout (or at least *something* from requests)
    assert "slow" in str(ei.value)

    # Slept max_retries-1 times: 0.5, then 1.5
    assert sleeps == [0.5, 1.5]

    errs = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any("Failed to download https://host/bad.gz after 3 attempts" in m for m in errs)


def test_uses_provided_session_and_stream_flag(monkeypatch):
    seen = {"called": 0, "stream": None, "timeout": None}

    class _Sess:
        def get(self, url, *, stream, timeout):
            seen["called"] += 1
            seen["stream"] = stream
            seen["timeout"] = timeout
            return _GoodResp()

    sess = _Sess()
    _ = stream_download_with_retries(
        "https://example.com/x", session=sess, timeout=7.0
    )
    assert seen["called"] == 1
    assert seen["stream"] is True
    assert seen["timeout"] == 7.0


def test_http_error_is_retried(monkeypatch):
    """If raise_for_status() raises, it should be treated as a retryable failure."""
    import requests

    calls = {"n": 0}

    def _get(url, *, stream, timeout):
        calls["n"] += 1
        # First attempt returns a 500-like resp that raises
        if calls["n"] == 1:
            return _BadResp(requests.HTTPError("bad status"))
        # Second attempt succeeds
        return _GoodResp()

    class _Sess:
        get = staticmethod(_get)

    monkeypatch.setattr("ngram_acquire.io.download.time.sleep", lambda s: None)

    resp = stream_download_with_retries(
        "https://example.com/y", session=_Sess(), max_retries=2, delay_seconds=0.1
    )
    assert isinstance(resp, _GoodResp)
    assert calls["n"] == 2
